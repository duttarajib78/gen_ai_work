import tempfile
import unittest
from pathlib import Path

from audit_agent.deadline import Deadline
from audit_agent.engine import AuditEngine
from audit_agent.errors import GrouperUnavailableError, LlmThrottledError
from audit_agent.reference_data import ReferenceData
from audit_agent.service import AuditService
from tests.fakes import PolicyLlm, ScriptedLlm, StubGrouper, make_reference_dir, make_settings, sample_payload


class ServiceScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ref = ReferenceData.from_directory(make_reference_dir(Path(cls._tmp.name)))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def service(self, llm, grouper=None, **env):
        settings = make_settings(**env)
        engine = AuditEngine(reference=self.ref, llm=llm, grouper=grouper, settings=settings)
        return AuditService(settings=settings, engine=engine, reference=self.ref)

    def run_audit(self, service, payload=None):
        payload = payload or sample_payload()
        return service.audit(service.validate(payload), Deadline(50))

    # ---- no alert ------------------------------------------------------------
    def test_all_diagnoses_supported_means_no_alert_and_one_model_call(self):
        llm, grouper = PolicyLlm(), StubGrouper()
        res = self.run_audit(self.service(llm, grouper))
        b = res.body
        self.assertEqual((res.status_code, b["status"], b["disposition"]), (200, "completed", "no_alert"))
        self.assertFalse(b["potential_alert"] or b["flagged"])
        self.assertIsNone(b["alert"])
        self.assertEqual(len(b["verdicts"]), 4)
        self.assertEqual([v["role"] for v in b["verdicts"]], ["PDx", "ADx", "ADx", "ADx"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(grouper.calls, [])
        self.assertEqual(b["usage"]["llm_calls"], 1)
        self.assertEqual(b["usage"]["input_tokens"], 1200)
        prov = b["provenance"]
        self.assertEqual(prov["model_id"], "global.anthropic.claude-sonnet-5-5")
        self.assertEqual(prov["prompt_version"], "pdx-adx-support/2026-10-15.1")
        self.assertEqual(prov["low_confidence_weak_threshold"], 0.5)

    def test_confident_weak_passes_but_low_confidence_weak_fails(self):
        llm = PolicyLlm({"I509": ("weak", 0.9), "E119": ("weak", 0.5)})
        b = self.run_audit(self.service(llm, StubGrouper())).body
        self.assertFalse(b["potential_alert"])
        llm = PolicyLlm({"I509": ("weak", 0.49)})
        b = self.run_audit(self.service(llm, StubGrouper())).body
        self.assertTrue(b["potential_alert"])
        self.assertEqual(b["failed_adx_codes"], ["I509"])
        self.assertEqual(next(v for v in b["verdicts"] if v["code"] == "I509")["flag_reason"], "low_confidence_weak")

    def test_threshold_is_configurable_and_reported(self):
        llm = PolicyLlm({"I509": ("weak", 0.6)})
        b = self.run_audit(self.service(llm, StubGrouper(), AUDIT_LOW_CONFIDENCE_WEAK_THRESHOLD="0.7")).body
        self.assertTrue(b["potential_alert"])
        self.assertEqual(b["provenance"]["low_confidence_weak_threshold"], 0.7)

    # ---- ADx-only alert ------------------------------------------------------
    def test_failed_adx_is_removed_without_a_replacement_search(self):
        llm, grouper = PolicyLlm({"N179": ("none", 0.85)}), StubGrouper()
        b = self.run_audit(self.service(llm, grouper)).body
        self.assertEqual((b["disposition"], b["flagged"]), ("alert", True))
        self.assertEqual(len(llm.calls), 1)  # no replacement search: the PDx is fine
        self.assertEqual(grouper.calls, [{"removed": {"N179"}, "resequence": None}])
        self.assertIsNone(b["correction"]["pdx_resequence"])
        fields = b["correction"]["corrected_diagnosis_fields"]
        self.assertEqual(fields["diagnosis_code_array"], ["J189", "I509", "E119"])
        self.assertEqual(fields["diagnosis_poa_array"], [1, 1, 1])
        self.assertEqual(b["alert"]["claim_seqno"], "1001")
        self.assertEqual(b["alert"]["recovery_amount"], 1971)
        self.assertEqual(b["alert"]["alert_instance_id"], 62248)
        self.assertEqual(set(b["alert"]), {"claim_seqno", "alert_info", "recovery_amount", "alert_instance_id"})
        self.assertIn("ADx N179", b["alert"]["alert_info"])
        self.assertEqual(b["regroup"]["regrouped_drg"], "E62A")

    # ---- PDx failures --------------------------------------------------------
    def test_failed_pdx_is_resequenced_to_the_models_choice(self):
        llm, grouper = PolicyLlm({"J189": ("none", 0.9)}, replacement="I509"), StubGrouper()
        b = self.run_audit(self.service(llm, grouper)).body
        self.assertEqual(b["disposition"], "alert")
        self.assertEqual(len(llm.calls), 2)
        seq = b["correction"]["pdx_resequence"]
        self.assertEqual((seq["old_pdx_code"], seq["new_principal_code"], seq["mechanical_fallback"]), ("J189", "I509", False))
        self.assertEqual(b["correction"]["corrected_diagnosis_fields"]["diagnosis_code_array"], ["I509", "N179", "E119", "J189"])
        self.assertIn("model-reasoned", b["alert"]["alert_info"])

    def test_no_clinical_fit_falls_back_mechanically_to_the_first_surviving_adx(self):
        llm = PolicyLlm({"J189": ("none", 0.9), "I509": ("none", 0.9)}, replacement=None)
        b = self.run_audit(self.service(llm, StubGrouper())).body
        seq = b["correction"]["pdx_resequence"]
        self.assertEqual((seq["new_principal_code"], seq["mechanical_fallback"]), ("N179", True))  # I509 failed, so N179 is first to survive
        self.assertEqual(b["correction"]["removed_adx_codes"], ["I509"])
        self.assertIn("MECHANICALLY", b["alert"]["alert_info"])

    def test_replacement_candidates_exclude_failed_adx(self):
        llm = PolicyLlm({"J189": ("none", 0.9), "N179": ("none", 0.9)}, replacement="I509")
        self.run_audit(self.service(llm, StubGrouper()))
        candidates = llm.calls[1]["user_text"].split("Candidate additional diagnoses")[1]
        self.assertIn("I509", candidates)
        self.assertIn("E119", candidates)
        self.assertNotIn("N179", candidates)

    def test_failed_pdx_with_no_surviving_adx_is_dropped_without_spending_a_call(self):
        for extra in ({"diagnosis_code_array": ["J189"], "diagnosis_poa_array": None, "condition_onset_array": None,
                       "dx_code_idx_array": None}, {}):
            llm, grouper = PolicyLlm({c: ("none", 0.9) for c in ("J189", "I509", "N179", "E119")}), StubGrouper()
            b = self.run_audit(self.service(llm, grouper), sample_payload(**extra)).body
            self.assertEqual((b["status"], b["disposition"], b["flagged"]), ("completed", "dropped_no_surviving_adx", False))
            self.assertIsNone(b["correction"])
            self.assertTrue(any("no_surviving_adx" in w for w in b["warnings"]))
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(grouper.calls, [])

    # ---- modes ---------------------------------------------------------------
    def test_verdicts_only_needs_no_grouper_fields_and_makes_no_regroup_call(self):
        llm, grouper = PolicyLlm({"J189": ("none", 0.9)}), StubGrouper()
        payload = {"episode": {"admission_no": 7, "diagnosis_code_array": ["J189", "I509"]}, "options": {"mode": "verdicts_only"}}
        b = self.run_audit(self.service(llm, grouper), payload).body
        self.assertEqual((b["disposition"], b["potential_alert"], b["flagged"]), ("verdicts_only", True, False))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(grouper.calls, [])
        self.assertIsNone(b["alert"])

    def test_grouper_disabled_returns_the_proposed_correction_only(self):
        llm = PolicyLlm({"N179": ("none", 0.9)})
        b = self.run_audit(self.service(llm, grouper=None)).body
        self.assertEqual((b["disposition"], b["flagged"]), ("correction_proposed", False))
        self.assertEqual(b["correction"]["removed_adx_codes"], ["N179"])
        self.assertIsNone(b["alert"])
        self.assertIsNone(b["regroup"])
        self.assertTrue(any("grouper_disabled" in w for w in b["warnings"]))

    def test_episode_text_is_returned_only_on_request(self):
        s = self.service(PolicyLlm())
        self.assertNotIn("episode_text", self.run_audit(s).body)
        self.assertTrue(self.run_audit(s, sample_payload(options={"include_episode_text": True})).body["episode_text"].startswith("EPISODE"))

    # ---- failures ------------------------------------------------------------
    def test_unusable_model_output_is_an_episode_level_failure_with_http_200(self):
        bad = {"verdicts": []}
        res = self.run_audit(self.service(ScriptedLlm(bad, bad, bad), StubGrouper()))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.body["status"], "failed")
        self.assertEqual(res.body["error"]["code"], "llm_output_invalid")
        self.assertFalse(res.body["error"]["retryable"])
        self.assertEqual(res.body["usage"]["llm_calls"], 3)

    def test_model_throttling_is_a_retryable_5xx(self):
        res = self.run_audit(self.service(ScriptedLlm(LlmThrottledError("slow down")), StubGrouper()))
        self.assertEqual(res.status_code, 503)
        self.assertTrue(res.body["error"]["retryable"])
        self.assertIsNone(res.body["partial"])

    def test_late_failure_still_returns_the_verdicts_already_paid_for(self):
        llm = PolicyLlm({"N179": ("none", 0.9)})
        res = self.run_audit(self.service(llm, StubGrouper(fail_with=GrouperUnavailableError("down"))))
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.body["error"]["code"], "grouper_unavailable")
        self.assertEqual(len(res.body["partial"]["verdicts"]), 4)
        self.assertTrue(res.body["partial"]["potential_alert"])

    def test_ambiguous_correction_is_routed_to_a_human_coder(self):
        payload = sample_payload(diagnosis_code_array=["J189", "I509", "J189"], diagnosis_poa_array=None,
                                 condition_onset_array=None, dx_code_idx_array=None)
        llm = PolicyLlm({"J189": ("none", 0.9)}, replacement="I509")
        res = self.run_audit(self.service(llm, StubGrouper()), payload)
        self.assertEqual((res.status_code, res.body["error"]["code"]), (200, "correction_not_applicable"))
        self.assertEqual(len(res.body["partial"]["verdicts"]), 2)

    def test_unexpected_exceptions_do_not_leak_details(self):
        res = self.run_audit(self.service(ScriptedLlm(RuntimeError("internal path /srv/secret")), StubGrouper()))
        self.assertEqual((res.status_code, res.body["error"]["code"]), (500, "internal_error"))
        self.assertNotIn("secret", str(res.body))


if __name__ == "__main__":
    unittest.main()
