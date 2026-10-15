import unittest

from audit_agent.alerts import _support_headline, build_alert_info, build_alert_record, get_claim_seqno
from audit_agent.errors import CorrectionError
from audit_agent.grouper.corrections import apply_diagnosis_corrections, rows_to_diag_poa, rows_to_full_diagnosis_fields
from audit_agent.grouper.interface import RegroupResult
from audit_agent.llm.verdicts import CodeVerdict
from audit_agent.schemas import Episode, parse_request
from tests.fakes import make_settings, sample_payload

S = make_settings()


def episode(**overrides):
    return parse_request(sample_payload(**overrides), S).episode


class CorrectionTests(unittest.TestCase):
    def test_removing_an_adx_trims_every_paired_array_at_the_same_index(self):
        rows = apply_diagnosis_corrections(episode(), {"N179"})
        fields = rows_to_full_diagnosis_fields(rows)
        self.assertEqual(fields["diagnosis_code_array"], ["J189", "I509", "E119"])
        self.assertEqual(fields["diagnosis_poa_array"], [1, 1, 1])      # N179's poa (0) dropped with it
        self.assertEqual(fields["condition_onset_array"], ["ADMIT", "ADMIT", "ADMIT"])
        self.assertEqual(fields["dx_code_idx_array"], [0, 1, 3])

    def test_resequence_promotes_new_principal_and_demotes_old_to_the_end(self):
        rows = apply_diagnosis_corrections(episode(), set(), ("J189", "I509"))
        fields = rows_to_full_diagnosis_fields(rows)
        self.assertEqual(fields["diagnosis_code_array"], ["I509", "N179", "E119", "J189"])
        self.assertEqual(fields["condition_onset_array"], ["ADMIT", "DURING", "ADMIT", "ADMIT"])
        self.assertEqual(fields["dx_code_idx_array"], [1, 2, 3, 0])  # each index travels with its own diagnosis

    def test_removal_and_resequence_combine(self):
        rows = apply_diagnosis_corrections(episode(), {"N179"}, ("J189", "E119"))
        self.assertEqual([r["code"] for r in rows], ["E119", "I509", "J189"])

    def test_poa_objects_are_unwrapped_and_must_match_their_position(self):
        poa = [{"code": "J189", "poa": 1}, {"code": "I509", "poa": None}, 0, 1]
        rows = apply_diagnosis_corrections(episode(diagnosis_poa_array=poa), set())
        self.assertEqual([r["diagnosis_poa_array"] for r in rows], [1, None, 0, 1])
        bad = Episode(admission_no=1, diagnosis_codes=("J189", "I509"),
                      diagnosis_poa=({"code": "J189", "poa": 1}, {"code": "WRONG", "poa": 1}))
        with self.assertRaises(CorrectionError):
            apply_diagnosis_corrections(bad, set())

    def test_misaligned_arrays_are_refused_rather_than_reordered(self):
        bad = Episode(admission_no=1, diagnosis_codes=("J189", "I509"), condition_onset=("ADMIT",))
        with self.assertRaises(CorrectionError):
            apply_diagnosis_corrections(bad, set())

    def test_ambiguous_resequence_is_refused(self):
        dup = Episode(admission_no=1, diagnosis_codes=("J189", "I509", "J189"))
        with self.assertRaises(CorrectionError):
            apply_diagnosis_corrections(dup, set(), ("J189", "I509"))
        with self.assertRaises(CorrectionError):
            apply_diagnosis_corrections(episode(), set(), ("J189", "ZZ99"))
        # removal by code string still handles repeats
        rows = apply_diagnosis_corrections(dup, {"J189"}, None)
        self.assertEqual([r["code"] for r in rows], ["I509"])

    def test_removing_everything_is_refused(self):
        with self.assertRaises(CorrectionError):
            apply_diagnosis_corrections(Episode(admission_no=1, diagnosis_codes=("A00",)), {"A00"})

    def test_nan_poa_becomes_null_for_the_grouper(self):
        rows = [{"code": "A00", "diagnosis_poa_array": float("nan")}, {"code": "B00"}]
        self.assertEqual(rows_to_diag_poa(rows), [{"code": "A00", "poa": None}, {"code": "B00", "poa": None}])


def verdict(code, role, level, conf, reasoning="Reasoning text."):
    return CodeVerdict(code, role, level, level != "none", reasoning, conf)


def regroup(**kw):
    base = dict(codes_removed=(), pdx_resequence=None, original_drg="E62B", original_drg_desc="Resp W CC",
                original_cost_weight=0.7, regrouped_drg="E62A", regrouped_drg_desc="Resp W/O CC",
                regrouped_cost_weight=0.55, drg_changed=True, cost_weight_drop=0.15, possible_recovery=1971.4,
                drg_ver_grouper="100")
    base.update(kw)
    return RegroupResult(**base)


class AlertTextTests(unittest.TestCase):
    pdx_ok = verdict("J189", "PDx", "clear", 0.9)

    def test_adx_only_alert(self):
        text = build_alert_info(
            pdx_verdict=self.pdx_ok,
            adx_verdicts=[verdict("I509", "ADx", "clear", 0.8), verdict("N179", "ADx", "none", 0.85, "No renal service.")],
            regroup=regroup(codes_removed=("N179",)), replacement_reasoning=None, mechanical_fallback=False, threshold=0.5)
        self.assertIn("ADx N179: NOT SUPPORTED", text)
        self.assertIn("REMOVED from the claim entirely", text)
        self.assertNotIn("IMPORTANT", text)
        self.assertNotIn("also found", text)
        self.assertIn("Removing N179 regroups to DRG E62A - Resp W/O CC", text)
        self.assertIn("cost weight 0.7", text)
        self.assertIn("cost weight 0.55", text)
        self.assertTrue(text.startswith("The admission was flagged by an AI-assisted coding-support review"))
        self.assertIn("<br>", text)
        self.assertIn("NOT a clinical determination", text)

    def test_model_reasoned_resequence(self):
        text = build_alert_info(
            pdx_verdict=verdict("J189", "PDx", "none", 0.9, "Nothing supports pneumonia."),
            adx_verdicts=[verdict("I509", "ADx", "clear", 0.8)],
            regroup=regroup(pdx_resequence=("J189", "I509")), replacement_reasoning="I509 is actively managed.",
            mechanical_fallback=False, threshold=0.5)
        self.assertIn("PDx J189: NOT SUPPORTED", text)
        self.assertIn("IMPORTANT -- this includes the PRINCIPAL diagnosis", text)
        self.assertIn("J189 was NOT deleted", text)
        self.assertIn("model-reasoned, not a mechanical fallback", text)
        self.assertIn("I509 is actively managed.", text)
        self.assertIn("Resequencing J189 to secondary and promoting I509 to principal regroups", text)

    def test_mechanical_fallback_wording(self):
        text = build_alert_info(
            pdx_verdict=verdict("J189", "PDx", "none", 0.9), adx_verdicts=[verdict("I509", "ADx", "clear", 0.8)],
            regroup=regroup(pdx_resequence=("J189", "I509")), replacement_reasoning="No candidate fits.",
            mechanical_fallback=True, threshold=0.5)
        self.assertIn("MECHANICALLY", text)
        self.assertIn("No candidate fits.", text)
        self.assertNotIn("model-reasoned, not a mechanical fallback", text)

    def test_pdx_and_adx_together(self):
        text = build_alert_info(
            pdx_verdict=verdict("J189", "PDx", "weak", 0.3),
            adx_verdicts=[verdict("I509", "ADx", "clear", 0.8), verdict("N179", "ADx", "none", 0.9)],
            regroup=regroup(pdx_resequence=("J189", "I509"), codes_removed=("N179",)),
            replacement_reasoning="Because.", mechanical_fallback=False, threshold=0.5)
        self.assertIn("PDx J189: LOW-CONFIDENCE SUPPORT", text)
        self.assertIn("confidence 0.30, below the 0.50 threshold", text)
        self.assertIn("was also found unsupported", text)
        self.assertIn("and removing N179,", text)

    def test_drg_unchanged_text(self):
        text = build_alert_info(
            pdx_verdict=self.pdx_ok, adx_verdicts=[verdict("N179", "ADx", "none", 0.9)],
            regroup=regroup(codes_removed=("N179",), drg_changed=False), replacement_reasoning=None,
            mechanical_fallback=False, threshold=0.5)
        self.assertIn("did not change the assigned DRG", text)

    def test_model_written_text_is_html_encoded(self):
        evil = '<img src=x onerror="alert(1)"><script>steal()</script>'
        text = build_alert_info(
            pdx_verdict=verdict("J189", "PDx", "none", 0.9, evil), adx_verdicts=[verdict("I509", "ADx", "clear", 0.8)],
            regroup=regroup(pdx_resequence=("J189", "I509")), replacement_reasoning=evil, mechanical_fallback=False,
            threshold=0.5)
        self.assertNotIn("<script>", text)
        self.assertNotIn("<img", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertEqual(text.count("<br>"), 1)  # only the template's own markup

    def test_only_failing_verdicts_produce_a_finding(self):
        for level in ("clear", "unspecified"):
            with self.subTest(level=level), self.assertRaises(ValueError):
                _support_headline(verdict("J189", "PDx", level, 0.9), 0.5)

    def test_claim_seqno_and_alert_record(self):
        self.assertEqual(get_claim_seqno(episode()), "1001")
        self.assertIsNone(get_claim_seqno(episode(hospital_claims_service_class_code_array=["OTHER", "OTHER"])))
        self.assertEqual(get_claim_seqno(episode(hospital_claims_service_class_code_array=["OTHER", "EPM"])), "1002")
        rec = build_alert_record(alert_info="x", claim_seqno="1001", possible_recovery=1971.5, rule_id=62248)
        self.assertEqual(rec, {"claim_seqno": "1001", "alert_info": "x", "recovery_amount": 1972, "alert_instance_id": 62248})
        self.assertIsNone(build_alert_record(alert_info="x", claim_seqno=None, possible_recovery=None, rule_id=1)["recovery_amount"])


if __name__ == "__main__":
    unittest.main()
