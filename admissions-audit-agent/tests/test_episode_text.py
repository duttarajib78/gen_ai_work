import tempfile
import unittest
from pathlib import Path

from audit_agent.episode_text import build_episode_text
from audit_agent.reference_data import ReferenceData
from audit_agent.schemas import parse_request
from tests.fakes import make_reference_dir, make_settings, sample_payload

S = make_settings()


class EpisodeTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ref = ReferenceData.from_directory(make_reference_dir(Path(cls._tmp.name)))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def episode(self, **overrides):
        return parse_request(sample_payload(**overrides), S).episode

    def test_exact_prompt_format(self):
        f = build_episode_text(self.episode(), self.ref)
        expected = "\n".join([
            "EPISODE", "",
            "PDx:", "  J189 \u2014 Pneumonia, unspecified", "",
            "ADx:",
            "  I509 \u2014 Heart failure, unspecified",
            "  N179 \u2014 Acute kidney failure, unspecified",
            "  E119 \u2014 Type 2 diabetes mellitus without complications", "",
            "ACHI:",
            "  3237500 \u2014 Intravenous administration of pharmacological agent (day 1)",
            "  9200900 \u2014 Chest physiotherapy (day 2)", "",
            "MBS / hospital services:",
            "  3 \u2014 Professional attendance by a general practitioner (day unknown)",
            "  23 \u2014 Consultation (superseded) [SUPERSEDED per current MBS extract] (day unknown)",
        ])
        self.assertEqual(f.text, expected)
        self.assertEqual(f.pdx_code, "J189")
        self.assertEqual(f.adx_codes, ("I509", "N179", "E119"))

    def test_nothing_identifying_or_financial_reaches_the_prompt(self):
        text = build_episode_text(self.episode(), self.ref).text
        for forbidden in ("555123", "E62B", "9200.0", "1952", "0000123X", "2026-08", "Emergency", "DRG_CASE_PAYMENT"):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("member", text.lower())

    def test_duplicate_codes_collapse_and_pdx_wins_ties(self):
        e = self.episode(
            diagnosis_code_array=["J189", "Y8313", "I509", "Y8313", "J189"],
            diagnosis_poa_array=None, condition_onset_array=None, dx_code_idx_array=None,
        )
        f = build_episode_text(e, self.ref)
        self.assertEqual(f.pdx_code, "J189")
        self.assertEqual(f.adx_codes, ("Y8313", "I509"))
        self.assertEqual(f.diagnosis_order, ("J189", "Y8313", "I509", "Y8313", "J189"))  # raw order kept for corrections

    def test_empty_sections_and_unknown_codes_are_visible(self):
        e = self.episode(diagnosis_code_array=["ZZ999"], procedure_code_array=[], medical_claims_array=[],
                         procedure_date_array=[], diagnosis_poa_array=None, condition_onset_array=None,
                         dx_code_idx_array=None)
        text = build_episode_text(e, self.ref).text
        self.assertIn("  ZZ999 \u2014 (description not found)", text)
        self.assertEqual(text.count("  (none)"), 3)  # ADx, ACHI, MBS

    def test_day_offsets(self):
        e = self.episode(procedure_date_array=["2026-08-01", None], dte_admitted="2026-08-01T22:00:00")
        text = build_episode_text(e, self.ref).text
        self.assertIn("pharmacological agent (day 0)", text)
        self.assertIn("Chest physiotherapy (day unknown)", text)
        e = parse_request(sample_payload(options={"mode": "verdicts_only"}, dte_admitted=None,
                                         procedure_date_array=["2026-08-01"]), S).episode
        self.assertIn("pharmacological agent (day unknown)", build_episode_text(e, self.ref).text)

    def test_mbs_array_field_is_selectable(self):
        e = self.episode(hospital_claims_array=["3"], medical_claims_array=["23"])
        self.assertIn("  3 \u2014 Professional", build_episode_text(e, self.ref, "hospital_claims_array").text)
        self.assertIn("  23 \u2014 Consultation", build_episode_text(e, self.ref, "medical_claims_array").text)


if __name__ == "__main__":
    unittest.main()
