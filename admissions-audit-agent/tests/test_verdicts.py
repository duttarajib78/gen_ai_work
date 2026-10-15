import unittest

from audit_agent.errors import LlmOutputError
from audit_agent.llm.verdicts import (
    CodeVerdict, InvalidAnswer, LlmTaskConfig, get_diagnosis_verdicts, get_replacement_pdx,
    replacement_validator, verdicts_validator,
)
from tests.fakes import ScriptedLlm, new_deadline, new_meter


def entry(code, role, level="clear", conf=0.9, reasoning="Weighed code X."):
    return {"code": code, "role": role, "support_level": level, "confidence": conf, "reasoning": reasoning}


def good(*extra):
    return {"verdicts": [entry("J189", "PDx"), entry("I509", "ADx"), *extra]}


class VerdictValidatorTests(unittest.TestCase):
    validate = staticmethod(verdicts_validator("J189", ["I509"]))

    def test_returns_pdx_first_then_adx_in_coded_order_and_normalises(self):
        args = {"verdicts": [entry(" i509 ", "ADx", "WEAK", "0.7"), entry("J189", "PDx", "none", 1.7)]}
        out = self.validate(args)
        self.assertEqual([v.code for v in out], ["J189", "I509"])
        self.assertEqual((out[0].support_level, out[0].supported, out[0].confidence), ("none", False, 1.0))
        self.assertEqual((out[1].support_level, out[1].supported, out[1].confidence), ("weak", True, 0.7))
        self.assertIsInstance(out[0], CodeVerdict)

    def test_rejections_carry_a_fixed_category(self):
        cases = {
            "not_a_list": {"verdicts": "x"},
            "bad_entry": {"verdicts": ["x"]},
            "missing_field": {"verdicts": [{"code": "J189"}]},
            "stray_code": {"verdicts": [entry("J189", "PDx"), entry("Z999", "ADx")]},
            "duplicate_code": {"verdicts": [entry("J189", "PDx"), entry("J189", "PDx")]},
            "wrong_role": {"verdicts": [entry("J189", "ADx"), entry("I509", "ADx")]},
            "bad_support_level": {"verdicts": [entry("J189", "PDx", "partial"), entry("I509", "ADx")]},
            "empty_reasoning": {"verdicts": [entry("J189", "PDx", reasoning="  "), entry("I509", "ADx")]},
            "bad_confidence": {"verdicts": [entry("J189", "PDx", conf="high"), entry("I509", "ADx")]},
            "missing_code": {"verdicts": [entry("J189", "PDx")]},
        }
        for category, args in cases.items():
            with self.subTest(category), self.assertRaises(InvalidAnswer) as ctx:
                self.validate(args)
            self.assertEqual(ctx.exception.category, category)

    def test_confidence_must_be_finite_and_not_boolean(self):
        for bad in (float("nan"), float("inf"), True, None, [0.5]):
            with self.subTest(bad=bad), self.assertRaises(InvalidAnswer):
                self.validate({"verdicts": [entry("J189", "PDx", conf=bad), entry("I509", "ADx")]})

    def test_reasoning_is_sanitised_and_capped(self):
        out = self.validate({"verdicts": [entry("J189", "PDx", reasoning="ok\u202e" + "y" * 20000), entry("I509", "ADx")]})
        self.assertNotIn("\u202e", out[0].reasoning)
        self.assertLessEqual(len(out[0].reasoning), 8000)


class AskLoopTests(unittest.TestCase):
    cfg = LlmTaskConfig(max_output_tokens=1000, max_output_tokens_ceiling=3000, max_attempts=3)

    def run_verdicts(self, llm):
        return get_diagnosis_verdicts(llm, episode_text="EPISODE", pdx_code="J189", adx_codes=["I509"],
                                      deadline=new_deadline(), meter=new_meter(), cfg=self.cfg)

    def test_first_valid_answer_is_used(self):
        llm = ScriptedLlm(good())
        self.assertEqual(len(self.run_verdicts(llm)), 2)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn('"J189" (PDx)', llm.calls[0]["user_text"])
        self.assertIn("<episode>\nEPISODE\n</episode>", llm.calls[0]["user_text"])

    def test_invalid_answer_is_re_asked_with_a_specific_reason(self):
        llm = ScriptedLlm({"verdicts": [entry("J189", "PDx")]}, good())
        self.run_verdicts(llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn("rejected", llm.calls[0]["user_text"])
        self.assertIn("Your previous submission was rejected", llm.calls[1]["user_text"])
        self.assertIn("I509", llm.calls[1]["user_text"])

    def test_exhausted_attempts_raise_without_echoing_model_text(self):
        leaky = {"verdicts": [entry("J189", "PDx", reasoning="PATIENT-SECRET"), entry("ZZZ", "ADx")]}
        llm = ScriptedLlm(leaky, leaky, leaky)
        with self.assertRaises(LlmOutputError) as ctx:
            self.run_verdicts(llm)
        self.assertEqual(ctx.exception.details["last_reason"], "stray_code")
        self.assertEqual(ctx.exception.details["attempts"], 3)
        self.assertNotIn("PATIENT-SECRET", str(ctx.exception.to_error_body()))

    def test_truncation_doubles_the_token_budget_up_to_the_ceiling(self):
        cut = LlmOutputError("cut off", code="llm_output_truncated")
        llm = ScriptedLlm(cut, cut, good())
        self.run_verdicts(llm)
        self.assertEqual([c["max_output_tokens"] for c in llm.calls], [1000, 2000, 3000])

    def test_output_errors_from_the_client_are_retried_too(self):
        llm = ScriptedLlm(LlmOutputError("no tool call"), good())
        self.run_verdicts(llm)
        self.assertEqual(len(llm.calls), 2)


class ReplacementTests(unittest.TestCase):
    cfg = LlmTaskConfig(max_attempts=2)
    old = CodeVerdict("J189", "PDx", "none", False, "No code supports it.", 0.9)

    def ask(self, llm, candidates=("I509", "E119")):
        return get_replacement_pdx(llm, episode_text="EPISODE", old_pdx=self.old, candidate_codes=list(candidates),
                                   deadline=new_deadline(), meter=new_meter(), cfg=self.cfg)

    def test_choice_from_candidates(self):
        llm = ScriptedLlm({"new_principal_code": " e119", "reasoning": "Because."})
        self.assertEqual(self.ask(llm).new_principal_code, "E119")
        text = llm.calls[0]["user_text"]
        self.assertIn("<previous_assessment>\nNo code supports it.\n</previous_assessment>", text)
        self.assertIn("  I509\n  E119", text)

    def test_none_is_a_valid_answer_in_several_spellings(self):
        for raw in (None, "null", "None", "  ", "NULL"):
            with self.subTest(raw=raw):
                self.assertIsNone(self.ask(ScriptedLlm({"new_principal_code": raw, "reasoning": "None fit."})).new_principal_code)

    def test_code_outside_the_candidate_list_is_rejected_then_re_asked(self):
        llm = ScriptedLlm({"new_principal_code": "N179", "reasoning": "x"}, {"new_principal_code": "I509", "reasoning": "y"})
        self.assertEqual(self.ask(llm).new_principal_code, "I509")
        self.assertIn("not in the candidate list", llm.calls[1]["user_text"])

    def test_validator_requires_reasoning_and_both_fields(self):
        v = replacement_validator(["I509"])
        for args in ({"reasoning": "x"}, {"new_principal_code": None}, {"new_principal_code": None, "reasoning": " "}):
            with self.subTest(args=args), self.assertRaises(InvalidAnswer):
                v(args)


if __name__ == "__main__":
    unittest.main()
