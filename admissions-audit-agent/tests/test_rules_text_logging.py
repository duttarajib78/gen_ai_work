import io
import json
import logging
import unittest

from audit_agent.logging_utils import JsonFormatter, exception_fields, log_event
from audit_agent.rules import diagnosis_fails, flag_reason
from audit_agent.textutil import clean_line, clean_text, escape_html


class RuleTests(unittest.TestCase):
    def test_truth_table(self):
        t = 0.5
        cases = [
            ("none", 0.99, True, "none"), ("none", 0.0, True, "none"),
            ("weak", 0.49, True, "low_confidence_weak"), ("weak", 0.5, False, None), ("weak", 0.95, False, None),
            ("clear", 0.0, False, None), ("clear", 1.0, False, None),
        ]
        for level, conf, fails, reason in cases:
            with self.subTest(level=level, conf=conf):
                self.assertEqual(diagnosis_fails(level, conf, t), fails)
                self.assertEqual(flag_reason(level, conf, t), reason)

    def test_threshold_is_an_argument_not_state(self):
        self.assertTrue(diagnosis_fails("weak", 0.6, 0.7))
        self.assertFalse(diagnosis_fails("weak", 0.6, 0.5))


class TextTests(unittest.TestCase):
    def test_clean_text_strips_unsafe_characters_and_caps_length(self):
        dirty = "a\x00b\u202ec\u200bd\r\n\n\n\n\ne" + "x" * 50
        out = clean_text(dirty, 20)
        self.assertNotIn("\x00", out)
        self.assertNotIn("\u202e", out)
        self.assertNotIn("\u200b", out)
        self.assertLessEqual(len(out), 20)
        self.assertTrue(out.endswith("[truncated]"))

    def test_clean_line_collapses_whitespace(self):
        self.assertEqual(clean_line("a \n\t b   c"), "a b c")

    def test_escape_html(self):
        self.assertEqual(escape_html('<script>alert("x")</script>&'), "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&amp;")
        self.assertEqual(escape_html(None), "")


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(JsonFormatter())
        self.logger = logging.getLogger("test.audit.logging")
        self.logger.handlers[:] = [handler]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def last(self):
        return json.loads(self.stream.getvalue().strip().splitlines()[-1])

    def test_only_allowlisted_fields_are_emitted(self):
        log_event(self.logger, "episode_completed", request_id="r1", status="completed",
                  reasoning="PATIENT HAS CANCER", episode_text="EPISODE ...", diagnosis_codes=["J189"])
        rec = self.last()
        self.assertEqual(rec["request_id"], "r1")
        for leaked in ("reasoning", "episode_text", "diagnosis_codes"):
            self.assertNotIn(leaked, rec)
        self.assertEqual(rec["dropped_fields"], 3)
        self.assertNotIn("CANCER", self.stream.getvalue())

    def test_admission_number_needs_explicit_opt_in(self):
        log_event(self.logger, "x", admission_no=555123, request_id="r")
        self.assertNotIn("admission_no", self.last())
        log_event(self.logger, "x", admission_no=555123, request_id="r", log_raw_identifiers=True)
        self.assertEqual(self.last()["admission_no"], 555123)

    def test_exception_fields_exclude_locals_and_scrub_message(self):
        try:
            secret_local = "PATIENT-DATA"  # noqa: F841
            raise ValueError("bad\nvalue")
        except ValueError as exc:
            fields = exception_fields(exc)
        self.assertEqual(fields["exception_type"], "ValueError")
        self.assertNotIn("\n", fields["exception_message"])
        self.assertNotIn("PATIENT-DATA", json.dumps(fields))
        self.assertTrue(fields["stack"][-1].startswith("test_rules_text_logging.py:"))


if __name__ == "__main__":
    unittest.main()
