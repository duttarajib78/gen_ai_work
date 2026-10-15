import json
import unittest

from tests.helpers import valid_payload_record  # noqa: I001

from drg_audit.errors import NotAcceptableError, PayloadTooLargeError, RequestValidationError, UnsupportedMediaTypeError
from drg_audit.serving.contract import (
    ParsedRecord, RejectedRecord, negotiate_accept, parse_request, serialize_response,
)
from drg_audit.settings import RequestLimits

LIMITS = RequestLimits()


def parse(payload, limits=LIMITS, content_type="application/json"):
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    return parse_request(body, content_type, limits)


class RequestLevelTests(unittest.TestCase):
    def assert_error(self, code, status, *args, **kwargs):
        with self.assertRaises(RequestValidationError) as ctx:
            parse(*args, **kwargs)
        self.assertEqual((ctx.exception.code, ctx.exception.status_code), (code, status))

    def test_valid_request(self):
        request = parse({"records": [valid_payload_record()]})
        self.assertEqual(len(request.valid), 1)
        record = request.valid[0].record
        self.assertEqual(record.medical_claims_array, ["55114", "38200"])  # ints are accepted and stringified
        self.assertEqual(record.los, 5)
        self.assertEqual(request.valid[0].warnings, ())

    def test_content_type_and_accept(self):
        with self.assertRaises(UnsupportedMediaTypeError):
            parse({"records": [valid_payload_record()]}, content_type="text/csv")
        self.assertEqual(negotiate_accept(None), "application/json")
        self.assertEqual(negotiate_accept("*/*"), "application/json")
        with self.assertRaises(NotAcceptableError):
            negotiate_accept("text/html")

    def test_size_limits(self):
        small = RequestLimits(max_payload_bytes=1024)
        with self.assertRaises(PayloadTooLargeError):
            parse({"records": [valid_payload_record(admission_no=str(i)) for i in range(50)]}, limits=small)
        few = RequestLimits(max_records=2)
        with self.assertRaises(PayloadTooLargeError):
            parse({"records": [valid_payload_record()] * 3}, limits=few)

    def test_malformed_bodies(self):
        self.assert_error("invalid_json", 400, "{not json")
        self.assert_error("invalid_json", 400, '{"records": [NaN]}')
        self.assert_error("invalid_json", 400, '{"records": [Infinity]}')
        self.assert_error("empty_body", 400, b"")
        self.assert_error("invalid_encoding", 400, b"\xff\xfe")
        self.assert_error("invalid_shape", 400, "[]")
        self.assert_error("invalid_records", 400, {"records": []})
        self.assert_error("invalid_records", 400, {"records": "x"})
        self.assert_error("unknown_fields", 400, {"records": [valid_payload_record()], "debug": True})

    def test_deeply_nested_json_is_rejected_not_crashed(self):
        self.assert_error("invalid_json", 400, "[" * 100000 + "]" * 100000)

    def test_options(self):
        request = parse({"records": [valid_payload_record()], "options": {"confidence_threshold": 0.9}})
        self.assertEqual(request.confidence_threshold, 0.9)
        for bad in ({"confidence_threshold": 0.1}, {"confidence_threshold": "0.9"},
                    {"confidence_threshold": True}, {"confidence_threshold": 1.5}, {"other": 1}, []):
            self.assert_error("invalid_options", 400, {"records": [valid_payload_record()], "options": bad})


class RecordLevelTests(unittest.TestCase):
    def rejected(self, **overrides):
        request = parse({"records": [valid_payload_record(**overrides)]})
        item = request.items[0]
        self.assertIsInstance(item, RejectedRecord, msg=f"expected rejection for {overrides}")
        return item

    def test_defaults_and_warnings(self):
        record = valid_payload_record()
        for key in ("member_age", "drg_benefit", "minutes_operating_theatre", "procedure_code_array"):
            record.pop(key)
        item = parse({"records": [record]}).items[0]
        self.assertIsInstance(item, ParsedRecord)
        self.assertEqual(item.record.member_age, 50.0)
        self.assertEqual(item.record.drg_benefit, 0.0)
        self.assertEqual(set(item.warnings), {"member_age_missing_defaulted", "drg_benefit_missing"})

    def test_missing_required_field(self):
        record = valid_payload_record()
        del record["drg"]
        item = parse({"records": [record]}).items[0]
        self.assertEqual((item.reason, item.field), ("missing_field", "drg"))

    def test_type_and_range_violations(self):
        cases = {
            "los": [True, 5.5, -1, 99999, "5"],
            "member_age": [-1, 500, "40", None.__class__],
            "drg_benefit": [1e12, "x"],
            "minutes_operating_theatre": [-1, 1e9],
            "diagnosis_code_array": [[], "I21", [""], [None], [1.5], [True], ["x" * 40]],
            "drg_vers": [10.0, "", "a b"],
            "drg": [1234, "F62A!!", "TOOLONGDRG1"],
            "admission_no": ["has space", "x" * 65, True, 1.5],
        }
        for field, values in cases.items():
            for value in values:
                if value is type(None):
                    continue
                with self.subTest(field=field, value=repr(value)[:30]):
                    try:
                        body = json.dumps({"records": [valid_payload_record(**{field: value})]})
                    except ValueError:
                        continue
                    item = parse(body).items[0]
                    self.assertIsInstance(item, RejectedRecord)

    def test_overflowing_number_literal_is_rejected_per_record(self):
        # 1e999 is syntactically valid JSON but overflows to infinity; it must be refused, not scored.
        body = ('{"records":[{"admission_no":"1","diagnosis_code_array":["I10"],'
                '"drg_vers":"10.0","drg":"F62A","los":1e999}]}')
        item = parse(body).items[0]
        self.assertIsInstance(item, RejectedRecord)
        self.assertEqual(item.field, "los")

    def test_injection_style_codes_are_rejected(self):
        for payload in ("I21.4; DROP TABLE x", "../../etc/passwd", "<script>", "I21\n4", "${jndi:x}", "a b"):
            item = self.rejected(diagnosis_code_array=[payload])
            self.assertEqual(item.field, "diagnosis_code_array")

    def test_unknown_record_fields_rejected(self):
        item = self.rejected(extra_field="x")
        self.assertEqual(item.reason, "unknown_fields")

    def test_rejection_never_echoes_submitted_values(self):
        secret = "PATIENT-SECRET value 123"
        item = self.rejected(admission_no=secret)
        self.assertIsNone(item.admission_no)  # invalid id is not echoed
        self.assertNotIn("SECRET", item.reason_text)
        item = self.rejected(los=-1)
        self.assertEqual(item.admission_no, "1001")  # a syntactically valid id is echoed for correlation

    def test_one_bad_record_does_not_block_the_batch(self):
        request = parse({"records": [valid_payload_record(), valid_payload_record(los=-3), 7,
                                     valid_payload_record(admission_no="1002")]})
        kinds = [type(i).__name__ for i in request.items]
        self.assertEqual(kinds, ["ParsedRecord", "RejectedRecord", "RejectedRecord", "ParsedRecord"])
        self.assertEqual([i.index for i in request.items], [0, 1, 2, 3])

    def test_integer_valued_float_los_is_accepted(self):
        item = parse({"records": [valid_payload_record(los=5.0)]}).items[0]
        self.assertEqual(item.record.los, 5)


class ResponseTests(unittest.TestCase):
    def test_serialization_is_strict_json(self):
        self.assertEqual(serialize_response({"a": 1.5}), '{"a":1.5}')
        with self.assertRaises(ValueError):
            serialize_response({"a": float("nan")})


if __name__ == "__main__":
    unittest.main()
