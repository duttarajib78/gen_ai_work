import json
import logging
import unittest

from tests.helpers import valid_payload_record  # noqa: I001

from drg_audit.schemas import AuditResult
from drg_audit.serving.service import InferenceService
from drg_audit.settings import RequestLimits


class StubEngine:
    model_version = "testversion0001"
    default_threshold = 0.85

    def __init__(self):
        self.calls = []

    def predict(self, records, confidence_threshold=None):
        self.calls.append((len(records), confidence_threshold))
        results = []
        for record in records:
            if len(record.drg) < 3:
                results.append(AuditResult.skipped(record.admission_no, "invalid_assigned_drg"))
            else:
                results.append(AuditResult(admission_no=record.admission_no, status="ok",
                                           is_billing_anomaly=record.drg.startswith("G")))
        return results


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = StubEngine()
        self.service = InferenceService(self.engine, RequestLimits())

    def call(self, records, options=None):
        payload = {"records": records}
        if options:
            payload["options"] = options
        request = self.service.parse(json.dumps(payload), "application/json")
        return self.service.score(request)

    def test_order_summary_and_merged_warnings(self):
        no_age = valid_payload_record(admission_no="3", drg="G62A")
        del no_age["member_age"]
        response = self.call([
            valid_payload_record(admission_no="1"),
            valid_payload_record(admission_no="2", los=-1),   # rejected
            no_age,                                           # anomaly + warning
            valid_payload_record(admission_no="4", drg="F6"),  # skipped by engine
        ])
        statuses = [r["status"] for r in response["results"]]
        self.assertEqual(statuses, ["ok", "rejected", "ok", "skipped"])
        self.assertEqual([r["admission_no"] for r in response["results"]], ["1", "2", "3", "4"])
        self.assertEqual(response["summary"], {"n_records": 4, "n_scored": 2, "n_anomalies": 1,
                                               "n_skipped": 1, "n_rejected": 1})
        self.assertIn("member_age_missing_defaulted", response["results"][2]["warnings"])
        self.assertEqual(response["model_version"], "testversion0001")
        self.assertEqual(self.engine.calls, [(3, 0.85)])  # rejected records never reach the model

    def test_threshold_override(self):
        response = self.call([valid_payload_record()], options={"confidence_threshold": 0.95})
        self.assertEqual(response["confidence_threshold"], 0.95)
        self.assertEqual(self.engine.calls[-1][1], 0.95)

    def test_engine_is_not_called_when_everything_is_rejected(self):
        response = self.call([valid_payload_record(los=-1)])
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(response["summary"]["n_rejected"], 1)

    def test_request_ids_are_unique(self):
        ids = {self.call([valid_payload_record()])["request_id"] for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_engine_contract_violation_surfaces(self):
        self.engine.predict = lambda records, confidence_threshold=None: []
        with self.assertRaises(RuntimeError):
            self.call([valid_payload_record()])

    def test_logs_never_contain_request_data(self):
        sensitive = valid_payload_record(admission_no="SENSITIVE-ADM-777",
                                         diagnosis_code_array=["Z99.SENSITIVE", "I10"])
        with self.assertLogs("drg_audit.serving", level=logging.INFO) as captured:
            self.call([sensitive])
        rendered = json.dumps([{**r.__dict__, "msg": r.getMessage()} for r in captured.records], default=str)
        self.assertNotIn("SENSITIVE", rendered)
        self.assertIn("request_scored", rendered)


if __name__ == "__main__":
    unittest.main()
