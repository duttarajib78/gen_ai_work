import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.helpers  # noqa: F401,I001  (puts src/ on sys.path)

import pandas as pd

from drg_audit.errors import DataValidationError
from drg_audit.postprocess import alerts
from drg_audit.postprocess.alerts import (
    AlertIOConfig, AlertsConfig, EligibilityConfig, GrouperConfig, RuleConfig, build_alert_records,
    generate_alerts, pick_claim_seqno, select_candidates,
)


def admissions_frame():
    n = 6
    return pd.DataFrame({
        "admission_no": [1, 2, 3, 4, 5, 6],
        "diagnosis_code_array": [["I21", "E11"], ["C50", "E11"], ["I21", "R07"], ["I21"], ["I21", "E11"],
                                 ["I21", "E11"]],
        "principal_diagnosis_code": ["I21", "C50", "I21", "I21", "I21", "I21"],
        "fund_id": [102, 102, 102, 102, 999, 103],
        "hospital_claims_seqno_array": [[10, 11]] * n,
        "hospital_claims_service_class_code_array": [["XXX", "EPM"]] * n,
        "hospital_claims_array": [["a", "b"]] * n,
    })


PAST = pd.DataFrame({"admission_no": [6]})
PREDICTIONS = pd.DataFrame({"admission_no": [1, 2, 3, 4, 5, 6], "is_billing_anomaly": [True] * 6})


class FakeGrouper:
    def __init__(self, service_classes=None, recovery=1234.56):
        self.calls = 0
        self.service_classes = service_classes
        self.recovery = recovery

    def regroup_without_principal_diagnosis(self, df, reference_data_dir):
        self.calls += 1
        out = df.copy()
        out["drg"] = "F62A"
        out["drg_rw"] = 1.8
        out["regroup_drg"] = "F62B"
        out["regroup_drg_desc"] = "Heart failure"
        out["regroup_drg_value"] = 1.1
        out["drg_ver_grouper"] = "10.0"
        out["possible_recovery"] = self.recovery
        if self.service_classes is not None:
            out["hospital_claims_service_class_code_array"] = [self.service_classes] * len(out)
        return out


class SelectionTests(unittest.TestCase):
    def test_only_eligible_flagged_admissions_remain(self):
        result = select_candidates(admissions_frame(), PAST, PREDICTIONS, EligibilityConfig())
        self.assertEqual(result["admission_no"].tolist(), [1])
        self.assertEqual(result["target_code"].tolist(), ["I21"])
        self.assertEqual(result["grouper_swap"].tolist(), ["remove"])

    def test_boolean_strings_are_parsed_correctly(self):
        predictions = pd.DataFrame({"admission_no": [1, 2], "is_billing_anomaly": ["True", "False"]})
        result = select_candidates(admissions_frame(), PAST, predictions, EligibilityConfig())
        self.assertEqual(result["admission_no"].tolist(), [1])

    def test_unflagged_admissions_are_ignored(self):
        predictions = PREDICTIONS.assign(is_billing_anomaly=False)
        self.assertTrue(select_candidates(admissions_frame(), PAST, predictions, EligibilityConfig()).empty)

    def test_pick_claim_seqno(self):
        self.assertEqual(pick_claim_seqno([1, 2], ["A", "EPM"], ["x", "y"], ["EPM"]), 2)
        self.assertIsNone(pick_claim_seqno([1], ["A"], ["x"], ["EPM"]))
        self.assertEqual(pick_claim_seqno([1, 2], ["EPM", "EPM"], ["foo", "bar"], ["EPM"], "bar|baz"), 2)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "out"
        pred_path = Path(self._tmp.name) / "pred.csv"
        PREDICTIONS.to_csv(pred_path, index=False)
        self.cfg = AlertsConfig(
            rule=RuleConfig(alert_instance_id=62206),
            grouper=GrouperConfig(lorica_stack_name="unused", reference_data_dir="unused"),
            io=AlertIOConfig(admissions_path="adm", past_alerts_path="past", predictions_path=str(pred_path),
                             output_dir=str(self.out)),
        )
        self._patch = mock.patch.object(
            alerts, "read_table", side_effect=lambda p: admissions_frame() if p == "adm" else PAST)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_writes_expected_alert_record(self):
        summary = generate_alerts(self.cfg, backend=FakeGrouper(), run_id="r1")
        self.assertEqual((summary["n_candidates"], summary["n_alerts"]), (1, 1))
        lines = Path(summary["output"]).read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(set(record), {"claim_seqno", "alert_info", "recovery_amount", "alert_instance_id"})
        self.assertEqual((record["claim_seqno"], record["recovery_amount"], record["alert_instance_id"]),
                         ("11", 1235, 62206))
        self.assertIn("PDx I21", record["alert_info"])
        self.assertIn("regroups to DRG F62B - Heart failure", record["alert_info"])
        self.assertNotIn("amout", record["alert_info"])

    def test_output_is_owner_only_and_reruns_do_not_duplicate(self):
        first = generate_alerts(self.cfg, backend=FakeGrouper(), run_id="same")
        second = generate_alerts(self.cfg, backend=FakeGrouper(), run_id="same")
        self.assertEqual(first["output"], second["output"])
        self.assertEqual(len(Path(second["output"]).read_text().splitlines()), 1)
        mode = stat.S_IMODE(os.stat(second["output"]).st_mode)
        self.assertEqual(mode, 0o600)
        third = generate_alerts(self.cfg, backend=FakeGrouper(), run_id="other")
        self.assertNotEqual(first["output"], third["output"])

    def test_no_candidates_does_not_crash_or_build_a_grouper(self):
        self.cfg = AlertsConfig(rule=self.cfg.rule, grouper=self.cfg.grouper, io=self.cfg.io,
                                eligibility=EligibilityConfig(fund_ids=[1]))
        with mock.patch.object(alerts, "build_backend", side_effect=AssertionError("must not be called")):
            summary = generate_alerts(self.cfg, run_id="none")
        self.assertEqual((summary["n_candidates"], summary["n_alerts"], summary["output"]), (0, 0, None))
        self.assertFalse(self.out.exists())

    def test_alerts_without_matching_claim_are_skipped_not_fatal(self):
        summary = generate_alerts(self.cfg, backend=FakeGrouper(service_classes=["XXX", "YYY"]), run_id="skip")
        self.assertEqual(summary["n_alerts"], 0)
        self.assertEqual(summary["skipped"]["no_matching_claim"], 1)
        self.assertIsNone(summary["output"])

    def test_nonfinite_recovery_is_skipped(self):
        summary = generate_alerts(self.cfg, backend=FakeGrouper(recovery=float("nan")), run_id="nan")
        self.assertEqual(summary["skipped"]["invalid_recovery_amount"], 1)

    def test_grouper_output_contract_is_enforced(self):
        broken = FakeGrouper().regroup_without_principal_diagnosis(admissions_frame().head(1), "x").drop(
            columns=["regroup_drg"])
        with self.assertRaisesRegex(DataValidationError, "regroup_drg"):
            build_alert_records(broken, RuleConfig(alert_instance_id=1))


if __name__ == "__main__":
    unittest.main()
