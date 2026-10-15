import unittest

from tests.helpers import make_labels  # noqa: I001

from drg_audit.inference.decision import InverseLabels, decide


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.labels = make_labels()
        self.inverse = InverseLabels.from_label_maps(self.labels)

    def run_decide(self, *, assigned="F62A", mdc=(0.95, 0.05), base=(0.05, 0.95), split=(0.9, 0.05, 0.03, 0.02),
                   benefit=1000.0, threshold=0.85):
        parts = (assigned[0], assigned[1:3], assigned[3] if len(assigned) > 3 else "D")
        return decide(admission_no="1", assigned_drg=assigned, assigned_parts=parts, benefit=benefit,
                      mdc_probs=mdc, base_probs=base, split_probs=split, inverse=self.inverse,
                      split_index=self.labels.split, threshold=threshold)

    def test_agreement_is_not_an_anomaly(self):
        result = self.run_decide()
        self.assertEqual((result.predicted_mdc, result.predicted_base_drg, result.predicted_complexity_split),
                         ("F", "62", "A"))
        self.assertFalse(result.is_billing_anomaly)
        self.assertEqual(result.audit_priority_score, 0.0)

    def test_mdc_discrepancy(self):
        result = self.run_decide(assigned="G62A")
        self.assertTrue(result.mdc_discrepancy_flag)
        self.assertTrue(result.is_billing_anomaly)

    def test_base_discrepancy(self):
        result = self.run_decide(assigned="F10A")
        self.assertTrue(result.base_discrepancy_flag)
        self.assertFalse(result.mdc_discrepancy_flag)

    def test_threshold_is_strictly_greater_than(self):
        result = self.run_decide(assigned="G62A", mdc=(0.85, 0.15), threshold=0.85)
        self.assertFalse(result.mdc_discrepancy_flag)

    def test_low_confidence_disagreement_is_ignored(self):
        result = self.run_decide(assigned="G62A", mdc=(0.6, 0.4))
        self.assertFalse(result.is_billing_anomaly)

    def test_split_inflation_only_when_assigned_is_more_complex(self):
        inflated = self.run_decide(assigned="F62A", split=(0.02, 0.03, 0.9, 0.05))  # model says C, billed A
        self.assertTrue(inflated.split_inflation_flag)
        deflated = self.run_decide(assigned="F62C", split=(0.9, 0.03, 0.04, 0.03))  # model says A, billed C
        self.assertFalse(deflated.split_inflation_flag)

    def test_priority_score_uses_probability_of_assigned_split(self):
        result = self.run_decide(assigned="F62A", split=(0.02, 0.03, 0.9, 0.05), benefit=2000.0)
        self.assertAlmostEqual(result.audit_priority_score, 2000.0 * (1 - 0.02), places=3)
        self.assertAlmostEqual(result.prob_assigned_split, 0.02, places=6)

    def test_unknown_assigned_split_has_zero_probability(self):
        result = self.run_decide(assigned="F62Z", split=(0.02, 0.03, 0.9, 0.05))
        self.assertEqual(result.prob_assigned_split, 0.0)

    def test_nonfinite_benefit_scores_zero(self):
        result = self.run_decide(assigned="G62A", benefit=float("nan"))
        self.assertTrue(result.is_billing_anomaly)
        self.assertEqual(result.audit_priority_score, 0.0)


if __name__ == "__main__":
    unittest.main()
