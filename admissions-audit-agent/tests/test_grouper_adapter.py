import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

from audit_agent.deadline import Deadline
from audit_agent.errors import DeadlineExceededError, GrouperInputError, GrouperUnavailableError
from audit_agent.grouper.au_drg import AuDrgGrouper
from audit_agent.grouper.drg_version import derive_grouper_version, derive_grouper_version_value
from audit_agent.schemas import parse_request
from tests.fakes import make_settings, sample_payload

S = make_settings()


def make_fake_au(*, grouper_error=None):
    drg_df = pd.DataFrame({"service_code": ["E62B", "E62A"], "drg_version": ["100", "100"], "value": [0.70, 0.55]})
    drg_desc = pd.DataFrame({"service_code": ["E62B", "E62A"],
                             "service_description": ["Respiratory infections W CC", "Respiratory infections W/O CC"]})
    calls = {"group": 0, "stack": None}

    class FakeGroupy:
        def __init__(self, lorica_stack_name=None):
            calls["stack"] = lorica_stack_name

    def drg_grouper_test(row, groupy, use_alt_diag=False, new_diag_col="new_diag_poa", new_proc_col="procedure_code_array"):
        calls["group"] += 1
        if grouper_error:
            raise grouper_error
        codes = [d["code"] for d in row[new_diag_col]]
        if len(codes) < 4 or codes[0] != "J189":
            return pd.Series({"regroup_drg": "E62A", "cost_weight": 0.55, "original_cost_weight": 0.70})
        return pd.Series({"regroup_drg": "E62B", "cost_weight": 0.70, "original_cost_weight": 0.70})

    def calculate_possible_recovery(res, drg_df, drg_desc):
        res = res.merge(drg_df[["service_code", "value", "drg_version"]],
                        left_on=["drg", "drg_ver_grouper"], right_on=["service_code", "drg_version"], how="left")
        res = res.merge(drg_df[["service_code", "value", "drg_version"]],
                        left_on=["regroup_drg", "drg_ver_grouper"], right_on=["service_code", "drg_version"], how="left")
        res["drg_rw"] = pd.to_numeric(res["original_cost_weight"], errors="coerce").fillna(res["value_x"])
        res["regroup_drg_value"] = pd.to_numeric(res["cost_weight"], errors="coerce").fillna(res["value_y"])
        res["possible_recovery"] = (1 - res["regroup_drg_value"] / res["drg_rw"]) * res["hos_benefit"]
        res = res.drop(columns=["service_code_x", "service_code_y", "drg_version_x", "drg_version_y", "value_x", "value_y"])
        res = res.merge(drg_desc, left_on="drg", right_on="service_code", how="left")
        res = res.merge(drg_desc, left_on="regroup_drg", right_on="service_code", how="left")
        res = res.rename(columns={"service_description_x": "drg_desc", "service_description_y": "regroup_drg_desc"})
        return res.drop(columns=["service_code_x", "service_code_y"])

    module = types.SimpleNamespace(Groupy=FakeGroupy, get_metadata=lambda path: (drg_df, drg_desc),
                                   drg_grouper_test=drg_grouper_test, calculate_possible_recovery=calculate_possible_recovery)
    return module, drg_df, drg_desc, calls


def make_grouper(**kw):
    module, drg_df, drg_desc, calls = make_fake_au(**kw)
    return AuDrgGrouper(au_grouper_module=module, groupy=module.Groupy(lorica_stack_name="t"), drg_df=drg_df,
                        drg_desc=drg_desc), calls


def ep(**o):
    return parse_request(sample_payload(**o), S).episode


class RegroupTests(unittest.TestCase):
    def test_adx_removal_regroups_and_computes_recovery(self):
        g, calls = make_grouper()
        r = g.regroup(ep(), removed_codes=frozenset({"N179"}), pdx_resequence=None, deadline=Deadline(50))
        self.assertEqual(calls["group"], 1)
        self.assertTrue(r.drg_changed)
        self.assertEqual((r.original_drg, r.regrouped_drg), ("E62B", "E62A"))
        self.assertEqual((r.original_drg_desc, r.regrouped_drg_desc),
                         ("Respiratory infections W CC", "Respiratory infections W/O CC"))
        self.assertAlmostEqual(r.cost_weight_drop, 0.15)
        self.assertAlmostEqual(r.possible_recovery, (1 - 0.55 / 0.70) * 9200.0, places=4)
        self.assertEqual(r.drg_ver_grouper, "100")
        self.assertEqual(r.codes_removed, ("N179",))
        self.assertIsInstance(r.to_dict()["codes_removed"], list)

    def test_pure_resequence_is_sent_with_the_new_principal_first(self):
        seen = {}
        module, drg_df, drg_desc, _ = make_fake_au()
        original = module.drg_grouper_test
        def spy(row, groupy, **kw):
            seen["codes"] = [d["code"] for d in row[kw["new_diag_col"]]]
            seen["poa"] = [d["poa"] for d in row[kw["new_diag_col"]]]
            return original(row, groupy, **kw)
        module.drg_grouper_test = spy
        g = AuDrgGrouper(au_grouper_module=module, groupy=object(), drg_df=drg_df, drg_desc=drg_desc)
        r = g.regroup(ep(), removed_codes=frozenset(), pdx_resequence=("J189", "I509"), deadline=Deadline(50))
        self.assertEqual(seen["codes"], ["I509", "N179", "E119", "J189"])
        self.assertEqual(seen["poa"], [1, 0, 1, 1])
        self.assertEqual(r.pdx_resequence, ("J189", "I509"))
        self.assertTrue(r.drg_changed)

    def test_unchanged_drg_is_reported_as_unchanged(self):
        g, _ = make_grouper()
        r = g.regroup(ep(), removed_codes=frozenset(), pdx_resequence=None, deadline=Deadline(50))
        self.assertFalse(r.drg_changed)
        self.assertAlmostEqual(r.cost_weight_drop, 0.0)

    def test_bad_input_style_errors_and_outages_are_distinguished(self):
        for exc, expected in ((KeyError("member_dob"), GrouperInputError), (ValueError("bad"), GrouperInputError),
                              (ConnectionError("down"), GrouperUnavailableError), (RuntimeError("500"), GrouperUnavailableError)):
            with self.subTest(type(exc).__name__):
                g, _ = make_grouper(grouper_error=exc)
                with self.assertRaises(expected) as ctx:
                    g.regroup(ep(), removed_codes=frozenset({"N179"}), pdx_resequence=None, deadline=Deadline(50))
                self.assertNotIn("member_dob", str(ctx.exception))  # messages carry the exception type only

    def test_no_call_is_started_without_enough_time(self):
        g, calls = make_grouper()
        with self.assertRaises(DeadlineExceededError):
            g.regroup(ep(), removed_codes=frozenset({"N179"}), pdx_resequence=None, deadline=Deadline(2.0))
        self.assertEqual(calls["group"], 0)


class StartupTests(unittest.TestCase):
    def test_missing_au_grouper_module_fails_with_guidance(self):
        sys.modules.pop("au_grouper", None)
        with self.assertRaises(GrouperUnavailableError) as ctx:
            AuDrgGrouper.from_settings(S)
        self.assertIn("vendor", str(ctx.exception))

    def test_from_settings_uses_the_drg_directory_and_stack_name(self):
        module, _, _, calls = make_fake_au()
        sys.modules["au_grouper"] = module
        self.addCleanup(sys.modules.pop, "au_grouper", None)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "drg").mkdir()
            settings = make_settings(AUDIT_MODEL_DIR=tmp, AUDIT_GROUPER_BACKEND="au_drg")
            AuDrgGrouper.from_settings(settings)
            self.assertEqual(calls["stack"], "hambs-shed")
            with tempfile.TemporaryDirectory() as empty, self.assertRaises(Exception):
                AuDrgGrouper.from_settings(make_settings(AUDIT_MODEL_DIR=empty, AUDIT_GROUPER_BACKEND="au_drg"))


class DrgVersionTests(unittest.TestCase):
    def test_mapping_matches_the_grouper_harness(self):
        cases = [("100", "[]", "100"), ("0100", "[]", "100"), ("10", "[]", "100"), ("62", "[]", "60"), ("6X", "[]", "60"),
                 ("6.0", "[]", "60"), ("80", "[]", "80"), ("8x", "[]", "80"), ("92", "[]", "90"), ("9x", "[]", "90"),
                 ("110", "[]", "110"), ("52", "[]", "52"), ("999", "[]", "60"), (None, "[]", "60"), (float("nan"), "[]", "60"),
                 ("0", "[]", "60"), ("62", "['CWO']", "100"), ("110", "['DRG_CASE_PAYMENT', 'CWO']", "100")]
        for vers, classes, expected in cases:
            with self.subTest(vers=vers, classes=classes):
                self.assertEqual(derive_grouper_version_value(vers, classes), expected)

    def test_dataframe_wrapper(self):
        df = pd.DataFrame([{"drg_vers": "0100", "hospital_claims_service_class_code_array": ["DRG_CASE_PAYMENT"]}])
        out = derive_grouper_version(df)
        self.assertEqual(out.loc[0, "drg_ver_grouper"], "100")
        self.assertEqual(out.loc[0, "drg_vers"], "100")
        self.assertEqual(df.loc[0, "drg_vers"], "0100")  # input untouched


if __name__ == "__main__":
    unittest.main()
