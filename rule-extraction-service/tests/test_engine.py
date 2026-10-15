import json
import os
import tempfile
import unittest

from rule_extractor.engine import actions, code_match, conditions, grammar, rules_store
from rule_extractor.engine.compare import run_episode_against_ruleset
from rule_extractor.engine.episode import Episode
from rule_extractor.engine.grouper import MockGrouperAdapter, load_grouper


def ep(**kw):
    base = dict(episode_id="e1", diagnosis_code_array=["A41.0", "B95.6", "E11.9"], procedure_code_array=["40903-00"],
                medical_claims_array=["49318"], condition_onset_array=["1", "1", "2"],
                member_sex="F", member_age=40)
    base.update(kw)
    return Episode(**base)


class CodeMatchTests(unittest.TestCase):
    def test_canonical_forms_match(self):
        self.assertTrue(code_match.matches("E11.9", "E119"))
        self.assertTrue(code_match.matches("e11.9", "E11.9"))
        self.assertTrue(code_match.matches("30571-00", "3057100"))

    def test_prefix_and_wildcard(self):
        self.assertTrue(code_match.matches("B95.6", "B95*"))
        self.assertTrue(code_match.matches("E11.65", "E11.*"))
        self.assertTrue(code_match.matches("30571-00", "3057?-00"))
        self.assertFalse(code_match.matches("B96.1", "B95*"))

    def test_pattern_validation(self):
        self.assertTrue(code_match.validate_pattern("*"))
        self.assertTrue(code_match.validate_pattern("a b"))
        self.assertTrue(code_match.validate_pattern("x" * 40))
        self.assertEqual(code_match.validate_pattern("B95*"), [])
        with self.assertRaises(ValueError):
            code_match.normalize_patterns([])


class ConditionTests(unittest.TestCase):
    def test_all_documented_examples_validate(self):
        for example in grammar.EXAMPLES:
            with self.subTest(shape=example["shape"]):
                self.assertEqual(conditions.validate_condition(example["flag_condition"]), [])
                self.assertEqual(actions.validate_action(example["correction_action"]), [])
                rule, errors = rules_store.build_executable("R1", example["recommended_action"],
                                                            example["flag_condition"], example["correction_action"])
                self.assertEqual(errors, [])
                self.assertIsNotNone(rule)

    def test_every_operator(self):
        e = ep(hours_of_mechanical_ventilation=50)
        cases = {
            "has_diag": ({"has_diagnosis": "B95*"}, True),
            "has_proc": ({"has_procedure": ["40903-00", "x1"]}, True),
            "has_mbs": ({"has_mbs_item": "49318"}, True),
            "principal": ({"principal_is": "A41.0"}, True),
            "secondary": ({"secondary_has": "A41.0"}, False),
            "count": ({"count": {"in": "diagnosis", "match": "*9*", "op": ">=", "value": 2}}, True),
            "field_num": ({"field": {"name": "hours_of_mechanical_ventilation", "op": "<", "value": 96}}, True),
            "field_str": ({"field": {"name": "member_sex", "op": "==", "value": "f"}}, True),
            "field_in": ({"field": {"name": "member_age", "op": "in", "value": [40, 41]}}, True),
            "null_field": ({"field": {"name": "prosthesis_benefit", "op": ">", "value": 0}}, False),
            "is_null": ({"field": {"name": "prosthesis_benefit", "op": "is_null"}}, True),
            "onset": ({"onset": {"match": "E11*", "flag": "2"}}, True),
            "sequence": ({"sequence": {"first": "A41*", "then": "E11*"}}, True),
            "sequence_rev": ({"sequence": {"first": "E11*", "then": "A41*"}}, False),
            "not": ({"not": {"has_diagnosis": "Z99*"}}, True),
            "any": ({"any": [{"has_diagnosis": "Z99*"}, {"has_diagnosis": "A41*"}]}, True),
        }
        for name, (cond, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(conditions.evaluate(cond, e), expected)

    def test_invalid_conditions_are_rejected_with_paths(self):
        bad = [
            {"all": []}, {"nope": 1}, {"has_diagnosis": "*"}, {"a": 1, "b": 2}, "x",
            {"field": {"name": "member_sex", "op": ">", "value": "M"}},
            {"field": {"name": "evil", "op": "==", "value": 1}},
            {"count": {"in": "diagnosis", "match": "A*", "op": ">", "value": -1}},
        ]
        for node in bad:
            with self.subTest(node=node):
                self.assertTrue(conditions.validate_condition(node))
                with self.assertRaises(conditions.ConditionError):
                    conditions.evaluate(node, ep())

    def test_resource_limits(self):
        deep = {"has_diagnosis": "A00"}
        for _ in range(10):
            deep = {"not": deep}
        self.assertTrue(conditions.validate_condition(deep))
        wide = {"all": [{"has_diagnosis": "A00"}] * 16}
        self.assertEqual(conditions.validate_condition(wide), [])
        too_many = {"all": [wide, wide, wide, wide, wide]}
        self.assertTrue(conditions.validate_condition(too_many))

    def test_no_code_execution_surface(self):
        evil = {"has_diagnosis": "__import__('os').system('id')"}
        self.assertTrue(conditions.validate_condition(evil))


class ActionTests(unittest.TestCase):
    def test_remove_keeps_onset_aligned_and_does_not_mutate_input(self):
        e = ep()
        out = actions.apply_action({"action": "remove_code", "target": "diagnosis", "match": "B95*"}, e, automatable=True)
        self.assertTrue(out.changed)
        self.assertEqual(out.episode.diagnosis_code_array, ["A41.0", "E11.9"])
        self.assertEqual(out.episode.condition_onset_array, ["1", "2"])
        self.assertEqual(e.diagnosis_code_array, ["A41.0", "B95.6", "E11.9"])

    def test_remove_principal_promotes_next(self):
        out = actions.apply_action({"action": "remove_code", "target": "diagnosis", "match": "A41.0"}, ep(), automatable=True)
        self.assertEqual(out.episode.principal_diagnosis_code, "B95.6")

    def test_which_variants(self):
        e = ep(diagnosis_code_array=["Z37.0", "Z37.1", "Z37.2"], condition_onset_array=[])
        act = {"action": "remove_code", "target": "diagnosis", "match": "Z37*", "which": "all_but_first"}
        self.assertEqual(actions.apply_action(act, e, automatable=True).episode.diagnosis_code_array, ["Z37.0"])
        act["which"] = "last"
        self.assertEqual(actions.apply_action(act, e, automatable=True).episode.diagnosis_code_array, ["Z37.0", "Z37.1"])

    def test_resequence(self):
        out = actions.apply_action({"action": "resequence_principal", "new_principal": "E11*"}, ep(), automatable=True)
        self.assertEqual(out.episode.diagnosis_code_array[0], "E11.9")
        self.assertEqual(out.episode.condition_onset_array[0], "2")
        self.assertEqual(out.episode.principal_diagnosis_code, "E11.9")

    def test_replace(self):
        act = {"action": "replace_code", "target": "procedure", "match": "40903-00", "with": "40903-01"}
        out = actions.apply_action(act, ep(), automatable=True)
        self.assertEqual(out.episode.procedure_code_array, ["40903-01"])

    def test_automatable_false_is_a_hard_override(self):
        act = {"action": "remove_code", "target": "diagnosis", "match": "B95*"}
        out = actions.apply_action(act, ep(), automatable=False)
        self.assertFalse(out.changed)
        self.assertEqual(out.episode.diagnosis_code_array, ep().diagnosis_code_array)

    def test_validation(self):
        self.assertTrue(actions.validate_action({"action": "format_disk"}))
        self.assertTrue(actions.validate_action({"action": "replace_code", "target": "procedure", "match": "A", "with": "B*"}))
        self.assertTrue(actions.validate_action({"action": "remove_code", "target": "diagnosis", "match": "A1", "extra": 1}))


class BuildExecutableTests(unittest.TestCase):
    cond = {"has_diagnosis": "B95*"}

    def test_flag_for_clinical_review_forces_flag_only(self):
        rule, errors = rules_store.build_executable(
            "R1", "flag_for_clinical_review", self.cond,
            {"action": "remove_code", "target": "diagnosis", "match": "B95*"})
        self.assertEqual(errors, [])
        self.assertEqual(rule.correction_action, {"action": "flag_only"})
        self.assertFalse(rule.automatable)

    def test_action_must_match_recommendation(self):
        _, errors = rules_store.build_executable("R1", "remove_code", self.cond, {"action": "flag_only"})
        self.assertTrue(any("does not implement" in e for e in errors))

    def test_missing_action_rejected_but_narrative_only_allowed(self):
        _, errors = rules_store.build_executable("R1", "remove_code", self.cond, None)
        self.assertTrue(errors)
        rule, errors = rules_store.build_executable("R1", "remove_code", None, None)
        self.assertEqual((rule, errors), (None, []))

    def test_jsonl_roundtrip_and_remap(self):
        rule, _ = rules_store.build_executable("R1", "remove_code", self.cond,
                                               {"action": "remove_code", "target": "diagnosis", "match": "B95*"})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.jsonl")
            rules_store.save_all(path, rules_store.remap_ids([rule], {"R1": "RULE_051"}))
            loaded = rules_store.load_all(path)
            self.assertEqual(loaded[0].rule_id, "RULE_051")
            self.assertTrue(loaded[0].automatable)
            with open(path, "w") as fh:
                fh.write(json.dumps({"rule_id": "x", "flag_condition": {"bad": 1}, "correction_action": {"action": "flag_only"}}) + "\n")
            with self.assertRaises(ValueError):
                rules_store.load_all(path)


class CompareTests(unittest.TestCase):
    def test_flag_correct_regroup_loop_with_mock_grouper(self):
        rule, _ = rules_store.build_executable("R1", "remove_code", {"all": [{"has_diagnosis": "A41.0"}, {"has_diagnosis": "B95.6"}]},
                                               {"action": "remove_code", "target": "diagnosis", "match": "B95.6"})
        e = ep()
        results = list(run_episode_against_ruleset(e, [rule], MockGrouperAdapter()))
        self.assertTrue(results[0].was_flagged and results[0].corrected)
        self.assertEqual(results[0].before.cost_weight, 4.0)
        self.assertEqual(results[0].after.cost_weight, 3.0)
        self.assertEqual(results[0].cost_weight_delta, -1.0)
        self.assertTrue(results[0].preview_only)
        self.assertTrue(any("MOCK" in n for n in results[0].notes))

    def test_not_flagged(self):
        rule, _ = rules_store.build_executable("R1", "remove_code", {"has_diagnosis": "Z99*"},
                                               {"action": "remove_code", "target": "diagnosis", "match": "Z99*"})
        result = next(run_episode_against_ruleset(ep(), [rule], MockGrouperAdapter()))
        self.assertFalse(result.was_flagged)

    def test_grouper_factory_validation(self):
        self.assertIsInstance(load_grouper(None), MockGrouperAdapter)
        with self.assertRaises(ValueError):
            load_grouper("os.system('x')")


class EpisodeTests(unittest.TestCase):
    def test_from_dict_strict(self):
        e = Episode.from_dict({"episode_id": "1", "diagnosis_code_array": ["A41.0", "B95.6"], "member_age": 3})
        self.assertEqual(e.principal_diagnosis_code, "A41.0")
        with self.assertRaises(ValueError):
            Episode.from_dict({"surprise": 1})
        with self.assertRaises(ValueError):
            Episode.from_dict({"diagnosis_code_array": "A41.0"})
        with self.assertRaises(ValueError):
            Episode.from_dict({"diagnosis_code_array": ["A"], "condition_onset_array": ["1", "2"]})

    def test_principal_moves_to_front(self):
        e = Episode(diagnosis_code_array=["A", "B"], principal_diagnosis_code="B", condition_onset_array=["1", "2"])
        self.assertEqual(e.diagnosis_code_array, ["B", "A"])
        self.assertEqual(e.condition_onset_array, ["2", "1"])


if __name__ == "__main__":
    unittest.main()
