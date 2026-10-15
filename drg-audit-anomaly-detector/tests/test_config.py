import unittest

from tests.helpers import make_labels  # noqa: F401,I001

from drg_audit._mapping import dataclass_from_mapping
from drg_audit.errors import ArtifactError, ConfigError
from drg_audit.schemas import LabelMaps, TokenizationSpec
from drg_audit.settings import RuntimeSettings, ServingConfig


class MappingTests(unittest.TestCase):
    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ConfigError, "unknown"):
            dataclass_from_mapping(TokenizationSpec, {"max_length": 128})

    def test_type_checks_and_coercion(self):
        spec = dataclass_from_mapping(TokenizationSpec, {"max_len": 64, "sort_medical_claims": False})
        self.assertEqual((spec.max_len, spec.sort_medical_claims), (64, False))
        with self.assertRaises(ConfigError):
            dataclass_from_mapping(TokenizationSpec, {"max_len": "64"})
        with self.assertRaises(ConfigError):
            dataclass_from_mapping(TokenizationSpec, {"sort_medical_claims": 1})
        with self.assertRaises(ConfigError):  # validate() hook
            dataclass_from_mapping(TokenizationSpec, {"max_len": 4})

    def test_string_floats_from_yaml_are_coerced(self):
        cfg = ServingConfig.from_mapping({"decision": {"confidence_threshold": "0.9"}})
        self.assertEqual(cfg.decision.confidence_threshold, 0.9)

    def test_nested_unknown_key_path_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "limits"):
            ServingConfig.from_mapping({"limits": {"bogus": 1}})


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        cfg = ServingConfig.from_mapping({})
        self.assertEqual(cfg.decision.confidence_threshold, 0.85)
        self.assertEqual(cfg.runtime.precision, "fp32")

    def test_env_overrides(self):
        cfg = ServingConfig.from_mapping({}).with_env({
            "DRG_AUDIT_CONFIDENCE_THRESHOLD": "0.9", "DRG_AUDIT_MAX_RECORDS": "32",
            "DRG_AUDIT_DEVICE": "CPU", "DRG_AUDIT_TORCH_THREADS": "2", "DRG_AUDIT_PRECISION": "bf16"})
        self.assertEqual(cfg.decision.confidence_threshold, 0.9)
        self.assertEqual(cfg.limits.max_records, 32)
        self.assertEqual((cfg.runtime.device, cfg.runtime.torch_threads, cfg.runtime.precision),
                         ("cpu", 2, "bf16"))

    def test_invalid_env_values_fail_fast(self):
        base = ServingConfig.from_mapping({})
        for env in ({"DRG_AUDIT_DEVICE": "tpu"}, {"DRG_AUDIT_PRECISION": "int4"},
                    {"DRG_AUDIT_MAX_RECORDS": "abc"}, {"DRG_AUDIT_CONFIDENCE_THRESHOLD": "nan"},
                    {"DRG_AUDIT_CONFIDENCE_THRESHOLD": "1.5"}, {"DRG_AUDIT_TORCH_THREADS": "0"},
                    {"DRG_AUDIT_UNKNOWN_VERSION_POLICY": "ignore"}):
            with self.subTest(env=env), self.assertRaises(ConfigError):
                base.with_env(env)

    def test_runtime_validation(self):
        with self.assertRaises(ConfigError):
            RuntimeSettings(max_batch_size=0).validate()


class LabelMapTests(unittest.TestCase):
    def test_rejects_empty_and_non_contiguous(self):
        good = make_labels().to_dict()
        self.assertEqual(LabelMaps.from_mapping(good).sizes()["split"], 4)
        for mutate in (lambda d: d.update(mdc={}), lambda d: d.pop("base"), lambda d: d.update(split={"A": 1})):
            bad = make_labels().to_dict()
            mutate(bad)
            with self.assertRaises(ArtifactError):
                LabelMaps.from_mapping(bad)


if __name__ == "__main__":
    unittest.main()
