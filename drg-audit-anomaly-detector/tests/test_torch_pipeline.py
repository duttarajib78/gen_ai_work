"""End-to-end tests on a tiny model. They need torch, transformers, safetensors and scikit-learn and are skipped
automatically elsewhere. Run them in the dev environment: ``make test``."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.helpers  # noqa: F401,I001  (puts src/ on sys.path)

HAS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "safetensors", "sklearn"))


@unittest.skipUnless(HAS_DEPS, "requires torch, transformers, safetensors and scikit-learn")
class TinyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from drg_audit.config import EncoderSpec, FinetuneConfig, PretrainConfig, RuntimeConfig
        from drg_audit.data.io import dataframe_to_records
        from drg_audit.schemas import TokenizationSpec
        from drg_audit.synthetic import generate_frame
        from drg_audit.training.finetune import run_finetuning
        from drg_audit.training.pretrain import run_pretraining

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.records, report = dataframe_to_records(generate_frame(600, seed=3))
        runtime = RuntimeConfig(seed=7, require_gpu=False, amp=False, num_workers=0)
        pre_cfg = PretrainConfig(
            encoder=EncoderSpec(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64),
            tokenization=TokenizationSpec(max_len=32), min_code_frequency=3, val_fraction=0.2, batch_size=16,
            eval_batch_size=32, epochs=2, early_stopping_patience=2, uniformity_sample_size=100, runtime=runtime)
        ft_cfg = FinetuneConfig(val_fraction=0.2, batch_size=16, eval_batch_size=32, epochs=2,
                                early_stopping_patience=2, classifier_hidden_dim=16, runtime=runtime)
        cls.pre_dir, cls.bundle_dir = root / "pretrain", root / "bundle"
        provide = (cls.records, report)
        with mock.patch("drg_audit.training.pretrain.load_records", return_value=provide), \
                mock.patch("drg_audit.training.finetune.load_records", return_value=provide):
            cls.pre_metrics = run_pretraining(pre_cfg, input_path="unused", output_dir=cls.pre_dir)
            cls.ft_info = run_finetuning(ft_cfg, input_path="unused", pretrained_dir=cls.pre_dir,
                                         output_dir=cls.bundle_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def engine(self, **runtime):
        from drg_audit.inference.engine import AuditEngine
        from drg_audit.settings import RuntimeSettings
        return AuditEngine.from_bundle_dir(self.bundle_dir, settings=RuntimeSettings(device="cpu", **runtime))

    def test_training_produced_verifiable_artifacts(self):
        from drg_audit.artifacts import load_bundle, load_encoder_checkpoint
        bundle = load_bundle(self.bundle_dir)
        self.assertEqual(bundle.tokenization.name, "standard-v1")
        self.assertEqual(bundle.manifest["training"]["pretrained_bundle_id"],
                         load_encoder_checkpoint(self.pre_dir).manifest["bundle_id"])
        self.assertGreaterEqual(self.ft_info["best_epoch"], 1)
        self.assertEqual(self.pre_metrics["dataset"]["train_rows"] + self.pre_metrics["dataset"]["val_rows"], 600)

    def test_engine_scores_every_record_consistently(self):
        results = self.engine().predict(self.records[:40])
        self.assertEqual(len(results), 40)
        for r in results:
            self.assertEqual(r.status, "ok")
            for p in (r.mdc_confidence, r.base_confidence, r.split_confidence, r.prob_assigned_split):
                self.assertTrue(0.0 <= p <= 1.0)
            self.assertEqual(r.is_billing_anomaly,
                             r.mdc_discrepancy_flag or r.base_discrepancy_flag or r.split_inflation_flag)
            if not r.is_billing_anomaly:
                self.assertEqual(r.audit_priority_score, 0.0)

    def test_prediction_is_deterministic_and_batch_invariant(self):
        subset = self.records[:25]
        first = self.engine().predict(subset)
        again = self.engine().predict(subset)
        self.assertEqual([r.to_dict() for r in first], [r.to_dict() for r in again])
        small = self.engine(max_batch_size=3, dynamic_padding=False).predict(subset)
        # Different batch size and padding width change float rounding only; an argmax flip needs a near-tie.
        agree = sum((a.predicted_mdc, a.predicted_base_drg, a.predicted_complexity_split)
                    == (b.predicted_mdc, b.predicted_base_drg, b.predicted_complexity_split)
                    for a, b in zip(first, small))
        self.assertGreaterEqual(agree, len(subset) - 1)
        for a, b in zip(first, small):
            self.assertAlmostEqual(a.split_confidence, b.split_confidence, delta=1e-3)

    def test_skipped_records_and_version_policy(self):
        from dataclasses import replace
        base = self.records[0]
        short = replace(base, admission_no="s1", drg="F6")
        unseen = replace(base, admission_no="u1", drg_vers="99.0")
        results = self.engine().predict([short, unseen])
        self.assertEqual((results[0].status, results[0].reason), ("skipped", "invalid_assigned_drg"))
        self.assertIn("unknown_drg_version_fallback", results[1].warnings)
        strict = self.engine(unknown_version_policy="reject").predict([unseen])[0]
        self.assertEqual((strict.status, strict.reason), ("skipped", "unknown_drg_version"))

    def test_handler_round_trip(self):
        import inference
        from drg_audit.client import record_to_payload
        service = inference.model_fn(str(self.bundle_dir))
        payload = json.dumps({"records": [record_to_payload(r) for r in self.records[:5]]})
        body, content_type = inference.output_fn(
            inference.predict_fn(inference.input_fn(payload, "application/json"), service), "application/json")
        parsed = json.loads(body)
        self.assertEqual(content_type, "application/json")
        self.assertEqual(parsed["summary"]["n_records"], 5)
        self.assertEqual(parsed["summary"]["n_rejected"], 0)

    def test_tampered_bundle_does_not_load(self):
        from drg_audit.artifacts import WEIGHTS_NAME
        from drg_audit.errors import ArtifactError
        from drg_audit.inference.engine import AuditEngine
        copy = Path(self._tmp.name) / "tampered"
        shutil.copytree(self.bundle_dir, copy)
        with open(copy / WEIGHTS_NAME, "r+b") as handle:
            handle.seek(-8, 2)
            handle.write(b"\x00" * 8)
        with self.assertRaisesRegex(ArtifactError, "integrity"):
            AuditEngine.from_bundle_dir(copy)

    def test_export_import_round_trip_preserves_outputs(self):
        import torch
        from drg_audit.artifacts import load_bundle
        from drg_audit.models.joint import DrgAuditModel
        bundle = load_bundle(self.bundle_dir)
        model = DrgAuditModel.from_bundle(bundle).eval()
        clone = DrgAuditModel.build(bundle.encoder_config, bundle.labels.sizes(),
                                    int(bundle.classifier_config["hidden_dim"])).eval()
        clone.load_exported(model.export_tensors())
        ids = torch.randint(5, len(bundle.vocab), (3, 16))
        args = (ids, torch.ones_like(ids), torch.rand(3, 4), torch.zeros(3, dtype=torch.long))
        with torch.inference_mode():
            for a, b in zip(model(*args), clone(*args)):
                self.assertTrue(torch.equal(a, b))

    def test_validation_masking_is_deterministic_and_training_masking_is_not(self):
        import torch
        from drg_audit.data.datasets import ContrastiveCollator
        items = [(i, torch.arange(5, 5 + 20), torch.zeros(4), torch.tensor(0), torch.zeros(3, dtype=torch.long))
                 for i in range(8)]
        deterministic = ContrastiveCollator(mask_id=4, pad_id=0, probability=0.5, deterministic_seed=1)
        self.assertTrue(torch.equal(deterministic(items).input_ids, deterministic(items).input_ids))
        batch = deterministic(items)
        self.assertEqual(batch.input_ids.shape[0], 16)  # interleaved clean / masked
        self.assertTrue(torch.equal(batch.input_ids[0::2], torch.stack([it[1] for it in items])))  # clean views intact
        self.assertTrue((batch.input_ids[1::2] == 4).any())

    def test_contrastive_loss_prefers_aligned_views(self):
        import torch
        from drg_audit.training.losses import SimCSELoss
        base = torch.randn(16, 8)
        aligned = torch.stack([base, base + 0.01 * torch.randn(16, 8)], dim=1).reshape(32, 8)
        random_pairs = torch.randn(32, 8)
        loss = SimCSELoss(0.3)
        self.assertLess(loss(aligned).item(), loss(random_pairs).item())

    def test_legacy_encoder_buffers_are_ignored_on_load(self):
        import torch
        from drg_audit.models.joint import filter_encoder_state
        state = {"roberta.embeddings.position_ids": torch.zeros(1), "roberta.embeddings.word_embeddings.weight": torch.zeros(2)}
        self.assertEqual(list(filter_encoder_state(state)), ["roberta.embeddings.word_embeddings.weight"])


if __name__ == "__main__":
    unittest.main()
