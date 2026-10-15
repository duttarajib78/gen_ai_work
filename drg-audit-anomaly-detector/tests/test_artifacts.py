import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_labels, make_vocab  # noqa: I001

from drg_audit.artifacts import (
    MANIFEST_NAME, VOCAB_NAME, WEIGHTS_NAME, load_bundle, load_encoder_checkpoint, write_bundle,
    write_encoder_checkpoint,
)
from drg_audit.errors import ArtifactError
from drg_audit.schemas import TokenizationSpec


def weights(content=b"weights"):
    def write(path):
        path.write_bytes(content)
    return write


def write_sample(root, vocab=None, encoder_vocab_size=None, content=b"weights"):
    vocab = vocab or make_vocab(["a", "b"])
    return write_bundle(
        root, write_weights=weights(content),
        encoder_config={"vocab_size": encoder_vocab_size or len(vocab), "hidden_size": 8},
        classifier_config={"hidden_dim": 16},
        vocab=vocab, labels=make_labels(), inference_config={"decision": {"confidence_threshold": 0.85}},
        tokenization=TokenizationSpec.legacy_training(), training_info={"epochs": 3},
    )


class BundleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "bundle"

    def test_roundtrip(self):
        write_sample(self.root)
        bundle = load_bundle(self.root)
        self.assertEqual(len(bundle.model_version), 16)
        self.assertEqual(bundle.tokenization.name, "legacy-training-v1")
        self.assertFalse(bundle.tokenization.sort_medical_claims)
        self.assertEqual(bundle.manifest["training"], {"epochs": 3})
        self.assertEqual(bundle.labels.sizes(), {"version": 2, "mdc": 2, "base": 2, "split": 4})

    def test_bundle_id_is_content_derived(self):
        write_sample(self.root)
        write_sample(self.root.parent / "other")
        write_sample(self.root.parent / "different", content=b"other weights")
        ids = [load_bundle(p).model_version for p in (self.root, self.root.parent / "other",
                                                      self.root.parent / "different")]
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])

    def test_tampered_weights_fail_closed(self):
        write_sample(self.root)
        (self.root / WEIGHTS_NAME).write_bytes(b"evil weights")
        with self.assertRaisesRegex(ArtifactError, "integrity"):
            load_bundle(self.root)
        load_bundle(self.root, verify=False)  # explicit opt-out still loads

    def test_tampered_vocab_fails_closed(self):
        write_sample(self.root)
        (self.root / VOCAB_NAME).write_text(json.dumps({"[PAD]": 0}))
        with self.assertRaisesRegex(ArtifactError, "integrity"):
            load_bundle(self.root)

    def test_missing_file(self):
        write_sample(self.root)
        (self.root / WEIGHTS_NAME).unlink()
        with self.assertRaisesRegex(ArtifactError, "missing"):
            load_bundle(self.root)

    def test_manifest_missing_or_wrong_kind(self):
        with self.assertRaises(ArtifactError):
            load_bundle(self.root)
        write_sample(self.root)
        with self.assertRaisesRegex(ArtifactError, "kind"):
            load_encoder_checkpoint(self.root)

    def test_manifest_path_traversal_is_refused(self):
        write_sample(self.root)
        manifest_path = self.root / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["../evil"] = {"sha256": "0" * 64, "bytes": 1}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ArtifactError, "unsafe"):
            load_bundle(self.root)

    def test_vocab_validation(self):
        bad_order = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        write_sample(self.root, vocab=bad_order)
        with self.assertRaisesRegex(ArtifactError, "special tokens"):
            load_bundle(self.root)

    def test_vocab_size_mismatch(self):
        write_sample(self.root, encoder_vocab_size=999)
        with self.assertRaisesRegex(ArtifactError, "vocab_size"):
            load_bundle(self.root)

    def test_encoder_checkpoint_roundtrip(self):
        vocab = make_vocab(["a"])
        write_encoder_checkpoint(self.root, write_weights=weights(), encoder_config={"vocab_size": len(vocab)},
                                 vocab=vocab, tokenization=TokenizationSpec.standard(), metrics={"val_loss": 1.0})
        checkpoint = load_encoder_checkpoint(self.root)
        self.assertEqual(checkpoint.tokenization.name, "standard-v1")
        with self.assertRaisesRegex(ArtifactError, "kind"):
            load_bundle(self.root)

    def test_label_map_must_be_contiguous(self):
        write_sample(self.root)
        path = self.root / "label_maps.json"
        data = json.loads(path.read_text())
        data["mdc"] = {"F": 0, "G": 5}
        path.write_text(json.dumps(data))
        with self.assertRaises(ArtifactError):
            load_bundle(self.root, verify=False)


if __name__ == "__main__":
    unittest.main()
