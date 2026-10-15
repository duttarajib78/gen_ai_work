import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_labels, make_vocab  # noqa: I001

import _packaging as pk
import build_model_package as bmp

from drg_audit.artifacts import WEIGHTS_NAME, write_bundle
from drg_audit.errors import ArtifactError
from drg_audit.schemas import TokenizationSpec


def write_stub_bundle(root: Path, content: bytes = b"weights") -> Path:
    vocab = make_vocab(["a", "b"])
    return write_bundle(
        root, write_weights=lambda p: p.write_bytes(content),
        encoder_config={"vocab_size": len(vocab)}, classifier_config={"hidden_dim": 16}, vocab=vocab,
        labels=make_labels(), inference_config={}, tokenization=TokenizationSpec.standard())


def names_in(tar_path: Path) -> set[str]:
    with tarfile.open(tar_path) as tar:
        return {m.name for m in tar.getmembers()}


class BuildPackageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bundle = write_stub_bundle(self.root / "bundle")
        self.out = self.root / "dist" / "model.tar.gz"

    def build(self, **kwargs):
        kwargs.setdefault("dependency_mode", "pypi")
        return bmp.build_package(bundle_dir=self.bundle, output=self.out, **kwargs)

    def test_package_layout_and_least_code(self):
        meta = self.build()
        names = names_in(self.out)
        for expected in ("manifest.json", WEIGHTS_NAME, "vocab.json", "label_maps.json", "classifier_config.json",
                         "encoder_config.json", "inference_config.json", "code/inference.py",
                         "code/requirements.txt", "code/drg_audit/serving/service.py",
                         "code/drg_audit/inference/engine.py", "code/drg_audit/models/joint.py"):
            self.assertIn(expected, names)
        for forbidden in ("code/drg_audit/training", "code/drg_audit/data", "code/drg_audit/postprocess",
                          "code/drg_audit/cli.py", "code/drg_audit/client.py", "code/drg_audit/config.py",
                          "code/drg_audit/synthetic.py", "code/drg_audit/archive.py"):
            self.assertFalse(any(n == forbidden or n.startswith(forbidden + "/") for n in names), forbidden)
        self.assertEqual(meta["dependency_mode"], "pypi")

    def test_sidecar_files_match_the_package(self):
        meta = self.build()
        sha_line = (self.out.parent / "model.tar.gz.sha256").read_text().split()
        self.assertEqual(sha_line[0], meta["sha256"])
        self.assertEqual(meta["sha256"], pk.sha256_file(self.out))
        stored = json.loads((self.out.parent / "package_meta.json").read_text())
        self.assertEqual(stored["bundle_id"], meta["bundle_id"])
        self.assertEqual(stored["bytes"], self.out.stat().st_size)

    def test_build_is_reproducible(self):
        first = self.build()["sha256"]
        second = self.build()["sha256"]
        self.assertEqual(first, second)

    def test_pypi_requirements_have_no_offline_flags(self):
        self.build()
        with tarfile.open(self.out) as tar:
            text = tar.extractfile("code/requirements.txt").read().decode()
        self.assertIn("transformers==", text)
        self.assertIn("safetensors==", text)
        self.assertNotIn("--no-index", text)

    def test_wheelhouse_mode_writes_offline_requirements(self):
        def fake_vendor(requirements, wheelhouse, *, python_version, platforms):
            wheelhouse.mkdir(parents=True)
            (wheelhouse / "safetensors-0.4.5-py3-none-any.whl").write_bytes(b"wheel")
            return [{"file": "safetensors-0.4.5-py3-none-any.whl", "sha256": "x", "bytes": 5}]

        with mock.patch.object(pk, "vendor_wheels", side_effect=fake_vendor):
            meta = self.build(dependency_mode="wheelhouse")
        with tarfile.open(self.out) as tar:
            text = tar.extractfile("code/requirements.txt").read().decode()
        self.assertIn("--no-index", text)
        self.assertIn("--find-links=/opt/ml/model/code/wheelhouse", text)
        self.assertIn("code/wheelhouse/safetensors-0.4.5-py3-none-any.whl", names_in(self.out))
        self.assertEqual(len(meta["wheelhouse"]), 1)

    def test_tampered_bundle_is_refused(self):
        (self.bundle / WEIGHTS_NAME).write_bytes(b"tampered")
        with self.assertRaisesRegex(ArtifactError, "integrity"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_invalid_dependency_mode(self):
        with self.assertRaises(ValueError):
            self.build(dependency_mode="download-at-runtime")

    def test_import_closure_detects_missing_modules(self):
        code = self.root / "code"
        (code / "drg_audit").mkdir(parents=True)
        (code / "drg_audit" / "__init__.py").write_text("")
        (code / "inference.py").write_text("from drg_audit.nope import thing\nimport drg_audit.also_missing\n")
        with self.assertRaisesRegex(RuntimeError, "nope"):
            pk.assert_import_closure(code)
        (code / "inference.py").write_text("from drg_audit import __version__\n")
        pk.assert_import_closure(code)  # resolvable -> no error

    def test_cli_accepts_a_training_job_artifact(self):
        artifact = self.root / "training_model.tar.gz"
        with tarfile.open(artifact, "w:gz") as tar:
            for path in sorted(self.bundle.iterdir()):
                tar.add(path, arcname=f"./{path.name}")
        code = bmp.main(["--model-tar", str(artifact), "--output", str(self.out), "--dependency-mode", "pypi"])
        self.assertEqual(code, 0)
        self.assertTrue(self.out.exists())

    def test_requirement_includes_are_flattened(self):
        (self.root / "base.in").write_text("a==1\n# comment\nb==2  # inline\n")
        (self.root / "top.in").write_text("-r base.in\nc==3\na==1\n")
        self.assertEqual(pk.read_requirements(self.root / "top.in"), ["a==1", "b==2", "c==3"])
        (self.root / "bad.in").write_text("--index-url http://evil.example/simple\n")
        with self.assertRaises(ValueError):
            pk.read_requirements(self.root / "bad.in")


if __name__ == "__main__":
    unittest.main()
