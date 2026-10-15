import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from rule_extractor.errors import IntegrityError
from rule_extractor.model_package import MANIFEST_NAME, build_model_dir, package_tar, verify_model_dir

from .helpers import make_env


class ModelPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.settings = make_env(root)
        rules = root / "rules.jsonl"
        rules.write_text("")
        self.model = root / "model"
        build_model_dir(self.settings.paths.corpus_dir, root / "template.xlsx", self.model,
                        base_rules_path=rules, evaluation_rules_path=rules)

    def test_fresh_model_dir_verifies(self):
        self.assertEqual(verify_model_dir(self.model), [])
        for rel in ("corpus/manifest.json", "corpus/acs/p0001.txt", "templates/rule_catalogue.xlsx",
                    "base/executable_rules.jsonl", "rules/executable_rules.jsonl", MANIFEST_NAME):
            self.assertTrue((self.model / rel).is_file(), rel)

    def test_tampering_is_detected(self):
        target = self.model / "corpus" / "acs" / "p0003.txt"
        target.write_text("tampered")
        self.assertTrue(any("checksum mismatch" in p for p in verify_model_dir(self.model)))

    def test_extra_missing_and_symlink_files_are_detected(self):
        (self.model / "corpus" / "extra.txt").write_text("x")
        os.symlink("/etc/passwd", self.model / "corpus" / "link.txt")
        (self.model / "corpus" / "acs" / "p0002.txt").unlink()
        problems = "\n".join(verify_model_dir(self.model))
        for fragment in ("unexpected file: corpus/extra.txt", "symlink not allowed: corpus/link.txt", "missing or not a regular file: corpus/acs/p0002.txt"):
            self.assertIn(fragment, problems)

    def test_missing_or_bad_manifest(self):
        (self.model / MANIFEST_NAME).write_text("{not json")
        self.assertTrue(verify_model_dir(self.model)[0].startswith("manifest.json unreadable"))
        (self.model / MANIFEST_NAME).unlink()
        self.assertIn("missing", verify_model_dir(self.model)[0])

    def test_unsafe_manifest_paths_rejected(self):
        import json
        manifest = json.loads((self.model / MANIFEST_NAME).read_text())
        manifest["files"]["../outside"] = {"sha256": "0" * 64, "size": 1}
        (self.model / MANIFEST_NAME).write_text(json.dumps(manifest))
        self.assertTrue(any("unsafe path" in p for p in verify_model_dir(self.model)))

    def test_tar_is_deterministic_and_round_trips(self):
        out1, out2 = Path(self.tmp.name) / "a.tar.gz", Path(self.tmp.name) / "b.tar.gz"
        self.assertEqual(package_tar(self.model, out1), package_tar(self.model, out2))
        extract = Path(self.tmp.name) / "extract"
        with tarfile.open(out1) as tar:
            names = tar.getnames()
            self.assertIn("manifest.json", names)
            self.assertFalse(any(n.startswith("/") or ".." in n.split("/") for n in names))
            tar.extractall(extract, filter="data")
        self.assertEqual(verify_model_dir(extract), [])

    def test_refuses_to_package_unhealthy_dir_and_non_empty_target(self):
        (self.model / "corpus" / "extra.txt").write_text("x")
        with self.assertRaises(IntegrityError):
            package_tar(self.model, Path(self.tmp.name) / "x.tar.gz")
        with self.assertRaises(IntegrityError):
            build_model_dir(self.settings.paths.corpus_dir, Path(self.tmp.name) / "template.xlsx", self.model)


if __name__ == "__main__":
    unittest.main()
