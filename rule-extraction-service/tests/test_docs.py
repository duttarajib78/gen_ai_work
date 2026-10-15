import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocsTests(unittest.TestCase):
    def test_readme_documents_every_setting(self):
        source = (ROOT / "src" / "rule_extractor" / "settings.py").read_text()
        names = {"RULE_EXTRACTOR_" + n for n in re.findall(r'reader\.(?:str|int|float|bool|csv|choice)\(\s*"([A-Z_]+)"', source)}
        names |= {"RULE_EXTRACTOR_BEDROCK_REGION", "RULE_EXTRACTOR_CONFIG_FILE", "RULE_EXTRACTOR_LOG_LEVEL"}
        readme = (ROOT / "README.md").read_text()
        missing = sorted(n for n in names if n not in readme and n.replace("RULE_EXTRACTOR", "") not in readme)
        self.assertEqual(missing, [], f"undocumented settings: {missing}")

    def test_readme_lists_every_operation_and_error_code(self):
        from rule_extractor.settings import ALL_OPERATIONS
        readme = (ROOT / "README.md").read_text()
        for op in ALL_OPERATIONS:
            self.assertIn(f"`{op}`", readme, op)
        from rule_extractor import errors
        for cls in (errors.RequestValidationError, errors.NotFoundError, errors.ConflictError, errors.PolicyError, errors.UnavailableError):
            self.assertIn(f"`{cls.code}`", readme, cls.code)

    def test_readme_project_tree_matches_reality(self):
        readme = (ROOT / "README.md").read_text()
        for rel in ("src/rule_extractor/service/supervisor.py", "src/rule_extractor/agents/sdk_runner.py", "infra/cloudformation/endpoint.yaml",
                    "scripts/deploy.py", "buildspec.yml", "configs/base.yaml", "docker/entrypoint.sh"):
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertIn(Path(rel).name, readme)


if __name__ == "__main__":
    unittest.main()
