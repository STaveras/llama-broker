from io import StringIO
from pathlib import Path
import subprocess
import tempfile
import unittest

from llama_broker.runtime_control import project_add
from llama_broker.runtime_control_ui import project_branches, select_source


class TTY(StringIO):
    def isatty(self):
        return True


class RuntimeControlUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "control"
        repo = Path(self.tmp.name) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "trunk"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "README").write_text("one")
        subprocess.run(["git", "add", "README"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "first subject"], cwd=repo, check=True, capture_output=True)
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        project_add(self.root, "fixture", "https://example.invalid/source.git", default_branch="trunk")
        projects = __import__("json").loads((self.root / "projects.json").read_text())
        projects[0]["path"] = str(repo)
        (self.root / "projects.json").write_text(__import__("json").dumps(projects))

    def tearDown(self):
        self.tmp.cleanup()

    def test_piped_select_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "interactive TTY"):
            select_source(self.root, "fixture", input_stream=StringIO())

    def test_tty_select_returns_full_hash_and_subject(self):
        selected = select_source(self.root, "fixture", input_stream=TTY("1\n1\n"), output_stream=StringIO())
        self.assertEqual(selected["branch"], "trunk")
        self.assertEqual(selected["commit"], self.commit)
        self.assertEqual(selected["subject"], "first subject")

    def test_branches_does_not_assume_main(self):
        self.assertEqual(project_branches(self.root, "fixture"), ["trunk"])


if __name__ == "__main__":
    unittest.main()
