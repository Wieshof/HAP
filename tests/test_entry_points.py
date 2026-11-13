"""Tests for packaging and CLI entry points."""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import hap_contribution as hc
import hap_experiment as hx


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.pyproject_path = ROOT / "pyproject.toml"

    def test_pyproject_exists(self):
        self.assertTrue(self.pyproject_path.exists(), "pyproject.toml should exist")

    def test_pyproject_structure(self):
        with open(self.pyproject_path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("build-system", data)
        self.assertEqual(data["build-system"]["build-backend"], "setuptools.build_meta")

        self.assertIn("project", data)
        project = data["project"]
        self.assertEqual(project["name"], "hap")
        self.assertIn("version", project)
        self.assertIn("scripts", project)

        scripts = project["scripts"]
        self.assertEqual(scripts.get("hap-experiment"), "hap_experiment:main")
        self.assertEqual(scripts.get("hap-contribution"), "hap_contribution:main")

    def test_entry_point_callables(self):
        self.assertTrue(callable(hx.main))
        self.assertTrue(callable(hc.main))

    def test_entry_point_help(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with self.assertRaises(SystemExit) as cm_exp:
                hx.main(["--help"])
            self.assertEqual(cm_exp.exception.code, 0)

            with self.assertRaises(SystemExit) as cm_con:
                hc.main(["--help"])
            self.assertEqual(cm_con.exception.code, 0)

    def test_entry_point_execution(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir)
                hx.main([
                    "--questions", str(ROOT / "data" / "questions.csv"),
                    "--predictions", str(ROOT / "data" / "smoke_predictions.csv"),
                    "--out", str(out_dir / "exp"),
                ])
                self.assertTrue((out_dir / "exp" / "report.md").exists())
                self.assertTrue((out_dir / "exp" / "results.csv").exists())
                self.assertTrue((out_dir / "exp" / "results.json").exists())

                hc.main([
                    "--questions", str(ROOT / "data" / "questions.csv"),
                    "--predictions", str(ROOT / "data" / "smoke_predictions.csv"),
                    "--out", str(out_dir / "con"),
                ])
                self.assertTrue((out_dir / "con" / "contribution_cases.md").exists())
                self.assertTrue((out_dir / "con" / "buyer_test.md").exists())


if __name__ == "__main__":
    unittest.main()
