"""CLI smoke tests: version, patch-path, trace-info, attach --replay."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from xpref.cli import main

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "k3-q4-128tok.bin"


class TestCLISmoke(unittest.TestCase):
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("xpref", result.output)
        # CLI --version must stay in lockstep with the VERSION file
        version = (ROOT / "VERSION").read_text().strip()
        self.assertIn(version, result.output)
        # ...and with pyproject.toml [project].version (single source of truth
        # for the release workflow's wheel)
        pyproject = (ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        self.assertIsNotNone(match, "pyproject.toml [project].version not found")
        self.assertEqual(match.group(1), version)
        # ...and with the newest CHANGELOG heading (previously unguarded)
        changelog = (ROOT / "CHANGELOG.md").read_text()
        headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.M)
        self.assertTrue(headings, "no released version heading in CHANGELOG.md")
        self.assertEqual(headings[0], version,
                         "newest CHANGELOG heading must equal VERSION")
        # ...and with web/site.json meta.content_version (the surface that
        # drifted in v0.3.0)
        site = json.loads((ROOT / "web" / "site.json").read_text())
        self.assertEqual(site["meta"]["content_version"], f"v{version}")

    def test_patch_path(self):
        runner = CliRunner()
        result = runner.invoke(main, ["patch-path"])
        self.assertEqual(result.exit_code, 0, result.output)
        p = Path(result.output.strip())
        self.assertTrue(p.name == "llama_cpp_router_logit.patch")
        self.assertTrue(p.exists(), f"patch missing at {p}")

    def test_trace_info(self):
        runner = CliRunner()
        result = runner.invoke(main, ["trace-info", "--trace", str(SAMPLE)])
        self.assertEqual(result.exit_code, 0, result.output)
        import json
        meta = json.loads(result.output)
        self.assertEqual(meta["num_experts"], 896)
        self.assertEqual(meta["num_active"], 16)

    def test_attach_replay(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
            fh.write(b"\0" * (256 * 1024))  # 256 KiB fake checkpoint
            ckpt = fh.name
        try:
            result = runner.invoke(
                main,
                ["attach", "--replay", str(SAMPLE), "--checkpoint", ckpt],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("replay done", result.output)
            self.assertIn("recall=", result.output)
            self.assertIn("proj t/s=", result.output)
        finally:
            Path(ckpt).unlink(missing_ok=True)


class TestCLIInputErrors(unittest.TestCase):
    """v0.4.0: input failures must be one-line `Error:` messages, not tracebacks."""

    def test_bad_trace_reports_error_not_traceback(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            fh.write(b"NOTATRACE")
            bad = fh.name
        try:
            for argv in (["eval", "--trace", bad], ["trace-info", "--trace", bad]):
                result = CliRunner().invoke(main, argv)
                self.assertEqual(result.exit_code, 1, argv)
                self.assertIsInstance(result.exception, SystemExit,
                                      f"{argv} leaked {result.exception!r}")
                self.assertIn("Error: empty trace", result.output)
                self.assertNotIn("Traceback", result.output)
        finally:
            Path(bad).unlink(missing_ok=True)

    def test_missing_ring_reports_error_not_traceback(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
            fh.write(b"\0" * 1024)
            ckpt = fh.name
        try:
            result = CliRunner().invoke(
                main, ["attach", "--ring", "/tmp/xpref-no-such-ring.xring",
                       "--checkpoint", ckpt])
            self.assertEqual(result.exit_code, 1)
            self.assertIsInstance(result.exception, SystemExit)
            self.assertIn("Error: ring not found", result.output)
        finally:
            Path(ckpt).unlink(missing_ok=True)

    def test_malformed_layout_names_the_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            fh.write(b"not json")
            layout = fh.name
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh2:
            fh2.write(b"\0" * 256 * 1024)
            ckpt = fh2.name
        try:
            result = CliRunner().invoke(
                main, ["attach", "--replay", str(SAMPLE), "--checkpoint", ckpt,
                       "--layout", layout])
            self.assertEqual(result.exit_code, 1)
            self.assertIsInstance(result.exception, SystemExit)
            self.assertIn(layout, result.output)
        finally:
            Path(layout).unlink(missing_ok=True)
            Path(ckpt).unlink(missing_ok=True)

    def test_invalid_predictor_params_reported(self):
        cases = [
            (["--topk-weight", "nan"], "topk_weight must be finite"),
            (["--topk-weight", "-1"], "topk_weight must be finite"),
            (["--ngram-weight", "inf"], "ngram_weight must be finite"),
            (["--topk-weight", "0", "--ngram-weight", "0"], "must be > 0"),
            (["--ngram-n", "0"], "ngram_n must be >= 1"),
            (["--ngram-n", "-1"], "ngram_n must be >= 1"),
        ]
        for extra, message in cases:
            result = CliRunner().invoke(main, ["eval", *extra])
            self.assertEqual(result.exit_code, 1, extra)
            self.assertIn(message, result.output, extra)


if __name__ == "__main__":
    unittest.main()
