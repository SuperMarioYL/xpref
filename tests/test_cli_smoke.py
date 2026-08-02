"""CLI smoke tests: version, patch-path, trace-info, attach --replay."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
