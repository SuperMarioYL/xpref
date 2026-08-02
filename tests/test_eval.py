"""Eval-path tests: the ``evaluate`` function and the ``xpref eval`` command."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from xpref.cli import main
from xpref.predictor import evaluate
from xpref.trace import read_trace

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "k3-q4-128tok.bin"


class TestEval(unittest.TestCase):
    def test_evaluate_function(self):
        tr = read_trace(SAMPLE)
        res = evaluate(tr.logits, tr.fired)
        for key in ("recall_at_k", "recall_at_16", "per_layer_recall",
                    "predictions_made", "num_experts", "num_active"):
            self.assertIn(key, res)
        self.assertEqual(len(res["per_layer_recall"]), tr.num_layers)

    def test_eval_cli_human(self):
        runner = CliRunner()
        result = runner.invoke(main, ["eval", "--trace", str(SAMPLE)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("recall@16", result.output)
        self.assertIn("xpref  t/s", result.output)

    def test_eval_cli_json(self):
        runner = CliRunner()
        result = runner.invoke(main, ["eval", "--trace", str(SAMPLE), "--json"])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertGreater(data["recall_at_k"], 0.6)
        self.assertIn("projected_predictive_tps", data)


if __name__ == "__main__":
    unittest.main()
