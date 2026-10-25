"""Tests for src/hap_experiment.py. Run with: python -m unittest discover tests"""
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hap_experiment as hx  # noqa: E402


def make_dataset(rows, agents=None, split=None):
    """rows: {question_id: (outcome, {agent_id: p})}"""
    questions = {
        q: hx.Question(q, q, y, (split or {}).get(q, "test")) for q, (y, _) in rows.items()
    }
    ids = sorted({a for _, ps in rows.values() for a in ps})
    agents = agents or {a: hx.Agent(a, a, a, a, 1.0) for a in ids}
    forecasts = {(q, a): p for q, (_, ps) in rows.items() for a, p in ps.items()}
    return hx.Dataset(questions, agents, forecasts)


class MetricTests(unittest.TestCase):
    def test_brier_and_log_loss(self):
        self.assertAlmostEqual(hx.brier(0.7, 1), 0.09)
        self.assertAlmostEqual(hx.brier(0.7, 0), 0.49)
        self.assertAlmostEqual(hx.log_loss(0.5, 1), math.log(2))
        self.assertTrue(math.isfinite(hx.log_loss(0.0, 1)))

    def test_wrong_is_symmetric_at_half(self):
        self.assertFalse(hx.is_wrong(0.5, 0))
        self.assertFalse(hx.is_wrong(0.5, 1))
        self.assertTrue(hx.is_wrong(0.49, 1))
        self.assertTrue(hx.is_wrong(0.51, 0))

    def test_confidently_wrong(self):
        self.assertTrue(hx.is_confidently_wrong(0.75, 0))
        self.assertTrue(hx.is_confidently_wrong(0.25, 1))
        self.assertFalse(hx.is_confidently_wrong(0.74, 0))
        self.assertFalse(hx.is_confidently_wrong(0.9, 1))

    def test_pearson(self):
        self.assertAlmostEqual(hx.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(hx.pearson([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertEqual(hx.pearson([1, 1, 1], [1, 2, 3]), 0.0)
        with self.assertRaises(ValueError):
            hx.pearson([1, 2], [1, 2, 3])

    def test_provenance_overlap(self):
        a = hx.Agent("a", "gpt", "openai", "web", 1)
        b = hx.Agent("b", "gpt", "openai", "offline", 1)
        c = hx.Agent("c", "llama", "self", "offline", 1)
        self.assertAlmostEqual(hx.provenance_overlap(a, b), 2 / 3)
        self.assertEqual(hx.provenance_overlap(a, c), 0.0)


class EvaluateTests(unittest.TestCase):
    def test_pool_counts(self):
        data = make_dataset({
            "q1": (1, {"a": 0.2, "b": 0.1}),   # both wrong, group 0.15: confidently wrong
            "q2": (0, {"a": 0.4, "b": 0.6}),   # only b wrong
            "q3": (1, {"a": 0.9, "b": 0.8}),   # both right
        })
        r = hx.evaluate_pool(data, "x", ["a", "b"])
        self.assertEqual(r.n_questions, 3)
        self.assertEqual(r.everyone_wrong, 1)
        self.assertEqual(r.confidently_wrong, 1)
        self.assertAlmostEqual(r.brier, (0.85 ** 2 + 0.5 ** 2 + 0.15 ** 2) / 3)
        self.assertAlmostEqual(r.tail_brier, 0.85 ** 2)

    def test_missing_forecast_raises(self):
        data = make_dataset({"q1": (1, {"a": 0.2, "b": 0.1})})
        del data.forecasts[("q1", "b")]
        with self.assertRaises(ValueError):
            hx.evaluate_pool(data, "x", ["a", "b"])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        q, p = hx.generate_smoke_data(Path(self.dir.name))
        self.data = hx.load_dataset(q, p)

    def tearDown(self):
        self.dir.cleanup()

    def test_fixture_shape(self):
        self.assertEqual(len(self.data.question_ids("train")), 12)
        self.assertEqual(len(self.data.question_ids("test")), 12)
        self.assertEqual(len(self.data.agents), len(hx.SMOKE_AGENTS))

    def test_fixture_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            q, p = hx.generate_smoke_data(Path(d))
            self.assertEqual(hx.load_dataset(q, p), self.data)

    def test_same_model_is_one_family(self):
        ids = hx.select_same_model(self.data, 4)
        self.assertEqual(len({self.data.agents[a].model_family for a in ids}), 1)

    def test_naive_mix_is_one_per_family(self):
        ids = hx.select_naive_mix(self.data, 4)
        self.assertEqual(len({self.data.agents[a].model_family for a in ids}), 4)

    def test_hap_with_zero_weights_is_accuracy_ranking(self):
        ids = hx.select_hap(self.data, 4, hx.HAPWeights(0, 0, 0))
        qids = self.data.question_ids("train")
        self.assertEqual(ids, hx._ranked_by_brier(self.data, self.data.agents, qids)[:4])

    def test_selection_ignores_test_split(self):
        # Flipping every test outcome must not change any selection.
        flipped = hx.Dataset(
            {q: (v if v.split == "train" else hx.Question(q, v.text, 1 - v.outcome, v.split))
             for q, v in self.data.questions.items()},
            self.data.agents, self.data.forecasts,
        )
        for select in (hx.select_same_model, hx.select_naive_mix, hx.select_hap):
            self.assertEqual(select(self.data, 4), select(flipped, 4))

    def test_run_writes_outputs(self):
        results = hx.run(self.data)
        out = Path(self.dir.name) / "out"
        hx.write_outputs(results, out)
        with open(out / "results.csv") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)
        self.assertIn("HAP diversity-aware", (out / "report.md").read_text())


class LoadTests(unittest.TestCase):
    def test_rejects_inconsistent_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            q, p = Path(d) / "q.csv", Path(d) / "p.csv"
            q.write_text("question_id,text,outcome,split\nq1,x,1,test\nq2,y,0,test\n")
            p.write_text(
                "question_id,agent_id,model_family,operator,pipeline,price,probability\n"
                "q1,a,gpt,openai,web,1,0.5\nq2,a,llama,openai,web,1,0.5\n"
            )
            with self.assertRaises(ValueError):
                hx.load_dataset(q, p)


if __name__ == "__main__":
    unittest.main()
