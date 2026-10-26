"""Tests for src/hap_contribution.py. Run with: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hap_contribution as hc  # noqa: E402
import hap_experiment as hx  # noqa: E402
from test_experiment import make_dataset  # noqa: E402


class IntervalTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(hc.percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertEqual(hc.percentile([1, 2, 3, 4, 5], 0.0), 1)
        self.assertAlmostEqual(hc.percentile([0, 10], 0.25), 2.5)

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(hc.bootstrap_indices(12), hc.bootstrap_indices(12))

    def test_rate_upper_bound(self):
        # 0 of 12: the exact one-sided 95% bound is 1 - 0.05 ** (1 / 12).
        self.assertAlmostEqual(hc.rate_upper_bound(0, 12), 1 - 0.05 ** (1 / 12), places=6)
        self.assertGreater(hc.rate_upper_bound(1, 12), hc.rate_upper_bound(0, 12))
        self.assertEqual(hc.rate_upper_bound(12, 12), 1.0)
        self.assertLess(hc.rate_upper_bound(0, 1000), 0.005)


class WithWithoutTests(unittest.TestCase):
    def test_contribution_sign(self):
        # c is useless and drags the average toward 50%; a and b are good.
        data = make_dataset({
            "q1": (1, {"a": 0.9, "b": 0.8, "c": 0.5}),
            "q2": (0, {"a": 0.2, "b": 0.1, "c": 0.5}),
            "q3": (1, {"a": 0.8, "b": 0.9, "c": 0.5}),
        })
        rows = {r.agent: r for r in hc.with_without(data, ["a", "b", "c"])}
        self.assertLess(rows["c"].contribution, 0)
        self.assertGreater(rows["a"].contribution, 0)
        full = hx.group_brier(data, ["a", "b", "c"], data.question_ids("test"))
        without_c = hx.group_brier(data, ["a", "b"], data.question_ids("test"))
        self.assertAlmostEqual(rows["c"].contribution, without_c - full)
        for r in rows.values():
            self.assertLessEqual(r.low, r.contribution + 1e-12)
            self.assertGreaterEqual(r.high, r.contribution - 1e-12)

    def test_needs_two_members(self):
        data = make_dataset({"q1": (1, {"a": 0.9})})
        with self.assertRaises(ValueError):
            hc.with_without(data, ["a"])


class CaseTests(unittest.TestCase):
    """Each fixture is built to show its pattern; these guard that it still does."""

    @classmethod
    def setUpClass(cls):
        cls.results = {c.name: hc.run_case(c) for c in hc.CASES}

    def row(self, case, agent):
        return next(t for t in self.results[case].table if t.agent == agent)

    def test_redundancy(self):
        ladder = [b for _, b in self.results["redundancy"].ladder]
        self.assertLess(max(ladder) - min(ladder), 0.005)
        self.assertGreater(self.row("redundancy", "gpt_2").most_similar_correlation, 0.95)

    def test_complementarity(self):
        local = self.row("complementarity", "local_1")
        others = [t for t in self.results["complementarity"].table if t.agent != "local_1"]
        self.assertTrue(all(local.solo_brier > t.solo_brier for t in others))
        self.assertEqual(local.verdict, "helps")

    def test_illusory_diversity(self):
        twin = self.row("illusory_diversity", "mistral_1")
        self.assertEqual(twin.provenance_overlap, 0.0)
        self.assertEqual(twin.most_similar, "gpt_1")
        self.assertGreater(twin.most_similar_correlation, 0.9)

    def test_counterproductive_diversity(self):
        bad = self.row("counterproductive_diversity", "contrarian_1")
        self.assertEqual(bad.verdict, "hurts")
        self.assertLess(bad.error_correlation, 0.2)
        for _, ids, _ in self.results["counterproductive_diversity"].selections:
            self.assertNotIn("contrarian_1", ids)


class BuyerTests(unittest.TestCase):
    def setUp(self):
        # cheap is good enough; pricey is slightly better. Both are trained and
        # tested on the same pattern of questions.
        rows = {}
        for i in range(40):
            y = i % 2
            rows[f"q{i:02d}"] = (y, {"cheap": 0.85 if y else 0.15, "pricey": 0.95 if y else 0.05})
        self.data = make_dataset(
            rows,
            agents={"cheap": hx.Agent("cheap", "small", "x", "p1", 0.2),
                    "pricey": hx.Agent("pricey", "large", "y", "p2", 2.0)},
            split={q: ("train" if int(q[1:]) < 20 else "test") for q in rows},
        )

    def test_picks_cheapest_option_that_meets_requirement(self):
        r = hc.buyer_test(self.data, hc.Requirement(max_brier=0.05, max_confident_wrong=0.2, budget=3),
                          max_size=2, k=2)
        self.assertEqual(r.chosen.agents, ("cheap",))
        self.assertEqual(r.most_accurate.agents, ("pricey",))
        self.assertTrue(r.chosen_test.meets(r.requirement))

    def test_budget_and_quality_can_exclude_everything(self):
        r = hc.buyer_test(self.data, hc.Requirement(max_brier=0.001, max_confident_wrong=0.2, budget=3),
                          max_size=2, k=2)
        self.assertIsNone(r.chosen)
        self.assertIn("No option meets", hc.format_buyer(r))

    def test_choice_ignores_test_split(self):
        flipped = hx.Dataset(
            {q: (v if v.split == "train" else hx.Question(q, v.text, 1 - v.outcome, v.split))
             for q, v in self.data.questions.items()},
            self.data.agents, self.data.forecasts,
        )
        req = hc.Requirement(max_brier=0.05, max_confident_wrong=0.2, budget=3)
        a = hc.buyer_test(self.data, req, max_size=2, k=2)
        b = hc.buyer_test(flipped, req, max_size=2, k=2)
        self.assertEqual(a.chosen.agents, b.chosen.agents)


if __name__ == "__main__":
    unittest.main()
