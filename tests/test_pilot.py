"""test_pilot.py - The pilot pipeline end to end, with markets and models mocked."""
from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hap_pilot as hp  # noqa: E402

DAY = 86400.0
T0 = 1_800_000_000.0  # registration time


def quiet(*_args, **_kwargs):
    pass


def agent(aid, family, provider="command", price=1.0, prompt="direct"):
    return {"agent_id": aid, "provider": provider, "model": f"m-{aid}", "family": family,
            "prompt": prompt, "price_in": price, "price_out": price, "command": "unused"}


AGENTS = [
    agent("sharp_a", "alpha", price=4.0),
    agent("sharp_b", "alpha", price=4.0, prompt="base_rate"),
    agent("alpha_twin", "alpha", price=4.0),
    agent("beta", "beta", price=1.0),
    agent("cheap", "gamma", price=0.1),
]


def config(**rules):
    base = {"name": "test", "agents": AGENTS, "cheap_baseline": "cheap",
            "platforms": ["polymarket", "kalshi"], "rules": {"n_questions": 40, **rules}}
    return base


def outcome(qid: str) -> int:
    return random.Random(qid).random() < 0.5


def fake_open(platform, n):
    def fetcher(rules, now, fetch=None):
        prefix = "PM" if platform == "polymarket" else "KALSHI"
        return [hp.OpenQuestion(f"{prefix}_{i}", platform, str(i), f"{platform} question {i}?",
                                "Resolves YES if it happens.", None, now + 30 * DAY,
                                0.7 if outcome(f"{prefix}_{i}") else 0.3, 1000.0 + i,
                                event=f"E{i // 2}" if platform == "kalshi" else None)
                for i in range(n)]
    return fetcher


def fake_provider(noise):
    def call(agent_def, system, prompt):
        title = prompt.split("Question: ")[1].split("\n")[0]
        platform, _, num = title.rstrip("?").split()
        qid = ("PM_" if platform == "polymarket" else "KALSHI_") + num
        rng = random.Random(f"{agent_def['agent_id']}-{qid}")
        truth = 0.8 if outcome(qid) else 0.2
        p = min(0.99, max(0.01, truth + rng.gauss(0, noise[agent_def["agent_id"]])))
        return hp.ModelReply(f"Some reasoning.\nProbability: {p:.2f}", 500, 1000)
    return call


NOISE = {"sharp_a": 0.15, "sharp_b": 0.15, "alpha_twin": 0.15, "beta": 0.25, "cheap": 0.35}


def fake_market_fetch(url, headers=None):
    if "gamma-api" in url:
        mid = url.rsplit("/", 1)[1]
        yes = outcome(f"PM_{mid}")
        return {"id": mid, "question": "Q?", "closed": True, "outcomes": '["Yes", "No"]',
                "outcomePrices": '["1", "0"]' if yes else '["0", "1"]',
                "closedTime": hp.iso(T0 + 30 * DAY)}
    ticker = url.rsplit("/", 1)[1]
    return {"market": {"ticker": ticker, "title": "Q?", "status": "finalized",
                       "result": "yes" if outcome(f"KALSHI_{ticker}") else "no",
                       "close_time": hp.iso(T0 + 30 * DAY)}}


class PilotHelpers(unittest.TestCase):
    def test_parse_probability(self):
        self.assertEqual(hp.parse_probability("blah\nProbability: 0.37"), 0.37)
        self.assertEqual(hp.parse_probability("**Probability:** .4"), 0.4)
        self.assertEqual(hp.parse_probability("Probability: 12%"), 0.12)
        self.assertEqual(hp.parse_probability("Probability: 0.2 ... Probability: 0.6"), 0.6)
        self.assertIsNone(hp.parse_probability("I think it is likely."))
        self.assertIsNone(hp.parse_probability("Probability: 7"))

    def test_config_rejects_placeholders_and_unknown_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            bad = config()
            bad["agents"] = AGENTS + [dict(agent("x", "y"), model="FILL_IN_MODEL")]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "placeholder"):
                hp.load_config(path)
            path.write_text(json.dumps(config(made_up=1)))
            with self.assertRaisesRegex(ValueError, "unknown rule"):
                hp.load_config(path)
            path.write_text(json.dumps(dict(config(), cheap_baseline="nobody")))
            with self.assertRaisesRegex(ValueError, "cheap_baseline"):
                hp.load_config(path)

    def test_example_config_only_fails_on_placeholders(self):
        example = Path(__file__).resolve().parent.parent / "pilot" / "config.example.json"
        with self.assertRaisesRegex(ValueError, "missing price_in, price_out|placeholder"):
            hp.load_config(example)

    def test_sampling_one_per_event_equal_quota_balanced_split(self):
        rules = dict(hp.DEFAULT_RULES, n_questions=10)
        cands = {"polymarket": fake_open("polymarket", 20)(rules, T0),
                 "kalshi": fake_open("kalshi", 20)(rules, T0)}
        chosen = hp.sample_questions(cands, rules)
        self.assertEqual(len(chosen), 10)
        kalshi = [q for q in chosen if q.platform == "kalshi"]
        self.assertEqual(len(kalshi), 5)
        self.assertEqual(len({q.event for q in kalshi}), 5)
        for platform in ("kalshi", "polymarket"):
            splits = [q.split for q in chosen if q.platform == platform]
            self.assertLessEqual(abs(splits.count("train") - splits.count("test")), 1)
        again = hp.sample_questions({p: fake_open(p, 20)(rules, T0) for p in cands}, rules)
        self.assertEqual([q.question_id for q in chosen], [q.question_id for q in again])


class PilotEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "run"
        self.saved = dict(hp.OPEN_FETCHERS)
        hp.OPEN_FETCHERS.update(polymarket=fake_open("polymarket", 30), kalshi=fake_open("kalshi", 60))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(config(), f)
        hp.register(hp.load_config(Path(f.name)), self.run, now=T0, log=quiet)

    def tearDown(self):
        hp.OPEN_FETCHERS.clear()
        hp.OPEN_FETCHERS.update(self.saved)
        self.tmp.cleanup()

    def collect(self, clock=lambda: T0 + 3600, **kw):
        return hp.forecast(self.run, providers={"command": fake_provider(NOISE)}, clock=clock, log=quiet, **kw)

    def test_full_pipeline_writes_report(self):
        counts = self.collect(workers=3)
        self.assertEqual(counts["ok"], 40 * len(AGENTS))
        self.assertEqual(self.collect()["ok"], 0)  # nothing left to do
        records = hp.read_jsonl(self.run / "forecasts.jsonl")
        self.assertTrue(all(r["cost_usd"] > 0 and r["requested_at"].endswith("Z") for r in records))
        hp.resolve(self.run, fetch=fake_market_fetch, clock=lambda: T0 + 31 * DAY, log=quiet)
        report = hp.score(self.run, log=quiet)
        self.assertIn("HAP diversity-aware", report)
        self.assertIn("Average of all models", report)
        self.assertIn("Market price at registration", report)
        self.assertIn("Cheap single model (cheap)", report)
        self.assertTrue((self.run / "report.md").exists())

    def test_registration_is_frozen(self):
        with self.assertRaisesRegex(RuntimeError, "never overwritten"):
            hp.register({"name": "x", "rules": hp.DEFAULT_RULES, "platforms": ["kalshi"],
                         "agents": AGENTS, "cheap_baseline": "cheap"}, self.run, now=T0, log=quiet)
        path = self.run / "registration.json"
        path.write_text(path.read_text().replace('"k": 3', '"k": 4'))
        with self.assertRaisesRegex(RuntimeError, "changed after registration"):
            self.collect()

    def test_forecast_window_is_enforced(self):
        with self.assertRaisesRegex(RuntimeError, "forecast window closed"):
            self.collect(clock=lambda: T0 + 49 * 3600)

    def test_failures_are_recorded_and_retried_three_times(self):
        def broken(agent_def, system, prompt):
            raise ConnectionError("down")
        for _ in range(4):
            hp.forecast(self.run, providers={"command": broken}, clock=lambda: T0 + 60,
                        only_agents=["cheap"], limit=1, log=quiet)
        records = hp.read_jsonl(self.run / "forecasts.jsonl")
        first = records[0]["question_id"]
        self.assertEqual(sum(r["question_id"] == first for r in records), 3)
        self.assertTrue(all(r["status"] == "error" for r in records))

    def test_late_and_after_close_forecasts_are_not_scored(self):
        self.collect()
        reg = hp.load_registration(self.run)
        records = hp.read_jsonl(self.run / "forecasts.jsonl")
        q = reg["questions"][0]["question_id"]
        resolutions = {q: {"outcome": 1, "close_time": T0 + 1800, "resolution_time": T0 + 1800}}
        kept, dropped = hp.usable_forecasts(reg, records, resolutions)
        self.assertFalse(any(k[0] == q for k in kept))
        self.assertEqual(dropped["after_close"], len(AGENTS))

    def test_dropping_an_agent_with_low_coverage(self):
        self.collect(only_agents=[a["agent_id"] for a in AGENTS if a["agent_id"] != "beta"])
        self.collect(only_agents=["beta"], limit=10)
        hp.resolve(self.run, fetch=fake_market_fetch, log=quiet)
        report = hp.score(self.run, log=quiet)
        self.assertIn("Agent beta dropped", report)

    def test_refusals_count_as_missing(self):
        def refuse(agent_def, system, prompt):
            return hp.ModelReply("", 10, 0, status="refused")
        hp.forecast(self.run, providers={"command": refuse}, clock=lambda: T0 + 60,
                    only_agents=["cheap"], limit=2, log=quiet)
        records = hp.read_jsonl(self.run / "forecasts.jsonl")
        self.assertEqual([r["status"] for r in records], ["refused", "refused"])
        self.assertIsNone(records[0]["probability"])


if __name__ == "__main__":
    unittest.main()
