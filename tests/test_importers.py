"""test_importers.py - Comprehensive tests for prediction market importers.

Covers Manifold, Metaculus, Polymarket, Kalshi, auto-detection, splitting,
and end-to-end integration with hap_experiment and hap_contribution.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hap_experiment as hx
from importers.base import (
    ImportedDataset,
    RawMarket,
    extremize_probability,
    parse_timestamp,
    read_json_records,
    sanitize_probability,
    split_chronological,
    split_random,
)
from importers.cli import main as cli_main
from importers.kalshi import KalshiImporter
from importers.manifold import ManifoldImporter
from importers.metaculus import MetaculusImporter
from importers.polymarket import PolymarketImporter
from importers.unified import (
    combine_multiplatform_markets,
    detect_platform_from_dict,
    detect_platform_from_file,
    get_importer,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_dumps"


class BaseUtilsTests(unittest.TestCase):
    def test_parse_timestamp(self):
        # Epoch seconds
        self.assertEqual(parse_timestamp(1700000000), 1700000000.0)
        # Epoch milliseconds
        self.assertEqual(parse_timestamp(1700000000000), 1700000000.0)
        # ISO string with Z
        self.assertEqual(parse_timestamp("2024-01-01T00:00:00Z"), 1704067200.0)
        # ISO string with offset
        self.assertEqual(parse_timestamp("2024-01-01T00:00:00+00:00"), 1704067200.0)
        # Empty / invalid
        self.assertIsNone(parse_timestamp(None))
        self.assertIsNone(parse_timestamp(""))
        self.assertIsNone(parse_timestamp("invalid-date-string"))

    def test_sanitize_probability(self):
        self.assertEqual(sanitize_probability(0.75), 0.75)
        self.assertEqual(sanitize_probability(0.0), 0.0001)
        self.assertEqual(sanitize_probability(1.0), 0.9999)
        self.assertEqual(sanitize_probability(-0.5), 0.0001)
        self.assertEqual(sanitize_probability(1.5), 0.9999)
        self.assertEqual(sanitize_probability("not-a-number", default=0.5), 0.5)

    def test_extremize_probability(self):
        # p > 0.5 should push higher
        self.assertGreater(extremize_probability(0.8), 0.8)
        # p < 0.5 should push lower
        self.assertLess(extremize_probability(0.2), 0.2)
        # p = 0.5 should stay near 0.5
        self.assertAlmostEqual(extremize_probability(0.5), 0.5, places=3)

    def test_splits(self):
        markets = [
            RawMarket(id=f"m_{i}", title=f"Q {i}", outcome=1, platform="test", close_time=1000 + i)
            for i in range(10)
        ]
        # Chronological
        chrono = split_chronological(markets, train_ratio=0.5)
        self.assertEqual(len(chrono), 10)
        train_count = sum(1 for s in chrono.values() if s == "train")
        test_count = sum(1 for s in chrono.values() if s == "test")
        self.assertEqual(train_count, 5)
        self.assertEqual(test_count, 5)
        # Older questions must be train
        self.assertEqual(chrono["m_0"], "train")
        self.assertEqual(chrono["m_9"], "test")

        # Random
        rand_split = split_random(markets, train_ratio=0.6, seed=123)
        self.assertEqual(sum(1 for s in rand_split.values() if s == "train"), 6)
        self.assertEqual(sum(1 for s in rand_split.values() if s == "test"), 4)


class ManifoldImporterTests(unittest.TestCase):
    def setUp(self):
        self.importer = ManifoldImporter()

    def test_parse_valid_binary(self):
        raw_yes = {
            "id": "q1",
            "question": "Will X happen?",
            "outcomeType": "BINARY",
            "isResolved": True,
            "resolution": "YES",
            "probability": 0.85,
            "initialProbability": 0.40,
            "resolutionTime": 1700000000000,
        }
        m_yes = self.importer.parse_market(raw_yes)
        self.assertIsNotNone(m_yes)
        self.assertEqual(m_yes.id, "MNF_q1")
        self.assertEqual(m_yes.outcome, 1)
        self.assertEqual(m_yes.predictions["closing"], 0.85)

        raw_no = {
            "id": "q2",
            "question": "Will Y happen?",
            "outcomeType": "BINARY",
            "isResolved": True,
            "resolution": "NO",
            "probability": 0.10,
        }
        m_no = self.importer.parse_market(raw_no)
        self.assertIsNotNone(m_no)
        self.assertEqual(m_no.outcome, 0)

    def test_filter_non_binary_and_unresolved(self):
        # Multiple choice
        self.assertIsNone(self.importer.parse_market({
            "id": "mc", "outcomeType": "MULTIPLE_CHOICE", "isResolved": True, "resolution": "A"
        }))
        # Unresolved
        self.assertIsNone(self.importer.parse_market({
            "id": "unres", "outcomeType": "BINARY", "isResolved": False
        }))
        # Canceled
        self.assertIsNone(self.importer.parse_market({
            "id": "canc", "outcomeType": "BINARY", "isResolved": True, "resolution": "CANCEL"
        }))

    def test_sample_dump(self):
        markets = self.importer.load_dump_file(SAMPLE_DIR / "manifold_sample.json")
        self.assertEqual(len(markets), 6)
        dataset = self.importer.to_hap_dataset(markets)
        summary = dataset.summary()
        self.assertEqual(summary["total_questions"], 6)
        self.assertEqual(summary["agent_count"], 8)


class MetaculusImporterTests(unittest.TestCase):
    def setUp(self):
        self.importer = MetaculusImporter()

    def test_parse_valid_binary(self):
        raw = {
            "id": 999,
            "title": "Will fusion achieve net energy?",
            "type": "binary",
            "resolved": True,
            "resolution": "yes",
            "resolve_time": "2024-05-01T12:00:00Z",
            "aggregations": {
                "recency_weighted": {"latest": {"centers": [0.72]}},
                "unweighted": {"latest": {"centers": [0.68]}},
            },
            "metaculus_prediction": {"full": {"q2": 0.74}},
        }
        m = self.importer.parse_market(raw)
        self.assertIsNotNone(m)
        self.assertEqual(m.id, "META_999")
        self.assertEqual(m.outcome, 1)
        self.assertEqual(m.predictions["recency_weighted"], 0.72)
        self.assertEqual(m.predictions["unweighted"], 0.68)
        self.assertEqual(m.predictions["peer_weighted"], 0.74)

    def test_filter_ambiguous(self):
        raw = {
            "id": 1000,
            "title": "Ambiguous event",
            "type": "binary",
            "resolved": True,
            "resolution": "ambiguous",
        }
        self.assertIsNone(self.importer.parse_market(raw))

    def test_sample_dump(self):
        markets = self.importer.load_dump_file(SAMPLE_DIR / "metaculus_sample.json")
        self.assertEqual(len(markets), 6)
        dataset = self.importer.to_hap_dataset(markets)
        summary = dataset.summary()
        self.assertEqual(summary["total_questions"], 6)
        self.assertEqual(summary["agent_count"], 8)


class PolymarketImporterTests(unittest.TestCase):
    def setUp(self):
        self.importer = PolymarketImporter()

    def test_parse_valid_binary(self):
        raw = {
            "id": "p1",
            "question": "Will BTC reach $100k?",
            "closed": True,
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"1\", \"0\"]",
            "lastTradePrice": 0.92,
            "bestBid": 0.91,
            "bestAsk": 0.93,
            "closedTime": "2024-11-01T00:00:00Z",
        }
        m = self.importer.parse_market(raw)
        self.assertIsNotNone(m)
        self.assertEqual(m.id, "POLY_p1")
        self.assertEqual(m.outcome, 1)
        self.assertEqual(m.predictions["last_trade"], 0.92)
        self.assertEqual(m.predictions["midpoint"], 0.92)

    def test_filter_open_market(self):
        raw = {
            "id": "p2",
            "question": "Open market",
            "closed": False,
            "outcomes": "[\"Yes\", \"No\"]",
        }
        self.assertIsNone(self.importer.parse_market(raw))

    def test_sample_dump(self):
        markets = self.importer.load_dump_file(SAMPLE_DIR / "polymarket_sample.json")
        self.assertEqual(len(markets), 6)
        dataset = self.importer.to_hap_dataset(markets)
        summary = dataset.summary()
        self.assertEqual(summary["total_questions"], 6)
        self.assertEqual(summary["agent_count"], 8)


class KalshiImporterTests(unittest.TestCase):
    def setUp(self):
        self.importer = KalshiImporter()

    def test_parse_valid_binary(self):
        raw = {
            "ticker": "RATE-CUT",
            "title": "Will rate cut occur?",
            "market_type": "binary",
            "status": "settled",
            "result": "no",
            "settlement_value_dollars": "0.0000",
            "last_price_dollars": "0.1500",
            "yes_bid_dollars": "0.1400",
            "yes_ask_dollars": "0.1600",
            "close_time": "2024-09-01T18:00:00Z",
        }
        m = self.importer.parse_market(raw)
        self.assertIsNotNone(m)
        self.assertEqual(m.id, "KALSHI_RATE-CUT")
        self.assertEqual(m.outcome, 0)
        self.assertEqual(m.predictions["last_price"], 0.15)
        self.assertEqual(m.predictions["midpoint"], 0.15)

    def test_cents_parsing(self):
        raw = {
            "ticker": "CENTS-TEST",
            "title": "Cents test",
            "market_type": "binary",
            "status": "settled",
            "settlement_value": 100,
            "last_price": 75,
            "yes_bid": 74,
            "yes_ask": 76,
        }
        m = self.importer.parse_market(raw)
        self.assertIsNotNone(m)
        self.assertEqual(m.outcome, 1)
        self.assertEqual(m.predictions["last_price"], 0.75)

    def test_sample_dump(self):
        markets = self.importer.load_dump_file(SAMPLE_DIR / "kalshi_sample.json")
        self.assertEqual(len(markets), 6)
        dataset = self.importer.to_hap_dataset(markets)
        summary = dataset.summary()
        self.assertEqual(summary["total_questions"], 6)
        self.assertEqual(summary["agent_count"], 8)


class RealApiShapeTests(unittest.TestCase):
    """Shapes the live APIs actually return, as seen in November 2026."""

    def test_object_wrapped_dump_is_unwrapped(self):
        # The Kalshi API answers {"cursor": ..., "markets": [...]} on one line.
        record = {"ticker": "K-1", "title": "T", "status": "finalized",
                  "result": "yes", "market_type": "binary",
                  "last_price_dollars": "0.9000",
                  "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kalshi.json"
            path.write_text(json.dumps({"cursor": "abc", "markets": [record, record]}))
            self.assertEqual(len(read_json_records(path)), 2)
            self.assertEqual(detect_platform_from_file(path), "kalshi")
            markets = KalshiImporter().load_dump_file(path)
        self.assertEqual(len(markets), 2)
        # The empty 0 / 1 book is a placeholder: the quote is rebuilt around
        # the last trade instead of being read as bid 0 / ask 1.
        preds = markets[0].predictions
        self.assertEqual(preds["midpoint"], 0.9)
        self.assertEqual((preds["yes_bid"], preds["yes_ask"]), (0.89, 0.91))

    def test_kalshi_without_any_price_is_skipped(self):
        raw = {"ticker": "K-2", "title": "T", "status": "finalized", "result": "no",
               "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}
        self.assertIsNone(KalshiImporter().parse_market(raw))

    def test_polymarket_placeholder_prices_are_skipped(self):
        # Gamma reports lastTradePrice 0, bestBid 0, bestAsk 1 after resolution.
        raw = {"id": "1", "question": "Q?", "closed": True,
               "outcomes": '["Yes", "No"]', "outcomePrices": '["0", "1"]',
               "lastTradePrice": 0, "bestBid": 0, "bestAsk": 1}
        self.assertIsNone(PolymarketImporter().parse_market(raw))

    def test_metaculus_fetch_needs_token(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(["metaculus", "--fetch", "--out", "unused"])
        self.assertEqual(code, 1)
        self.assertIn("--api-key", err.getvalue())

    def test_missing_dump_is_a_clean_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(["manifold", "--dump", "/no/such/file.json", "--out", "unused"])
        self.assertEqual(code, 1)
        self.assertIn("not found", err.getvalue())


class UnifiedAndDetectionTests(unittest.TestCase):
    def test_detect_platform_from_file(self):
        self.assertEqual(detect_platform_from_file(SAMPLE_DIR / "manifold_sample.json"), "manifold")
        self.assertEqual(detect_platform_from_file(SAMPLE_DIR / "metaculus_sample.json"), "metaculus")
        self.assertEqual(detect_platform_from_file(SAMPLE_DIR / "polymarket_sample.json"), "polymarket")
        self.assertEqual(detect_platform_from_file(SAMPLE_DIR / "kalshi_sample.json"), "kalshi")

    def test_combine_multiplatform_markets(self):
        m_mnf = ManifoldImporter().load_dump_file(SAMPLE_DIR / "manifold_sample.json")
        m_meta = MetaculusImporter().load_dump_file(SAMPLE_DIR / "metaculus_sample.json")
        combined = combine_multiplatform_markets({"manifold": m_mnf, "metaculus": m_meta})
        summary = combined.summary()
        self.assertEqual(summary["total_questions"], 12)
        self.assertEqual(summary["agent_count"], 8)


class EndToEndPipelineTests(unittest.TestCase):
    def test_cli_and_hap_pipeline_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mnf_out = out_dir / "manifold_out"

            # 1. Run CLI for Manifold
            code = cli_main([
                "manifold",
                "--dump", str(SAMPLE_DIR / "manifold_sample.json"),
                "--out", str(mnf_out),
                "--split", "chronological",
            ])
            self.assertEqual(code, 0)
            q_file = mnf_out / "questions.csv"
            p_file = mnf_out / "predictions.csv"
            self.assertTrue(q_file.exists())
            self.assertTrue(p_file.exists())

            # 2. Verify dataset loads cleanly in hap_experiment.load_dataset
            dataset = hx.load_dataset(q_file, p_file)
            self.assertEqual(len(dataset.questions), 6)
            self.assertEqual(len(dataset.agents), 8)

            # 3. Run multi-platform combiner
            multi_out = out_dir / "multi_out"
            code_multi = cli_main([
                "multi",
                "--manifold", str(SAMPLE_DIR / "manifold_sample.json"),
                "--metaculus", str(SAMPLE_DIR / "metaculus_sample.json"),
                "--polymarket", str(SAMPLE_DIR / "polymarket_sample.json"),
                "--kalshi", str(SAMPLE_DIR / "kalshi_sample.json"),
                "--out", str(multi_out),
            ])
            self.assertEqual(code_multi, 0)
            multi_q = multi_out / "questions.csv"
            multi_p = multi_out / "predictions.csv"
            self.assertTrue(multi_q.exists())
            self.assertTrue(multi_p.exists())

            # 4. Run hap_experiment benchmark on multi dataset
            exp_out = out_dir / "benchmark"
            hx.main([
                "--questions", str(multi_q),
                "--predictions", str(multi_p),
                "--out", str(exp_out),
            ])
            self.assertTrue((exp_out / "report.md").exists())
            self.assertTrue((exp_out / "results.csv").exists())
            self.assertTrue((exp_out / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
