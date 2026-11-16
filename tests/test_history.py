"""test_history.py - Prices at a fixed time before close (no network: API calls are mocked)."""
from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from importers.base import RawMarket, parse_timestamp
from importers.cli import main as cli_main
from importers.history import (
    apply_price_history,
    history_agents,
    market_end_time,
    price_at,
    prices_from_history,
)
from importers.kalshi import KalshiImporter
from importers.manifold import ManifoldImporter
from importers.metaculus import MetaculusImporter
from importers.polymarket import PolymarketImporter

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_dumps"
HOUR = 3600.0
DAY = 86400.0


def market(mid="M1", created=0.0, close=10 * DAY, resolved=None, **metadata):
    return RawMarket(id=mid, title="Q?", outcome=1, platform="test",
                     created_time=created, close_time=close, resolution_time=resolved,
                     predictions={"closing": 0.99}, metadata=metadata)


class PriceLookupTests(unittest.TestCase):
    def test_price_at_takes_last_point_not_after(self):
        history = [(30.0, 0.7), (10.0, 0.4), (20.0, 0.5)]
        self.assertIsNone(price_at(history, 5.0))
        self.assertEqual(price_at(history, 10.0), 0.4)
        self.assertEqual(price_at(history, 25.0), 0.5)
        self.assertEqual(price_at(history, 99.0), 0.7)

    def test_earlier_time_falls_back_to_first_price_only_before_cutoff(self):
        history = [(10.0, 0.4), (20.0, 0.6)]
        self.assertEqual(prices_from_history(history, [5.0, 25.0]), [0.4, 0.6])
        # The first point is after the cutoff: nothing may be used.
        self.assertEqual(prices_from_history([(30.0, 0.9)], [5.0, 25.0]), [None, None])

    def test_end_time_is_earlier_of_close_and_resolution(self):
        self.assertEqual(market_end_time(market(close=100.0, resolved=50.0)), 50.0)
        self.assertEqual(market_end_time(market(close=100.0, resolved=None)), 100.0)
        self.assertIsNone(market_end_time(market(close=None)))

    def test_timestamps_from_the_live_apis(self):
        self.assertEqual(parse_timestamp("2026-11-16 20:32:21.5+00"), 1794861141.5)
        self.assertEqual(parse_timestamp("2026-11-16T20:16:04.62743Z"), 1794860164.62743)
        self.assertEqual(parse_timestamp("2024-06-06T12:00:00"), 1717675200.0)
        self.assertEqual(parse_timestamp(1794844091521), 1794844091.521)


class FakeImporter:
    platform_name = "fake"

    def __init__(self, prices):
        self.prices = prices
        self.asked = []

    def prices_at(self, m, times, api_key=None):
        self.asked.append((m.id, list(times)))
        result = self.prices[m.id]
        if isinstance(result, Exception):
            raise result
        return result


class ApplyHistoryTests(unittest.TestCase):
    def test_replaces_final_price_and_reports_left_out(self):
        markets = [
            market("ok"),
            market("young", created=9.9 * DAY),
            market("noprice"),
            market("broken"),
            market("noclose", close=None),
        ]
        fake = FakeImporter({"ok": [0.3, 0.6], "noprice": [None, None],
                             "broken": RuntimeError("HTTP 500")})
        kept, skipped = apply_price_history(fake, markets, 24)
        self.assertEqual([m.id for m in kept], ["ok"])
        self.assertEqual(skipped, {"opened after the cutoff": 1, "no price before the cutoff": 1,
                                   "price history request failed": 1, "no close time": 1})
        # Asked for the price 7 days before the cutoff and at the cutoff.
        self.assertEqual(fake.asked[0], ("ok", [10 * DAY - 24 * HOUR - 7 * DAY, 10 * DAY - 24 * HOUR]))
        preds = kept[0].predictions
        self.assertNotIn("closing", preds)
        self.assertEqual(preds["at_cutoff"], 0.6)
        self.assertEqual(preds["earlier"], 0.3)
        self.assertEqual(preds["average"], 0.45)
        self.assertEqual(kept[0].metadata["cutoff_time"], 10 * DAY - 24 * HOUR)

    def test_missing_earlier_price_uses_cutoff_price(self):
        kept, _ = apply_price_history(FakeImporter({"ok": [None, 0.6]}), [market("ok")], 24)
        self.assertEqual(kept[0].predictions["earlier"], 0.6)

    def test_rejects_non_positive_hours(self):
        with self.assertRaises(ValueError):
            apply_price_history(FakeImporter({}), [], 0)

    def test_agents_read_history_keys(self):
        kept, _ = apply_price_history(FakeImporter({"ok": [0.3, 0.6]}), [market("ok")], 24)
        values = {a.agent_id: a.extractor(kept[0]) for a in history_agents(24)}
        self.assertEqual(values["crowd_at_cutoff"], 0.6)
        self.assertEqual(values["crowd_earlier"], 0.3)
        self.assertEqual(values["uniform_baseline"], 0.5)
        self.assertEqual(len({a.model_family for a in history_agents(24)}), 4)


class PlatformHistoryTests(unittest.TestCase):
    def test_manifold_uses_last_bet_before_each_time(self):
        m = market("MNF_abc", contract_id="abc", initial_probability=0.5)
        calls = []

        def fake_fetch(url, headers=None):
            calls.append(url)
            if "beforeTime=100000" in url:
                return [{"probAfter": 0.72}]
            return []

        with patch("importers.manifold.fetch_http_json", side_effect=fake_fetch):
            prices = ManifoldImporter().prices_at(m, [50.0, 100.0])
        # No bet before t=50: the initial probability applies.
        self.assertEqual(prices, [0.5, 0.72])
        self.assertTrue(all("contractId=abc" in u and "order=desc" in u for u in calls))

    def test_manifold_initial_price_from_first_bet_when_missing(self):
        m = market("MNF_abc", contract_id="abc", initial_probability=None)

        def fake_fetch(url, headers=None):
            if "order=asc" in url:
                return [{"probBefore": 0.33, "probAfter": 0.4}]
            return []

        with patch("importers.manifold.fetch_http_json", side_effect=fake_fetch):
            self.assertEqual(ManifoldImporter().prices_at(m, [50.0, 100.0]), [0.33, 0.33])

    def test_polymarket_reads_yes_token_history(self):
        raw = {"id": "9", "question": "Q?", "closed": True,
               "outcomes": '["No", "Yes"]', "outcomePrices": '["0", "1"]',
               "clobTokenIds": '["NO_TOKEN", "YES_TOKEN"]',
               "lastTradePrice": 0, "bestBid": 0, "bestAsk": 1,
               "closedTime": "2026-11-10T00:00:00Z"}
        imp = PolymarketImporter()
        self.assertIsNone(imp.parse_market(raw))  # final-price mode: placeholder prices
        imp.require_final_price = False
        m = imp.parse_market(raw)
        self.assertEqual(m.outcome, 1)
        self.assertEqual(m.metadata["yes_token_id"], "YES_TOKEN")
        history = {"history": [{"t": 100, "p": 0.2}, {"t": 200, "p": 0.55}, {"t": 300, "p": 0.99}]}
        with patch("importers.polymarket.fetch_http_json", return_value=history) as fetch:
            self.assertEqual(imp.prices_at(m, [150.0, 250.0]), [0.2, 0.55])
        self.assertIn("market=YES_TOKEN", fetch.call_args[0][0])

    def test_kalshi_candles_use_trades_then_real_quotes(self):
        m = market("KALSHI_T", ticker="T", event_ticker="EV", series_ticker=None)
        candles = {"candlesticks": [
            {"end_period_ts": 100, "price": {}, "yes_bid": {"close_dollars": "0.0000"},
             "yes_ask": {"close_dollars": "1.0000"}},                       # empty book: ignored
            {"end_period_ts": 200, "price": {}, "yes_bid": {"close_dollars": "0.4000"},
             "yes_ask": {"close_dollars": "0.5000"}},                       # quote midpoint
            {"end_period_ts": 300, "price": {"close_dollars": "0.6100"}},   # traded price
            {"end_period_ts": 400, "price": {"close_dollars": "0.9900"}},   # after the cutoff
        ]}

        def fake_fetch(url, headers=None):
            if "/events/EV" in url:
                return {"event": {"series_ticker": "SER"}}
            self.assertIn("/series/SER/markets/T/candlesticks", url)
            return candles

        with patch("importers.kalshi.fetch_http_json", side_effect=fake_fetch):
            # Before the first real price, the earlier time gets the opening price.
            self.assertEqual(KalshiImporter().prices_at(m, [150.0, 350.0]), [0.45, 0.61])
            self.assertEqual(KalshiImporter().prices_at(m, [150.0, 250.0]), [0.45, 0.45])
            # A cutoff before any real price has none.
            self.assertEqual(KalshiImporter().prices_at(m, [50.0, 150.0]), [None, None])

    def test_metaculus_history_comes_from_the_record(self):
        raw = {"id": 7, "title": "Q?", "type": "binary", "resolved": True, "resolution": "no",
               "close_time": "2026-01-10T00:00:00Z",
               "aggregations": {"recency_weighted": {
                   "latest": {"centers": [0.05]},
                   "history": [{"start_time": 1000, "centers": [0.4]},
                               {"start_time": 2000, "centers": [0.2]}]}}}
        imp = MetaculusImporter()
        m = imp.parse_market(raw)
        self.assertEqual(imp.prices_at(m, [1500.0, 2500.0]), [0.4, 0.2])


class CliHistoryTests(unittest.TestCase):
    def test_cli_writes_history_agents(self):
        history = {"history": [{"t": 1.0, "p": 0.3}, {"t": 1.6e9, "p": 0.8}]}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("importers.polymarket.fetch_http_json", return_value=history), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli_main(["polymarket", "--dump", str(SAMPLE_DIR / "polymarket_sample.json"),
                             "--hours-before-close", "24", "--out", tmp])
            self.assertEqual(code, 0, out.getvalue())
            with open(Path(tmp) / "predictions.csv", newline="") as f:
                rows = list(csv.DictReader(f))
        agents = {r["agent_id"] for r in rows}
        self.assertEqual(agents, {a.agent_id for a in history_agents(24)})
        cutoff = [r for r in rows if r["agent_id"] == "crowd_at_cutoff"]
        self.assertTrue(all(r["probability"] == "0.8000" for r in cutoff))
        self.assertIn("price_24h_before_close", {r["pipeline"] for r in rows})


if __name__ == "__main__":
    unittest.main()
