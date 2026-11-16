"""kalshi.py - Importer for settled binary markets from Kalshi.

API documentation: https://docs.kalshi.com/welcome
"""
from __future__ import annotations

import urllib.parse
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (
    AgentDefinition,
    BaseImporter,
    RawMarket,
    extremize_probability,
    fetch_http_json,
    parse_timestamp,
    sanitize_probability,
    usable_price,
    usable_quote,
)
from .history import prices_from_history

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiImporter(BaseImporter):
    """Importer for Kalshi market dumps and Trade API v2."""
    platform_name = "kalshi"

    def get_default_agents(self) -> List[AgentDefinition]:
        """Default diverse forecasting agents derived from Kalshi exchange contracts.

        Spans 4 model families (kalshi, statistical, baseline, heuristic) with at least
        4 members in the primary family to satisfy HAP benchmark group selection rules.
        """
        return [
            # Primary kalshi family (4 members)
            AgentDefinition(
                agent_id="kalshi_last_price",
                model_family="kalshi",
                operator="cftc_exchange",
                pipeline="last_trade_execution",
                price=1.0,
                extractor=lambda m: m.predictions.get("last_price", 0.5),
            ),
            AgentDefinition(
                agent_id="kalshi_midpoint",
                model_family="kalshi",
                operator="cftc_exchange",
                pipeline="bid_ask_midpoint",
                price=0.8,
                extractor=lambda m: m.predictions.get("midpoint", 0.5),
            ),
            AgentDefinition(
                agent_id="kalshi_yes_bid",
                model_family="kalshi",
                operator="cftc_exchange",
                pipeline="best_yes_bid",
                price=0.5,
                extractor=lambda m: m.predictions.get("yes_bid", 0.5),
            ),
            AgentDefinition(
                agent_id="kalshi_yes_ask",
                model_family="kalshi",
                operator="cftc_exchange",
                pipeline="best_yes_ask",
                price=0.5,
                extractor=lambda m: m.predictions.get("yes_ask", 0.5),
            ),
            # Statistical family (2 members)
            AgentDefinition(
                agent_id="kalshi_shrunk",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="bayesian_shrinkage",
                price=0.2,
                extractor=lambda m: m.predictions.get("shrunk", 0.5),
            ),
            AgentDefinition(
                agent_id="kalshi_extremized",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="odds_extremization",
                price=0.3,
                extractor=lambda m: m.predictions.get("extremized", 0.5),
            ),
            # Baseline family (1 member)
            AgentDefinition(
                agent_id="kalshi_uniform_prior",
                model_family="baseline",
                operator="hap_baseline",
                pipeline="uniform_prior",
                price=0.1,
                extractor=lambda m: 0.5000,
            ),
            # Heuristic family (1 member)
            AgentDefinition(
                agent_id="kalshi_contrarian",
                model_family="heuristic",
                operator="hap_baseline",
                pipeline="contrarian_dampener",
                price=0.4,
                extractor=lambda m: m.predictions.get("contrarian", 0.5),
            ),
        ]

    def _parse_dollar_or_cents(self, dollar_val: Any, cents_val: Any, default: Optional[float] = None) -> Optional[float]:
        """Parse dollar string (e.g. '0.6500') or integer cents (e.g. 65) into float probability in [0, 1]."""
        if dollar_val is not None:
            try:
                num = float(dollar_val)
                if 0.0 <= num <= 1.0:
                    return num
                if 1.0 < num <= 100.0:
                    return num / 100.0
            except (ValueError, TypeError):
                pass
        if cents_val is not None:
            try:
                num = float(cents_val)
                if 0.0 <= num <= 1.0:
                    return num
                if 1.0 < num <= 100.0:
                    return num / 100.0
            except (ValueError, TypeError):
                pass
        return default

    def parse_market(self, raw: Dict[str, Any]) -> Optional[RawMarket]:
        """Parse a Kalshi market object. Returns None if not a settled binary market."""
        m_type = raw.get("market_type")
        if m_type is not None and str(m_type).lower() != "binary":
            return None

        status = str(raw.get("status") or "").lower()
        if status not in ("settled", "determined", "finalized", "closed"):
            return None

        result = str(raw.get("result") or "").lower()
        outcome: Optional[int] = None
        if result == "yes":
            outcome = 1
        elif result == "no":
            outcome = 0

        if outcome is None:
            sv_dollar = raw.get("settlement_value_dollars")
            sv_cents = raw.get("settlement_value")
            sv = self._parse_dollar_or_cents(sv_dollar, sv_cents)
            if sv is not None:
                if sv >= 0.99:
                    outcome = 1
                elif sv <= 0.01:
                    outcome = 0

        if outcome is None:
            return None

        ticker = str(raw.get("ticker") or "")
        if not ticker:
            return None

        title = str(raw.get("title") or "").strip()
        subtitle = str(raw.get("yes_sub_title") or "").strip()
        if subtitle and subtitle.lower() not in title.lower():
            full_title = f"{title} ({subtitle})"
        else:
            full_title = title
        if not full_title:
            return None

        last_price = self._parse_dollar_or_cents(raw.get("last_price_dollars"), raw.get("last_price"))
        yes_bid = self._parse_dollar_or_cents(raw.get("yes_bid_dollars"), raw.get("yes_bid"))
        yes_ask = self._parse_dollar_or_cents(raw.get("yes_ask_dollars"), raw.get("yes_ask"))

        last_price = usable_price(last_price)
        yes_bid, yes_ask = usable_quote(yes_bid, yes_ask)
        if last_price is None and yes_bid is None:
            # Settled markets report an empty book as 0 / 1; without a real
            # price there is no forecast to keep, unless it will come from
            # price history instead.
            if self.require_final_price:
                return None
            last_price = 0.5

        if yes_bid is not None and yes_ask is not None:
            midpoint = 0.5 * (yes_bid + yes_ask)
        else:
            midpoint = last_price

        if last_price is None:
            last_price = midpoint
        if yes_bid is None:
            yes_bid = max(0.01, midpoint - 0.01)
        if yes_ask is None:
            yes_ask = min(0.99, midpoint + 0.01)

        shrunk = 0.75 * last_price + 0.25 * 0.5
        ext = extremize_probability(last_price, d=1.25)
        contra = 0.5 + 0.6 * (last_price - 0.5)

        res_time = parse_timestamp(raw.get("settlement_time") or raw.get("settlement_ts"))
        close_time = parse_timestamp(raw.get("close_time"))
        created_time = parse_timestamp(raw.get("open_time"))
        vol = float(raw["volume"]) if raw.get("volume") is not None else None

        predictions = {
            "last_price": sanitize_probability(last_price),
            "midpoint": sanitize_probability(midpoint),
            "yes_bid": sanitize_probability(yes_bid),
            "yes_ask": sanitize_probability(yes_ask),
            "shrunk": sanitize_probability(shrunk),
            "extremized": sanitize_probability(ext),
            "contrarian": sanitize_probability(contra),
        }

        return RawMarket(
            id=f"KALSHI_{ticker}",
            title=full_title,
            outcome=outcome,
            platform=self.platform_name,
            description=str(raw.get("rules_primary") or ""),
            url=f"https://kalshi.com/markets/{ticker}",
            resolution_time=res_time,
            close_time=close_time,
            created_time=created_time,
            volume=vol,
            predictions=predictions,
            metadata={
                "ticker": ticker,
                "event_ticker": raw.get("event_ticker"),
                "series_ticker": raw.get("series_ticker"),
            },
        )

    def _series_ticker(self, market: RawMarket) -> str:
        series = market.metadata.get("series_ticker")
        if series:
            return str(series)
        event = market.metadata.get("event_ticker")
        if not event:
            raise RuntimeError(f"{market.id}: no event ticker to look up its series")
        cache = self.__dict__.setdefault("_series_cache", {})
        if event not in cache:
            data = fetch_http_json(f"{KALSHI_API_BASE}/events/{urllib.parse.quote(str(event))}")
            ev = data.get("event") if isinstance(data, dict) else None
            cache[event] = (ev or {}).get("series_ticker")
        if not cache[event]:
            raise RuntimeError(f"{market.id}: Kalshi returned no series for event {event}")
        return str(cache[event])

    @staticmethod
    def _candle_price(candle: Dict[str, Any]) -> Optional[float]:
        """Closing traded price of an hourly candle, else the midpoint of a real quote."""
        def dollars(block: Any) -> Optional[float]:
            if isinstance(block, dict) and block.get("close_dollars") not in (None, ""):
                try:
                    return float(block["close_dollars"])
                except (TypeError, ValueError):
                    return None
            return None

        traded = usable_price(dollars(candle.get("price")))
        if traded is not None:
            return traded
        bid, ask = usable_quote(dollars(candle.get("yes_bid")), dollars(candle.get("yes_ask")))
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        return None

    def prices_at(self, market: RawMarket, times: Sequence[float],
                  api_key: Optional[str] = None) -> List[Optional[float]]:
        """Market price at each Unix time, from hourly candlesticks."""
        ticker = market.metadata.get("ticker")
        if not ticker:
            raise RuntimeError(f"{market.id}: no ticker")
        series = self._series_ticker(market)
        start = min(times) - 86400.0
        if market.created_time is not None:
            start = max(start, market.created_time - 3600.0)
        query = urllib.parse.urlencode({
            "start_ts": str(int(start)),
            "end_ts": str(int(max(times))),
            "period_interval": "60",
        })
        url = (f"{KALSHI_API_BASE}/series/{urllib.parse.quote(series)}/markets/"
               f"{urllib.parse.quote(str(ticker))}/candlesticks?{query}")
        data = fetch_http_json(url)
        history = []
        for candle in (data.get("candlesticks") or []) if isinstance(data, dict) else []:
            price = self._candle_price(candle)
            ts = candle.get("end_period_ts")
            if price is not None and ts is not None:
                history.append((float(ts), price))
        return prices_from_history(history, times)

    def fetch_live_raw(self, limit: int = 100, api_key: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """Fetch settled markets live from Kalshi Trade API v2."""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        results: List[Dict[str, Any]] = []
        batch_size = min(limit, 100)
        cursor: Optional[str] = None

        while len(results) < limit:
            params = {
                "status": "settled",
                # Leave out multivariate combo markets (parlays), which often
                # trade for seconds and dominate the newest settled markets.
                "mve_filter": "exclude",
                "limit": str(batch_size),
            }
            if kwargs.get("series"):
                params["series_ticker"] = str(kwargs["series"])
            if cursor:
                params["cursor"] = cursor

            url = f"{KALSHI_API_BASE}/markets?{urllib.parse.urlencode(params)}"
            data = fetch_http_json(url, headers=headers)
            items = data.get("markets", []) if isinstance(data, dict) else []
            if not items:
                break
            results.extend(items)
            if len(items) < batch_size:
                break
            cursor = data.get("cursor") if isinstance(data, dict) else None
            if not cursor:
                break

        return results[:limit]
