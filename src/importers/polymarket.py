"""polymarket.py - Importer for resolved binary markets from Polymarket.

API documentation: https://data-api.polymarket.com/v2/docs#description/introduction
Gamma API: https://gamma-api.polymarket.com
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Optional

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

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


class PolymarketImporter(BaseImporter):
    """Importer for Polymarket dumps and Gamma/Data API."""
    platform_name = "polymarket"

    def get_default_agents(self) -> List[AgentDefinition]:
        """Default diverse forecasting agents derived from Polymarket order book and trades.

        Spans 4 model families (polymarket, statistical, baseline, heuristic) with at least
        4 members in the primary family to satisfy HAP benchmark group selection rules.
        """
        return [
            # Primary polymarket family (4 members)
            AgentDefinition(
                agent_id="polymarket_last_trade",
                model_family="polymarket",
                operator="clob_market",
                pipeline="last_trade_execution",
                price=1.0,
                extractor=lambda m: m.predictions.get("last_trade", 0.5),
            ),
            AgentDefinition(
                agent_id="polymarket_midpoint",
                model_family="polymarket",
                operator="clob_market",
                pipeline="bid_ask_midpoint",
                price=0.8,
                extractor=lambda m: m.predictions.get("midpoint", 0.5),
            ),
            AgentDefinition(
                agent_id="polymarket_best_bid",
                model_family="polymarket",
                operator="clob_market",
                pipeline="orderbook_bid",
                price=0.5,
                extractor=lambda m: m.predictions.get("best_bid", 0.5),
            ),
            AgentDefinition(
                agent_id="polymarket_best_ask",
                model_family="polymarket",
                operator="clob_market",
                pipeline="orderbook_ask",
                price=0.5,
                extractor=lambda m: m.predictions.get("best_ask", 0.5),
            ),
            # Statistical family (2 members)
            AgentDefinition(
                agent_id="polymarket_shrunk",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="bayesian_shrinkage",
                price=0.2,
                extractor=lambda m: m.predictions.get("shrunk", 0.5),
            ),
            AgentDefinition(
                agent_id="polymarket_extremized",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="odds_extremization",
                price=0.3,
                extractor=lambda m: m.predictions.get("extremized", 0.5),
            ),
            # Baseline family (1 member)
            AgentDefinition(
                agent_id="polymarket_uniform_prior",
                model_family="baseline",
                operator="hap_baseline",
                pipeline="uniform_prior",
                price=0.1,
                extractor=lambda m: 0.5000,
            ),
            # Heuristic family (1 member)
            AgentDefinition(
                agent_id="polymarket_contrarian",
                model_family="heuristic",
                operator="hap_baseline",
                pipeline="contrarian_dampener",
                price=0.4,
                extractor=lambda m: m.predictions.get("contrarian", 0.5),
            ),
        ]

    def _parse_json_field(self, val: Any) -> Any:
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val
        return val

    def parse_market(self, raw: Dict[str, Any]) -> Optional[RawMarket]:
        """Parse a Polymarket market object. Returns None if not a resolved binary market."""
        closed = raw.get("closed")
        if closed is not True and str(closed).lower() != "true":
            return None

        raw_outcomes = self._parse_json_field(raw.get("outcomes"))
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) != 2:
            return None

        outcomes_norm = [str(o).strip().lower() for o in raw_outcomes]

        outcome: Optional[int] = None

        prices = self._parse_json_field(raw.get("outcomePrices"))
        if isinstance(prices, list) and len(prices) >= 2:
            try:
                p0 = float(prices[0])
                p1 = float(prices[1])
                if outcomes_norm == ["yes", "no"]:
                    if p0 >= 0.99 and p1 <= 0.01:
                        outcome = 1
                    elif p0 <= 0.01 and p1 >= 0.99:
                        outcome = 0
                elif outcomes_norm == ["no", "yes"]:
                    if p0 >= 0.99 and p1 <= 0.01:
                        outcome = 0
                    elif p0 <= 0.01 and p1 >= 0.99:
                        outcome = 1
            except (ValueError, TypeError):
                pass

        if outcome is None:
            tokens = raw.get("tokens")
            if isinstance(tokens, list) and len(tokens) >= 2:
                t0_winner = tokens[0].get("winner")
                t1_winner = tokens[1].get("winner")
                if t0_winner is True and not t1_winner:
                    outcome = 1 if outcomes_norm[0] == "yes" else 0
                elif t1_winner is True and not t0_winner:
                    outcome = 1 if outcomes_norm[1] == "yes" else 0

        if outcome is None:
            winner = str(raw.get("winnerOutcome") or raw.get("resolution") or "").strip().lower()
            if winner in ("yes", "1", "1.0", "true"):
                outcome = 1
            elif winner in ("no", "0", "0.0", "false"):
                outcome = 0

        if outcome is None:
            return None

        mid = str(raw.get("id") or raw.get("conditionId") or raw.get("slug") or "")
        if not mid:
            return None

        title = str(raw.get("question") or raw.get("title") or "").strip()
        if not title:
            return None

        last_trade = raw.get("lastTradePrice")
        best_bid = raw.get("bestBid")
        best_ask = raw.get("bestAsk")

        lt_val: Optional[float] = None
        bid_val: Optional[float] = None
        ask_val: Optional[float] = None

        if last_trade is not None:
            try:
                lt_val = float(last_trade)
            except (ValueError, TypeError):
                pass

        if best_bid is not None:
            try:
                bid_val = float(best_bid)
            except (ValueError, TypeError):
                pass

        if best_ask is not None:
            try:
                ask_val = float(best_ask)
            except (ValueError, TypeError):
                pass

        lt_val = usable_price(lt_val)
        bid_val, ask_val = usable_quote(bid_val, ask_val)
        if lt_val is None and bid_val is None:
            # A resolved market without a real price (Gamma reports 0 / 0 / 1
            # once it settles) carries no forecast; skip it.
            return None

        if bid_val is not None and ask_val is not None:
            mid_val = 0.5 * (bid_val + ask_val)
        else:
            mid_val = lt_val

        if lt_val is None:
            lt_val = mid_val
        if bid_val is None:
            bid_val = max(0.01, mid_val - 0.01)
        if ask_val is None:
            ask_val = min(0.99, mid_val + 0.01)

        shrunk_val = 0.75 * lt_val + 0.25 * 0.5
        ext_val = extremize_probability(lt_val, d=1.25)
        contra_val = 0.5 + 0.6 * (lt_val - 0.5)

        close_time = parse_timestamp(raw.get("closedTime") or raw.get("endDate"))
        res_time = parse_timestamp(raw.get("umaEndDate") or raw.get("closedTime"))
        created_time = parse_timestamp(raw.get("startDate") or raw.get("createdAt"))
        vol = float(raw["volume"]) if raw.get("volume") is not None else None

        predictions = {
            "last_trade": sanitize_probability(lt_val),
            "midpoint": sanitize_probability(mid_val),
            "best_bid": sanitize_probability(bid_val),
            "best_ask": sanitize_probability(ask_val),
            "shrunk": sanitize_probability(shrunk_val),
            "extremized": sanitize_probability(ext_val),
            "contrarian": sanitize_probability(contra_val),
        }

        return RawMarket(
            id=f"POLY_{mid}",
            title=title,
            outcome=outcome,
            platform=self.platform_name,
            description=str(raw.get("description") or ""),
            url=f"https://polymarket.com/market/{raw.get('slug', mid)}",
            resolution_time=res_time,
            close_time=close_time,
            created_time=created_time,
            volume=vol,
            predictions=predictions,
            metadata={
                "condition_id": raw.get("conditionId"),
                "clob_token_ids": raw.get("clobTokenIds"),
            },
        )

    def fetch_live_raw(self, limit: int = 100, api_key: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """Fetch closed/resolved markets live from Polymarket Gamma API."""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        results: List[Dict[str, Any]] = []
        batch_size = min(limit, 100)
        offset = 0

        while len(results) < limit:
            params = {
                "closed": "true",
                "limit": str(batch_size),
                "offset": str(offset),
            }
            url = f"{GAMMA_API_BASE}/markets?{urllib.parse.urlencode(params)}"
            batch = fetch_http_json(url, headers=headers)
            if not isinstance(batch, list) or not batch:
                break
            results.extend(batch)
            if len(batch) < batch_size:
                break
            offset += len(batch)

        return results[:limit]
