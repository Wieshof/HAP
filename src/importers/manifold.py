"""manifold.py - Importer for resolved binary markets from Manifold Markets.

API documentation: https://docs.manifold.markets/api
"""
from __future__ import annotations

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
)

MANIFOLD_API_BASE = "https://api.manifold.markets/v0"


class ManifoldImporter(BaseImporter):
    """Importer for Manifold Markets dumps and API."""
    platform_name = "manifold"

    def __init__(self, allow_mkt_resolution: bool = False):
        self.allow_mkt_resolution = allow_mkt_resolution

    def get_default_agents(self) -> List[AgentDefinition]:
        """Default diverse forecasting agents derived from Manifold market data.

        Spans 4 model families (manifold, statistical, baseline, heuristic) with at least
        4 members in the primary family to satisfy HAP benchmark group selection rules.
        """
        return [
            # Primary manifold crowd family (4 members)
            AgentDefinition(
                agent_id="manifold_crowd_final",
                model_family="manifold",
                operator="manifold_markets",
                pipeline="closing_probability",
                price=1.0,
                extractor=lambda m: m.predictions.get("closing", 0.5),
            ),
            AgentDefinition(
                agent_id="manifold_crowd_open",
                model_family="manifold",
                operator="manifold_markets",
                pipeline="opening_probability",
                price=0.5,
                extractor=lambda m: m.predictions.get("opening", 0.5),
            ),
            AgentDefinition(
                agent_id="manifold_midpoint",
                model_family="manifold",
                operator="manifold_markets",
                pipeline="midpoint_average",
                price=0.8,
                extractor=lambda m: m.predictions.get("midpoint", 0.5),
            ),
            AgentDefinition(
                agent_id="manifold_recency_decay",
                model_family="manifold",
                operator="manifold_markets",
                pipeline="recency_decay_blend",
                price=0.9,
                extractor=lambda m: m.predictions.get("recency_blend", 0.5),
            ),
            # Statistical family (2 members)
            AgentDefinition(
                agent_id="manifold_shrunk",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="bayesian_shrinkage",
                price=0.2,
                extractor=lambda m: m.predictions.get("shrunk", 0.5),
            ),
            AgentDefinition(
                agent_id="manifold_extremized",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="odds_extremization",
                price=0.3,
                extractor=lambda m: m.predictions.get("extremized", 0.5),
            ),
            # Baseline family (1 member)
            AgentDefinition(
                agent_id="manifold_uniform_prior",
                model_family="baseline",
                operator="hap_baseline",
                pipeline="uniform_prior",
                price=0.1,
                extractor=lambda m: 0.5000,
            ),
            # Heuristic family (1 member)
            AgentDefinition(
                agent_id="manifold_contrarian",
                model_family="heuristic",
                operator="hap_baseline",
                pipeline="contrarian_dampener",
                price=0.4,
                extractor=lambda m: m.predictions.get("contrarian", 0.5),
            ),
        ]

    def parse_market(self, raw: Dict[str, Any]) -> Optional[RawMarket]:
        """Parse a Manifold market object. Returns None if not a resolved binary market."""
        outcome_type = raw.get("outcomeType")
        if outcome_type != "BINARY":
            return None

        if not raw.get("isResolved", False):
            return None

        resolution = raw.get("resolution")
        outcome: Optional[int] = None
        if resolution == "YES":
            outcome = 1
        elif resolution == "NO":
            outcome = 0
        elif resolution == "MKT" and self.allow_mkt_resolution:
            res_prob = raw.get("resolutionProbability")
            if res_prob is not None:
                outcome = 1 if float(res_prob) >= 0.5 else 0
        elif resolution in (1, 1.0, "1"):
            outcome = 1
        elif resolution in (0, 0.0, "0"):
            outcome = 0

        if outcome is None:
            return None

        mid = str(raw.get("id") or raw.get("slug") or "")
        if not mid:
            return None

        title = str(raw.get("question") or raw.get("title") or "").strip()
        if not title:
            return None

        prob_closing = raw.get("probability")
        prob_closing = float(prob_closing) if prob_closing is not None else 0.5

        prob_open = raw.get("initialProbability")
        prob_open = float(prob_open) if prob_open is not None else 0.5

        prob_mid = 0.5 * (prob_closing + prob_open)
        prob_recency = 0.85 * prob_closing + 0.15 * prob_open
        prob_shrunk = 0.75 * prob_closing + 0.25 * 0.5
        prob_ext = extremize_probability(prob_closing, d=1.25)
        prob_contra = 0.5 + 0.6 * (prob_closing - 0.5)

        res_time = parse_timestamp(raw.get("resolutionTime"))
        close_time = parse_timestamp(raw.get("closeTime"))
        created_time = parse_timestamp(raw.get("createdTime"))
        vol = float(raw["volume"]) if raw.get("volume") is not None else None

        predictions = {
            "closing": sanitize_probability(prob_closing),
            "opening": sanitize_probability(prob_open),
            "midpoint": sanitize_probability(prob_mid),
            "recency_blend": sanitize_probability(prob_recency),
            "shrunk": sanitize_probability(prob_shrunk),
            "extremized": sanitize_probability(prob_ext),
            "contrarian": sanitize_probability(prob_contra),
        }

        return RawMarket(
            id=f"MNF_{mid}",
            title=title,
            outcome=outcome,
            platform=self.platform_name,
            description=str(raw.get("description") or ""),
            url=raw.get("url"),
            resolution_time=res_time,
            close_time=close_time,
            created_time=created_time,
            volume=vol,
            predictions=predictions,
            metadata={
                "unique_bettors": raw.get("uniqueBettorCount"),
                "raw_resolution": resolution,
            },
        )

    def fetch_live_raw(self, limit: int = 100, api_key: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """Fetch resolved binary markets live from Manifold API."""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Key {api_key}"

        results: List[Dict[str, Any]] = []
        batch_size = min(limit, 1000)
        before: Optional[str] = None

        while len(results) < limit:
            params = {
                "filter": "resolved",
                "contractType": "BINARY",
                "sort": "resolve-date",
                "limit": str(batch_size),
            }
            if before:
                params["before"] = before

            url = f"{MANIFOLD_API_BASE}/search-markets?{urllib.parse.urlencode(params)}"
            batch = fetch_http_json(url, headers=headers)
            if not isinstance(batch, list) or not batch:
                break

            results.extend(batch)
            if len(batch) < batch_size:
                break
            before = batch[-1].get("id")
            if not before:
                break

        return results[:limit]
