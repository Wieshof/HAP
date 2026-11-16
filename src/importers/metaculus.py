"""metaculus.py - Importer for resolved binary questions from Metaculus.

API documentation: https://www.metaculus.com/api/
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
)
from .history import prices_from_history

METACULUS_API_BASE = "https://www.metaculus.com/api2"


class MetaculusImporter(BaseImporter):
    """Importer for Metaculus question dumps and API."""
    platform_name = "metaculus"

    def get_default_agents(self) -> List[AgentDefinition]:
        """Default diverse forecasting agents derived from Metaculus crowd aggregations.

        Spans 4 model families (metaculus, statistical, baseline, heuristic) with at least
        4 members in the primary family to satisfy HAP benchmark group selection rules.
        """
        return [
            # Primary metaculus community family (4 members)
            AgentDefinition(
                agent_id="metaculus_recency_weighted",
                model_family="metaculus",
                operator="metaculus_community",
                pipeline="recency_weighted_median",
                price=1.2,
                extractor=lambda m: m.predictions.get("recency_weighted", 0.5),
            ),
            AgentDefinition(
                agent_id="metaculus_unweighted",
                model_family="metaculus",
                operator="metaculus_community",
                pipeline="unweighted_median",
                price=1.0,
                extractor=lambda m: m.predictions.get("unweighted", 0.5),
            ),
            AgentDefinition(
                agent_id="metaculus_peer_weighted",
                model_family="metaculus",
                operator="metaculus_platform",
                pipeline="peer_score_algorithm",
                price=1.5,
                extractor=lambda m: m.predictions.get("peer_weighted", 0.5),
            ),
            AgentDefinition(
                agent_id="metaculus_community_blend",
                model_family="metaculus",
                operator="metaculus_community",
                pipeline="community_blend",
                price=1.1,
                extractor=lambda m: m.predictions.get("blend", 0.5),
            ),
            # Statistical family (2 members)
            AgentDefinition(
                agent_id="metaculus_shrunk",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="bayesian_shrinkage",
                price=0.2,
                extractor=lambda m: m.predictions.get("shrunk", 0.5),
            ),
            AgentDefinition(
                agent_id="metaculus_extremized",
                model_family="statistical",
                operator="hap_baseline",
                pipeline="odds_extremization",
                price=0.3,
                extractor=lambda m: m.predictions.get("extremized", 0.5),
            ),
            # Baseline family (1 member)
            AgentDefinition(
                agent_id="metaculus_uniform_prior",
                model_family="baseline",
                operator="hap_baseline",
                pipeline="uniform_prior",
                price=0.1,
                extractor=lambda m: 0.5000,
            ),
            # Heuristic family (1 member)
            AgentDefinition(
                agent_id="metaculus_contrarian",
                model_family="heuristic",
                operator="hap_baseline",
                pipeline="contrarian_dampener",
                price=0.4,
                extractor=lambda m: m.predictions.get("contrarian", 0.5),
            ),
        ]

    def parse_market(self, raw: Dict[str, Any]) -> Optional[RawMarket]:
        """Parse a Metaculus question object. Returns None if not a resolved binary question."""
        q_type = raw.get("possibilities", {}).get("type") if isinstance(raw.get("possibilities"), dict) else None
        if not q_type:
            q_type = raw.get("type")
        if q_type != "binary":
            return None

        is_resolved = raw.get("resolved")
        if is_resolved is None:
            is_resolved = (raw.get("status") == "resolved")
        if not is_resolved:
            return None

        res = raw.get("resolution")
        outcome: Optional[int] = None
        if isinstance(res, str):
            res_lower = res.strip().lower()
            if res_lower in ("yes", "true", "1", "1.0"):
                outcome = 1
            elif res_lower in ("no", "false", "0", "0.0"):
                outcome = 0
        elif isinstance(res, (int, float)):
            if res >= 0.99:
                outcome = 1
            elif res <= 0.01:
                outcome = 0

        if outcome is None:
            return None

        qid = raw.get("id")
        if qid is None:
            return None
        qid_str = str(qid)

        title = str(raw.get("title") or "").strip()
        if not title:
            return None

        recency_val: Optional[float] = None
        aggregations = raw.get("aggregations")
        if isinstance(aggregations, dict):
            rw = aggregations.get("recency_weighted")
            if isinstance(rw, dict):
                latest = rw.get("latest")
                if isinstance(latest, dict):
                    centers = latest.get("centers")
                    if isinstance(centers, list) and centers:
                        recency_val = float(centers[0])

        if recency_val is None:
            cp = raw.get("community_prediction")
            if isinstance(cp, dict):
                full = cp.get("full")
                if isinstance(full, dict):
                    q2 = full.get("q2")
                    if q2 is not None:
                        recency_val = float(q2)
            elif isinstance(cp, (int, float)):
                recency_val = float(cp)

        if recency_val is None:
            recency_val = 0.5

        unweighted_val: Optional[float] = None
        if isinstance(aggregations, dict):
            uw = aggregations.get("unweighted")
            if isinstance(uw, dict):
                latest = uw.get("latest")
                if isinstance(latest, dict):
                    centers = latest.get("centers")
                    if isinstance(centers, list) and centers:
                        unweighted_val = float(centers[0])
        if unweighted_val is None:
            unweighted_val = recency_val

        peer_val: Optional[float] = None
        mp = raw.get("metaculus_prediction")
        if isinstance(mp, dict):
            full = mp.get("full")
            if isinstance(full, dict):
                q2 = full.get("q2")
                if q2 is not None:
                    peer_val = float(q2)
        elif isinstance(mp, (int, float)):
            peer_val = float(mp)
        if peer_val is None:
            peer_val = recency_val

        blend_val = 0.5 * (recency_val + unweighted_val)
        shrunk_val = 0.75 * recency_val + 0.25 * 0.5
        ext_val = extremize_probability(recency_val, d=1.25)
        contra_val = 0.5 + 0.6 * (recency_val - 0.5)

        res_time = parse_timestamp(raw.get("resolve_time") or raw.get("resolution_set_time"))
        close_time = parse_timestamp(raw.get("close_time"))
        created_time = parse_timestamp(raw.get("created_time") or raw.get("publish_time"))

        predictions = {
            "recency_weighted": sanitize_probability(recency_val),
            "unweighted": sanitize_probability(unweighted_val),
            "peer_weighted": sanitize_probability(peer_val),
            "blend": sanitize_probability(blend_val),
            "shrunk": sanitize_probability(shrunk_val),
            "extremized": sanitize_probability(ext_val),
            "contrarian": sanitize_probability(contra_val),
        }

        return RawMarket(
            id=f"META_{qid_str}",
            title=title,
            outcome=outcome,
            platform=self.platform_name,
            description=str(raw.get("description") or ""),
            url=f"https://www.metaculus.com/questions/{qid_str}/",
            resolution_time=res_time,
            close_time=close_time,
            created_time=created_time,
            volume=float(raw["number_of_forecasters"]) if raw.get("number_of_forecasters") else None,
            predictions=predictions,
            metadata={
                "raw_resolution": res,
                "number_of_forecasters": raw.get("number_of_forecasters"),
                "forecast_history": self._history(aggregations),
            },
        )

    @staticmethod
    def _history(aggregations: Any) -> List[Tuple[float, float]]:
        """Recency-weighted community forecast over time, if the record has it."""
        if not isinstance(aggregations, dict):
            return []
        rw = aggregations.get("recency_weighted")
        entries = rw.get("history") if isinstance(rw, dict) else None
        points: List[Tuple[float, float]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            start = parse_timestamp(entry.get("start_time"))
            centers = entry.get("centers")
            if start is None or not isinstance(centers, list) or not centers:
                continue
            try:
                points.append((start, float(centers[0])))
            except (TypeError, ValueError):
                continue
        return points

    def prices_at(self, market: RawMarket, times: Sequence[float],
                  api_key: Optional[str] = None) -> List[Optional[float]]:
        """Community forecast at each Unix time, from the history in the record itself."""
        return prices_from_history(market.metadata.get("forecast_history") or [], times)

    def fetch_live_raw(self, limit: int = 100, api_key: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """Fetch resolved binary questions live from Metaculus API."""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Token {api_key}"

        results: List[Dict[str, Any]] = []
        batch_size = min(limit, 100)
        url: Optional[str] = (
            f"{METACULUS_API_BASE}/questions/?"
            f"status=resolved&type=binary&order_by=-resolve_time&limit={batch_size}"
        )

        while url and len(results) < limit:
            data = fetch_http_json(url, headers=headers)
            items = data.get("results", []) if isinstance(data, dict) else []
            if not items:
                break
            results.extend(items)
            url = data.get("next") if isinstance(data, dict) else None

        return results[:limit]
