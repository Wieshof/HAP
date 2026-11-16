"""unified.py - Unified interface, auto-detection, and multi-source loader for prediction markets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .base import (
    AgentDefinition,
    BaseImporter,
    ImportedDataset,
    RawMarket,
    extremize_probability,
    read_json_records,
    sanitize_probability,
    split_chronological,
    split_random,
)
from .kalshi import KalshiImporter
from .manifold import ManifoldImporter
from .metaculus import MetaculusImporter
from .polymarket import PolymarketImporter

IMPORTERS: Dict[str, type[BaseImporter]] = {
    "manifold": ManifoldImporter,
    "metaculus": MetaculusImporter,
    "polymarket": PolymarketImporter,
    "kalshi": KalshiImporter,
}


def detect_platform_from_dict(raw: Dict[str, Any]) -> Optional[str]:
    """Inspect dictionary keys to identify which platform dump it originated from."""
    keys = set(raw.keys())

    # Manifold markers
    if "outcomeType" in keys or "uniqueBettorCount" in keys or "isResolved" in keys:
        return "manifold"

    # Metaculus markers
    if (
        "possibilities" in keys
        or "aggregations" in keys
        or "community_prediction" in keys
        or "resolve_time" in keys
        or "number_of_forecasters" in keys
    ):
        return "metaculus"

    # Polymarket markers
    if (
        "clobTokenIds" in keys
        or "conditionId" in keys
        or "umaResolutionStatus" in keys
        or "outcomePrices" in keys
    ):
        return "polymarket"

    # Kalshi markers
    if (
        "ticker" in keys
        or "event_ticker" in keys
        or "settlement_value_dollars" in keys
        or "yes_sub_title" in keys
    ):
        return "kalshi"

    return None


def detect_platform_from_file(path: Union[str, Path]) -> str:
    """Read initial lines/records of a file to detect its platform."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    for record in read_json_records(file_path):
        plat = detect_platform_from_dict(record)
        if plat:
            return plat

    raise ValueError(f"Could not automatically detect platform for {file_path}")


def get_importer(platform: str, **kwargs) -> BaseImporter:
    """Instantiate the importer corresponding to a platform name."""
    plat_norm = platform.strip().lower()
    cls = IMPORTERS.get(plat_norm)
    if not cls:
        raise ValueError(
            f"Unknown platform {platform!r}. Supported platforms: {', '.join(sorted(IMPORTERS.keys()))}"
        )
    return cls(**kwargs)


def load_dump_file(path: Union[str, Path], platform: Optional[str] = None) -> Tuple[str, List[RawMarket]]:
    """Load dump file, auto-detecting platform if not explicitly passed."""
    if not platform or platform.lower() == "auto":
        platform = detect_platform_from_file(path)
    importer = get_importer(platform)
    markets = importer.load_dump_file(path)
    return platform, markets


def get_canonical_cross_platform_agents() -> List[AgentDefinition]:
    """Canonical agent definitions that extract normalized signals across all platforms.
    Enables creating unified cross-platform benchmark datasets that satisfy HAP's complete matrix.
    Spans 4 families: crowd (4), statistical (2), baseline (1), heuristic (1).
    """
    def _extract_primary(m: RawMarket) -> float:
        return m.predictions.get(
            "closing",
            m.predictions.get(
                "recency_weighted",
                m.predictions.get(
                    "last_trade",
                    m.predictions.get("last_price", 0.5)
                )
            )
        )

    return [
        # Primary crowd family (4 members)
        AgentDefinition(
            agent_id="crowd_consensus",
            model_family="crowd",
            operator="platform_crowd",
            pipeline="closing_consensus",
            price=1.0,
            extractor=_extract_primary,
        ),
        AgentDefinition(
            agent_id="crowd_midpoint",
            model_family="crowd",
            operator="platform_crowd",
            pipeline="midpoint_estimate",
            price=0.8,
            extractor=lambda m: m.predictions.get("midpoint", _extract_primary(m)),
        ),
        AgentDefinition(
            agent_id="crowd_early_signal",
            model_family="crowd",
            operator="platform_crowd",
            pipeline="early_quote_or_bid",
            price=0.5,
            extractor=lambda m: m.predictions.get(
                "opening",
                m.predictions.get(
                    "unweighted",
                    m.predictions.get(
                        "best_bid",
                        m.predictions.get("yes_bid", 0.5)
                    )
                )
            ),
        ),
        AgentDefinition(
            agent_id="crowd_recency_blend",
            model_family="crowd",
            operator="platform_crowd",
            pipeline="recency_decay_blend",
            price=0.9,
            extractor=lambda m: m.predictions.get(
                "recency_blend",
                m.predictions.get("blend", _extract_primary(m))
            ),
        ),
        # Statistical family (2 members)
        AgentDefinition(
            agent_id="bayesian_shrinkage",
            model_family="statistical",
            operator="hap_baseline",
            pipeline="shrinkage_to_prior",
            price=0.2,
            extractor=lambda m: m.predictions.get("shrunk", 0.75 * _extract_primary(m) + 0.25 * 0.5),
        ),
        AgentDefinition(
            agent_id="odds_extremization",
            model_family="statistical",
            operator="hap_baseline",
            pipeline="odds_extremization",
            price=0.3,
            extractor=lambda m: m.predictions.get(
                "extremized",
                extremize_probability(_extract_primary(m), d=1.25)
            ),
        ),
        # Baseline family (1 member)
        AgentDefinition(
            agent_id="uniform_baseline",
            model_family="baseline",
            operator="hap_baseline",
            pipeline="uniform_prior",
            price=0.1,
            extractor=lambda m: 0.5000,
        ),
        # Heuristic family (1 member)
        AgentDefinition(
            agent_id="contrarian_dampener",
            model_family="heuristic",
            operator="hap_baseline",
            pipeline="contrarian_dampener",
            price=0.4,
            extractor=lambda m: m.predictions.get(
                "contrarian",
                0.5 + 0.6 * (_extract_primary(m) - 0.5)
            ),
        ),
    ]


def combine_multiplatform_markets(
    markets_by_platform: Dict[str, List[RawMarket]],
    split_strategy: str = "chronological",
    train_ratio: float = 0.5,
    seed: int = 42,
) -> ImportedDataset:
    """Combine parsed markets from multiple platforms into one unified HAP dataset."""
    all_markets: List[RawMarket] = []
    for plat, mkts in markets_by_platform.items():
        all_markets.extend(mkts)

    if not all_markets:
        raise ValueError("No markets provided for combined dataset")

    agents = get_canonical_cross_platform_agents()

    if split_strategy == "random":
        splits = split_random(all_markets, train_ratio=train_ratio, seed=seed)
    else:
        splits = split_chronological(all_markets, train_ratio=train_ratio)

    questions: Dict[str, Dict[str, Any]] = {}
    agent_dict: Dict[str, Dict[str, Any]] = {}
    forecasts: Dict[Tuple[str, str], float] = {}

    for a in agents:
        agent_dict[a.agent_id] = {
            "model_family": a.model_family,
            "operator": a.operator,
            "pipeline": a.pipeline,
            "price": a.price,
        }

    for m in all_markets:
        clean_text = " ".join(m.title.split()).strip() or f"Question {m.id}"
        questions[m.id] = {
            "text": clean_text,
            "outcome": int(m.outcome),
            "split": splits.get(m.id, "test"),
        }
        for a in agents:
            prob = a.extractor(m)
            forecasts[(m.id, a.agent_id)] = sanitize_probability(prob)

    return ImportedDataset(questions=questions, agents=agent_dict, forecasts=forecasts)
