"""history.py - Market prices at a fixed time before close.

The price a platform reports for a resolved question is its last one, and on
many markets the last trades happen after the answer is public. A fair
forecast is the price some hours before the market closed. This module asks
each platform for its price history, reads the price at that cutoff, and
builds the benchmark's signals from it.

Each importer provides ``prices_at(market, times, api_key)``, which returns the
market probability at each requested Unix time, or None where the market had
no price yet.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
from typing import Counter, Dict, List, Optional, Sequence, Tuple

from .base import AgentDefinition, RawMarket, extremize_probability, sanitize_probability

PricePoint = Tuple[float, float]  # (Unix seconds, probability of yes)

EARLIER_DAYS = 7.0


def price_at(history: Sequence[PricePoint], when: float) -> Optional[float]:
    """Return the last price recorded at or before ``when``, or None."""
    points = sorted(history)
    times = [t for t, _ in points]
    idx = bisect.bisect_right(times, when)
    if idx == 0:
        return None
    return points[idx - 1][1]


def first_price_before(history: Sequence[PricePoint], when: float) -> Optional[float]:
    """Return the earliest recorded price, provided it is at or before ``when``."""
    points = sorted(history)
    if not points or points[0][0] > when:
        return None
    return points[0][1]


def prices_from_history(history: Sequence[PricePoint], times: Sequence[float]) -> List[Optional[float]]:
    """Price at each requested time, falling back to the first recorded price.

    The fallback only applies to times before the first point when that first
    point still lies before the last requested time (the cutoff). It answers
    "what did the market say at its start" for markets younger than the
    earlier horizon, and never uses information from after the cutoff.
    """
    cutoff = max(times) if times else 0.0
    opening = first_price_before(history, cutoff)
    out: List[Optional[float]] = []
    for t in times:
        p = price_at(history, t)
        out.append(p if p is not None else (opening if t < cutoff else None))
    return out


def market_end_time(market: RawMarket) -> Optional[float]:
    """The earlier of close and resolution time: the latest moment that can be trusted."""
    candidates = [t for t in (market.close_time, market.resolution_time) if t is not None]
    return min(candidates) if candidates else None


def history_predictions(at_cutoff: float, earlier: float) -> Dict[str, float]:
    """Signals built from two pre-cutoff prices."""
    return {
        "at_cutoff": sanitize_probability(at_cutoff),
        "earlier": sanitize_probability(earlier),
        "average": sanitize_probability(0.5 * (at_cutoff + earlier)),
        "recent_blend": sanitize_probability(0.85 * at_cutoff + 0.15 * earlier),
        "shrunk": sanitize_probability(0.75 * at_cutoff + 0.25 * 0.5),
        "extremized": sanitize_probability(extremize_probability(at_cutoff, d=1.25)),
        "dampened": sanitize_probability(0.5 + 0.6 * (at_cutoff - 0.5)),
    }


def history_agents(hours_before_close: float) -> List[AgentDefinition]:
    """Signals for price-history imports, the same on every platform.

    All of them except the 50% baseline come from one market's prices, so
    they are baselines, not independent forecasters.
    """
    h = f"{hours_before_close:g}h"

    def key(name: str):
        return lambda m: m.predictions.get(name, 0.5)

    return [
        AgentDefinition("crowd_at_cutoff", "crowd", "market_crowd", f"price_{h}_before_close", 1.0, key("at_cutoff")),
        AgentDefinition("crowd_earlier", "crowd", "market_crowd", f"price_{EARLIER_DAYS:g}d_before_cutoff_or_first", 0.5, key("earlier")),
        AgentDefinition("crowd_average", "crowd", "market_crowd", "mean_of_cutoff_and_earlier", 0.8, key("average")),
        AgentDefinition("crowd_recent_blend", "crowd", "market_crowd", "0.85_cutoff_0.15_earlier", 0.9, key("recent_blend")),
        AgentDefinition("shrunk_to_half", "statistical", "hap_baseline", "0.75_cutoff_0.25_half", 0.2, key("shrunk")),
        AgentDefinition("extremized", "statistical", "hap_baseline", "odds_power_1.25", 0.3, key("extremized")),
        AgentDefinition("uniform_baseline", "baseline", "hap_baseline", "always_0.5", 0.1, lambda m: 0.5),
        AgentDefinition("dampened", "heuristic", "hap_baseline", "0.5_plus_0.6_times_distance", 0.4, key("dampened")),
    ]


def apply_price_history(
    importer,
    markets: Sequence[RawMarket],
    hours_before_close: float,
    api_key: Optional[str] = None,
    earlier_days: float = EARLIER_DAYS,
    progress=None,
) -> Tuple[List[RawMarket], Counter]:
    """Replace each market's signals with prices from before the cutoff.

    Returns the markets that had a price before the cutoff, and a count of why
    the others were left out.
    """
    if hours_before_close <= 0:
        raise ValueError("--hours-before-close must be positive")

    kept: List[RawMarket] = []
    skipped: Counter = collections.Counter()

    for idx, market in enumerate(markets, start=1):
        if progress and idx % 25 == 0:
            progress(f"  price history: {idx} of {len(markets)} markets")
        end = market_end_time(market)
        if end is None:
            skipped["no close time"] += 1
            continue
        cutoff = end - hours_before_close * 3600.0
        if market.created_time is not None and market.created_time > cutoff:
            skipped["opened after the cutoff"] += 1
            continue
        earlier_time = cutoff - earlier_days * 86400.0
        try:
            earlier, at_cutoff = importer.prices_at(market, [earlier_time, cutoff], api_key=api_key)
        except RuntimeError:
            skipped["price history request failed"] += 1
            continue
        if at_cutoff is None:
            skipped["no price before the cutoff"] += 1
            continue
        if earlier is None:
            earlier = at_cutoff
        metadata = dict(market.metadata)
        metadata.update({"cutoff_time": cutoff, "hours_before_close": hours_before_close})
        kept.append(dataclasses.replace(
            market,
            predictions=history_predictions(at_cutoff, earlier),
            metadata=metadata,
        ))

    return kept, skipped
