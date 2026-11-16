"""base.py - Base classes and utilities for prediction market data importers.

Provides data structures, CSV serialization, train/test splitting, and common
helpers for importing resolved markets into HAP-compatible datasets.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

DEFAULT_EPSILON = 1e-4


@dataclasses.dataclass(frozen=True)
class RawMarket:
    """Normalized representation of a single prediction market question."""
    id: str
    title: str
    outcome: int  # 0 or 1
    platform: str
    description: str = ""
    url: Optional[str] = None
    resolution_time: Optional[float] = None  # Unix epoch in seconds
    close_time: Optional[float] = None       # Unix epoch in seconds
    created_time: Optional[float] = None     # Unix epoch in seconds
    volume: Optional[float] = None
    predictions: Dict[str, float] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def sort_key(self) -> float:
        """Timestamp for chronological ordering (prefers close_time, then resolution_time, then created_time)."""
        if self.close_time is not None:
            return self.close_time
        if self.resolution_time is not None:
            return self.resolution_time
        if self.created_time is not None:
            return self.created_time
        return 0.0


@dataclasses.dataclass(frozen=True)
class AgentDefinition:
    """Specification of an agent or signal derived from market data."""
    agent_id: str
    model_family: str
    operator: str
    pipeline: str
    price: float
    extractor: Callable[[RawMarket], float]


@dataclasses.dataclass
class ImportedDataset:
    """An in-memory dataset of questions and predictions ready for HAP."""
    questions: Dict[str, Dict[str, Any]]  # qid -> {text, outcome, split}
    agents: Dict[str, Dict[str, Any]]     # aid -> {model_family, operator, pipeline, price}
    forecasts: Dict[Tuple[str, str], float]  # (qid, aid) -> probability

    def write_csv(self, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """Write questions.csv and predictions.csv to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        q_path = out / "questions.csv"
        p_path = out / "predictions.csv"

        with open(q_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "text", "outcome", "split"])
            for qid in sorted(self.questions.keys()):
                q = self.questions[qid]
                writer.writerow([qid, q["text"], q["outcome"], q["split"]])

        with open(p_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "question_id", "agent_id", "model_family", "operator",
                "pipeline", "price", "probability"
            ])
            for qid in sorted(self.questions.keys()):
                for aid in sorted(self.agents.keys()):
                    meta = self.agents[aid]
                    prob = self.forecasts[(qid, aid)]
                    writer.writerow([
                        qid, aid, meta["model_family"], meta["operator"],
                        meta["pipeline"], f"{meta['price']:.1f}", f"{prob:.4f}"
                    ])

        return q_path, p_path

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of the imported dataset."""
        train_q = [q for q in self.questions.values() if q["split"] == "train"]
        test_q = [q for q in self.questions.values() if q["split"] == "test"]
        train_ones = sum(q["outcome"] for q in train_q)
        test_ones = sum(q["outcome"] for q in test_q)

        return {
            "total_questions": len(self.questions),
            "train_questions": len(train_q),
            "test_questions": len(test_q),
            "train_positive_rate": train_ones / len(train_q) if train_q else 0.0,
            "test_positive_rate": test_ones / len(test_q) if test_q else 0.0,
            "agent_count": len(self.agents),
            "agents": list(self.agents.keys()),
        }


def _normalize_iso(val: str) -> str:
    """Rewrite ISO 8601 variants the APIs use into a form Python 3.10 accepts.

    Handles a trailing Z, a "+00" offset without minutes (Polymarket) and
    fractional seconds with other than 3 or 6 digits (Kalshi).
    """
    val = val.strip().replace("Z", "+00:00")
    val = re.sub(r"([+-]\d{2})$", r"\1:00", val)
    def pad(match: "re.Match[str]") -> str:
        return "." + (match.group(1) + "000000")[:6]
    return re.sub(r"\.(\d+)(?=[+-]\d{2}:\d{2}$|$)", pad, val)


def parse_timestamp(val: Any) -> Optional[float]:
    """Parse various timestamp representations into a Unix epoch float in seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if val > 1e11:
            return float(val) / 1000.0
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            num = float(val)
            if num > 1e11:
                return num / 1000.0
            return num
        except ValueError:
            pass
        try:
            dt = datetime.datetime.fromisoformat(_normalize_iso(val))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    return None


def sanitize_probability(p: Any, default: float = 0.5, epsilon: float = DEFAULT_EPSILON) -> float:
    """Clamp probability strictly into [epsilon, 1 - epsilon] and round to 4 decimals."""
    try:
        val = float(p)
    except (TypeError, ValueError):
        val = default
    if not math.isfinite(val):
        val = default
    clamped = max(epsilon, min(1.0 - epsilon, val))
    return round(clamped, 4)


def extremize_probability(p: float, d: float = 1.3) -> float:
    """Extremize probability using odds ratio formula: O' = O^d."""
    p_clamped = max(0.001, min(0.999, float(p)))
    odds = p_clamped / (1.0 - p_clamped)
    ext_odds = odds ** d
    return ext_odds / (1.0 + ext_odds)


def split_chronological(markets: Sequence[RawMarket], train_ratio: float = 0.5) -> Dict[str, str]:
    """Split questions chronologically. Older questions go to train, newer to test."""
    if not markets:
        return {}
    sorted_markets = sorted(markets, key=lambda m: (m.sort_key(), m.id))
    n = len(sorted_markets)
    if n == 1:
        return {sorted_markets[0].id: "train"}
    n_train = max(1, min(n - 1, int(round(n * train_ratio))))
    splits = {}
    for idx, m in enumerate(sorted_markets):
        splits[m.id] = "train" if idx < n_train else "test"
    return splits


def split_random(markets: Sequence[RawMarket], train_ratio: float = 0.5, seed: int = 42) -> Dict[str, str]:
    """Split questions randomly using a fixed seed."""
    if not markets:
        return {}
    rng = random.Random(seed)
    shuffled = sorted(markets, key=lambda m: m.id)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 1:
        return {shuffled[0].id: "train"}
    n_train = max(1, min(n - 1, int(round(n * train_ratio))))
    splits = {}
    for idx, m in enumerate(shuffled):
        splits[m.id] = "train" if idx < n_train else "test"
    return splits


def fetch_http_json(url: str, headers: Optional[Dict[str, str]] = None,
                    timeout: float = 20.0) -> Any:
    """Fetch JSON from a URL using standard library urllib."""
    req_headers = {
        "User-Agent": "HAP-PredictionMarketImporter/1.0 (+https://github.com/Wieshof/HAP)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset)
            return json.loads(body)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP request to {url} failed with status {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to {url}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Error fetching {url}: {e}") from e


WRAPPER_KEYS = ("markets", "questions", "results", "data")


def read_json_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read market records from a JSON file or a JSONL file.

    A JSON file may hold a list of records or an object that wraps the list
    under one of WRAPPER_KEYS (as the Kalshi and Metaculus APIs return it).
    Anything that is not valid JSON as a whole is read line by line as JSONL.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
            elif isinstance(obj, list):
                records.extend(o for o in obj if isinstance(o, dict))
        return records
    if isinstance(data, list):
        return [o for o in data if isinstance(o, dict)]
    if isinstance(data, dict):
        for key in WRAPPER_KEYS:
            if isinstance(data.get(key), list):
                return [o for o in data[key] if isinstance(o, dict)]
        return [data]
    return []


def usable_price(value: Optional[float]) -> Optional[float]:
    """Return a traded price only if it lies strictly between 0 and 1.

    Resolved Polymarket markets report 0 as the last trade price and settled
    Kalshi markets report empty books as 0 / 1; those are placeholders, not
    forecasts.
    """
    if value is None or not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        return None
    return value


def usable_quote(bid: Optional[float], ask: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (bid, ask) only if both form a real two-sided quote, else (None, None)."""
    bid, ask = usable_price(bid), usable_price(ask)
    if bid is None or ask is None or bid > ask:
        return None, None
    return bid, ask


class BaseImporter:
    """Base class for platform-specific prediction market importers."""
    platform_name: str = "base"
    # Skip resolved markets that report no real final price. Price-history
    # imports switch this off, because their forecast comes from earlier prices.
    require_final_price: bool = True

    def prices_at(self, market: RawMarket, times: Sequence[float],
                  api_key: Optional[str] = None) -> List[Optional[float]]:
        """Market probability at each Unix time (see history.py)."""
        raise NotImplementedError

    def get_default_agents(self) -> List[AgentDefinition]:
        """Return the default set of agents/signals extracted for this platform."""
        raise NotImplementedError

    def parse_market(self, raw: Dict[str, Any]) -> Optional[RawMarket]:
        """Parse a raw JSON dictionary into a RawMarket object.
        Returns None if market is not resolved binary or should be filtered out."""
        raise NotImplementedError

    def fetch_live_raw(self, limit: int = 100, api_key: Optional[str] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw market objects live from the platform's API."""
        raise NotImplementedError

    def load_dump_file(self, path: Union[str, Path]) -> List[RawMarket]:
        """Load and parse markets from a local JSON or JSONL file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dump file not found: {file_path}")

        raw_items = read_json_records(file_path)

        parsed: List[RawMarket] = []
        for item in raw_items:
            m = self.parse_market(item)
            if m is not None:
                parsed.append(m)

        return parsed

    def to_hap_dataset(
        self,
        markets: Sequence[RawMarket],
        split_strategy: str = "chronological",
        train_ratio: float = 0.5,
        seed: int = 42,
        agents: Optional[List[AgentDefinition]] = None,
        min_questions: int = 2,
    ) -> ImportedDataset:
        """Convert a sequence of parsed RawMarket objects to an ImportedDataset."""
        if len(markets) < min_questions:
            raise ValueError(
                f"Dataset has {len(markets)} valid binary questions; "
                f"minimum required is {min_questions}"
            )

        unique_markets: Dict[str, RawMarket] = {}
        for m in markets:
            if m.id not in unique_markets:
                unique_markets[m.id] = m
        market_list = list(unique_markets.values())

        if split_strategy == "random":
            splits = split_random(market_list, train_ratio=train_ratio, seed=seed)
        else:
            splits = split_chronological(market_list, train_ratio=train_ratio)

        active_agents = agents or self.get_default_agents()
        if not active_agents:
            raise ValueError("No agents defined for dataset conversion")

        questions: Dict[str, Dict[str, Any]] = {}
        agent_dict: Dict[str, Dict[str, Any]] = {}
        forecasts: Dict[Tuple[str, str], float] = {}

        for a in active_agents:
            agent_dict[a.agent_id] = {
                "model_family": a.model_family,
                "operator": a.operator,
                "pipeline": a.pipeline,
                "price": a.price,
            }

        for m in market_list:
            clean_text = " ".join(m.title.split()).strip()
            if not clean_text:
                clean_text = f"Question {m.id}"

            questions[m.id] = {
                "text": clean_text,
                "outcome": int(m.outcome),
                "split": splits.get(m.id, "test"),
            }

            for a in active_agents:
                p = a.extractor(m)
                forecasts[(m.id, a.agent_id)] = sanitize_probability(p)

        return ImportedDataset(questions=questions, agents=agent_dict, forecasts=forecasts)
