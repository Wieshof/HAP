"""hap_pilot.py - a pilot with real model forecasts on open questions.

The benchmark and the importers can only score signals that already exist.
This pilot records new, independent forecasts from several AI models on
questions that are still open, then scores them once the answers are known.

Four steps, run in order:

  register  Snapshot open binary questions from Manifold, Polymarket and
            Kalshi, apply fixed filters, sample them, split them into
            selection (train) and evaluation (test) questions, and freeze the
            plan: the agents, the selection rules and the scoring rules. The
            registration file is hashed so later changes can be detected.
  forecast  Ask every agent for a probability on every question. Each answer
            is appended to forecasts.jsonl with its UTC time, the full model
            reply, the tokens used and the cost. Models get no tools and never
            see the market price.
  resolve   Look up each question's outcome. Questions that resolve N/A,
            cancel or stay open are left out.
  score     Select groups on the train questions with the rules frozen at
            registration, and compare them on the test questions with a
            cheap single model, the plain average of all models, and the
            market price at registration.

The question the pilot asks: does HAP select something better, or something
cheaper that is still good enough? A negative answer is a result too. With
50-100 questions the intervals are wide: this is a pilot, not proof.

Usage:
  hap-pilot register --config pilot/config.json --run pilot/runs/2026-11
  hap-pilot forecast --run pilot/runs/2026-11 [--dry-run] [--workers 4]
  hap-pilot resolve  --run pilot/runs/2026-11
  hap-pilot score    --run pilot/runs/2026-11
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hap_contribution as hc  # noqa: E402
import hap_experiment as hx  # noqa: E402
from importers.base import fetch_http_json, parse_timestamp, usable_quote  # noqa: E402
from importers.kalshi import KALSHI_API_BASE, KalshiImporter  # noqa: E402
from importers.manifold import MANIFOLD_API_BASE, ManifoldImporter  # noqa: E402
from importers.polymarket import GAMMA_API_BASE, PolymarketImporter  # noqa: E402

DAY = 86400.0
PLATFORMS = ("manifold", "polymarket", "kalshi")
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# The plan. Everything here is frozen into registration.json; a config file
# may override it, but only before registering.
# ---------------------------------------------------------------------------

DEFAULT_RULES: Dict[str, Any] = {
    # Question selection
    "n_questions": 60,
    "min_days_to_close": 14,
    "max_days_to_close": 75,
    "min_price": 0.05,               # leave out questions the market already treats as settled
    "max_price": 0.95,
    "max_spread": 0.10,              # Polymarket and Kalshi: bid-ask spread at most this
    "manifold_min_bettors": 20,
    "polymarket_min_volume": 10000,  # USD
    "kalshi_min_volume": 1000,       # contracts
    "one_market_per_event": True,    # avoid many near-duplicate sub-markets
    "seed": 20261120,
    # Forecast collection
    "max_forecast_lag_hours": 48,    # forecasts later than this after registration are not scored
    "min_agent_coverage": 0.9,       # agents answering fewer questions are dropped before scoring
    # Selection and scoring
    "k": 3,
    "hap_weights": {"correlation": 0.05, "provenance": 0.05, "price": 0.0},
    "contribution_min_gain": 0.002,
    "requirement": {"max_brier": 0.20, "max_confident_wrong": 0.25, "budget_usd": 0.05},
    "non_inferiority_margin": 0.02,
    "primary_metric": "brier",
}

SYSTEM_PROMPT = (
    "You are a careful forecaster. You give calibrated probabilities for questions "
    "about the future. You have no internet access and no tools; use what you know. "
    "If you are unsure, say so through the probability, not by declining."
)

PROMPTS = {
    "direct": (
        "Today is {today}. Estimate the probability that this question resolves YES.\n\n"
        "Question: {title}\n\nResolution details:\n{description}\n\n"
        "The question closes on {close_date}.\n\n"
        "Think it through briefly, then end your reply with one line of the form\n"
        "Probability: 0.XX"
    ),
    "base_rate": (
        "Today is {today}. Estimate the probability that this question resolves YES.\n\n"
        "Question: {title}\n\nResolution details:\n{description}\n\n"
        "The question closes on {close_date}.\n\n"
        "First name one or two reference classes and their base rates. Then adjust for "
        "what is specific to this case. End your reply with one line of the form\n"
        "Probability: 0.XX"
    ),
}

AGENT_FIELDS = ("agent_id", "provider", "model", "family", "prompt", "price_in", "price_out")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utc_now() -> float:
    return time.time()


def iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)


def _float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """Read the pilot config and fill in the default rules. Raises ValueError."""
    cfg = json.loads(Path(path).read_text())
    rules = dict(DEFAULT_RULES)
    unknown = set(cfg.get("rules", {})) - set(rules)
    if unknown:
        raise ValueError(f"unknown rule(s): {', '.join(sorted(unknown))}")
    rules.update(cfg.get("rules", {}))
    agents = cfg.get("agents") or []
    if len(agents) < 2:
        raise ValueError("the config needs at least two agents")
    seen = set()
    for a in agents:
        missing = [f for f in AGENT_FIELDS if a.get(f) in (None, "")]
        if missing:
            raise ValueError(f"agent {a.get('agent_id', '?')}: missing {', '.join(missing)}")
        if "FILL" in str(a["model"]).upper():
            raise ValueError(f"agent {a['agent_id']}: replace the placeholder model name")
        if a["agent_id"] in seen:
            raise ValueError(f"duplicate agent_id {a['agent_id']}")
        seen.add(a["agent_id"])
        if a["prompt"] not in PROMPTS:
            raise ValueError(f"agent {a['agent_id']}: prompt must be one of {', '.join(PROMPTS)}")
        if a["provider"] not in PROVIDERS:
            raise ValueError(f"agent {a['agent_id']}: provider must be one of {', '.join(PROVIDERS)}")
        if a["provider"] == "command" and not a.get("command"):
            raise ValueError(f"agent {a['agent_id']}: a command agent needs \"command\"")
    cheap = cfg.get("cheap_baseline")
    if cheap not in seen:
        raise ValueError("cheap_baseline must name one of the agents")
    platforms = cfg.get("platforms", list(PLATFORMS))
    if not platforms or set(platforms) - set(PLATFORMS):
        raise ValueError(f"platforms must be a subset of {', '.join(PLATFORMS)}")
    return {"name": cfg.get("name", "pilot"), "agents": agents, "cheap_baseline": cheap,
            "platforms": platforms, "rules": rules}


# ---------------------------------------------------------------------------
# Open questions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OpenQuestion:
    question_id: str
    platform: str
    market_id: str          # id used to look the market up again
    title: str
    description: str
    url: Optional[str]
    close_time: float
    market_price: float     # probability of YES at the snapshot
    liquidity: float        # bettors, USD volume or contracts, by platform
    event: Optional[str] = None
    split: str = ""


def _in_window(close: Optional[float], now: float, rules: Dict[str, Any]) -> bool:
    return (close is not None
            and now + rules["min_days_to_close"] * DAY <= close <= now + rules["max_days_to_close"] * DAY)


def _price_ok(p: Optional[float], rules: Dict[str, Any]) -> bool:
    return p is not None and rules["min_price"] <= p <= rules["max_price"]


def _midpoint(bid: Optional[float], ask: Optional[float], rules: Dict[str, Any]) -> Optional[float]:
    bid, ask = usable_quote(bid, ask)
    if bid is None or ask is None or ask - bid > rules["max_spread"]:
        return None
    return 0.5 * (bid + ask)


def open_manifold(rules: Dict[str, Any], now: float, fetch=fetch_http_json, pages: int = 4) -> List[OpenQuestion]:
    out: List[OpenQuestion] = []
    offset = 0
    for _ in range(pages):
        query = urllib.parse.urlencode({"filter": "open", "contractType": "BINARY",
                                        "sort": "liquidity", "limit": "500", "offset": str(offset)})
        try:
            batch = fetch(f"{MANIFOLD_API_BASE}/search-markets?{query}")
        except RuntimeError:
            if offset == 0:
                raise
            break  # the search API caps how deep offset may go
        if not isinstance(batch, list) or not batch:
            break
        offset += len(batch)
        for m in batch:
            close = parse_timestamp(m.get("closeTime"))
            p = _float(m.get("probability"))
            bettors = m.get("uniqueBettorCount") or 0
            if (m.get("outcomeType") == "BINARY" and not m.get("isResolved")
                    and _in_window(close, now, rules) and _price_ok(p, rules)
                    and bettors >= rules["manifold_min_bettors"]):
                out.append(OpenQuestion(f"MNF_{m['id']}", "manifold", str(m["id"]),
                                        str(m.get("question", "")).strip(), "", m.get("url"),
                                        close, p, float(bettors)))
    return out


def open_polymarket(rules: Dict[str, Any], now: float, fetch=fetch_http_json, pages: int = 6) -> List[OpenQuestion]:
    out: List[OpenQuestion] = []
    for page in range(pages):
        query = urllib.parse.urlencode({
            "closed": "false", "active": "true", "limit": "500", "offset": str(page * 500),
            "order": "volumeNum", "ascending": "false",
            "end_date_min": iso(now + rules["min_days_to_close"] * DAY),
            "end_date_max": iso(now + rules["max_days_to_close"] * DAY)})
        try:
            batch = fetch(f"{GAMMA_API_BASE}/markets?{query}")
        except RuntimeError:
            if page == 0:
                raise
            break  # the API caps how deep offset may go
        if not isinstance(batch, list) or not batch:
            break
        for m in batch:
            try:
                outcomes = [str(o).lower() for o in json.loads(m.get("outcomes") or "[]")]
            except (TypeError, ValueError):
                continue
            if sorted(outcomes) != ["no", "yes"]:
                continue
            p = _midpoint(_float(m.get("bestBid")), _float(m.get("bestAsk")), rules)
            close = parse_timestamp(m.get("endDate"))
            volume = _float(m.get("volumeNum")) or 0.0
            if (_in_window(close, now, rules) and _price_ok(p, rules)
                    and volume >= rules["polymarket_min_volume"]):
                events = m.get("events") or []
                event = str(events[0].get("id")) if events and isinstance(events[0], dict) else None
                out.append(OpenQuestion(f"PM_{m['id']}", "polymarket", str(m["id"]),
                                        str(m.get("question", "")).strip(),
                                        str(m.get("description") or ""),
                                        f"https://polymarket.com/market/{m.get('slug', '')}",
                                        close, p, volume, event))
    return out


def open_kalshi(rules: Dict[str, Any], now: float, fetch=fetch_http_json, pages: int = 20) -> List[OpenQuestion]:
    out: List[OpenQuestion] = []
    cursor = ""
    for _ in range(pages):
        params = {"status": "open", "mve_filter": "exclude", "limit": "1000",
                  "min_close_ts": str(int(now + rules["min_days_to_close"] * DAY)),
                  "max_close_ts": str(int(now + rules["max_days_to_close"] * DAY))}
        if cursor:
            params["cursor"] = cursor
        data = fetch(f"{KALSHI_API_BASE}/markets?{urllib.parse.urlencode(params)}")
        markets = data.get("markets", []) if isinstance(data, dict) else []
        for m in markets:
            if m.get("market_type", "binary") != "binary":
                continue
            p = _midpoint(_float(m.get("yes_bid_dollars")), _float(m.get("yes_ask_dollars")), rules)
            close = parse_timestamp(m.get("close_time"))
            volume = _float(m.get("volume_fp")) or _float(m.get("volume")) or 0.0
            if _in_window(close, now, rules) and _price_ok(p, rules) and volume >= rules["kalshi_min_volume"]:
                title = str(m.get("title", "")).strip()
                sub = str(m.get("yes_sub_title") or "").strip()
                if sub and sub.lower() not in title.lower():
                    title = f"{title} ({sub})"
                rules_text = "\n\n".join(str(m.get(k) or "") for k in ("rules_primary", "rules_secondary")).strip()
                out.append(OpenQuestion(f"KALSHI_{m['ticker']}", "kalshi", str(m["ticker"]), title,
                                        rules_text, f"https://kalshi.com/markets/{m['ticker']}",
                                        close, p, volume, m.get("event_ticker")))
        cursor = data.get("cursor") if isinstance(data, dict) else ""
        if not cursor or not markets:
            break
    return out


OPEN_FETCHERS: Dict[str, Callable[..., List[OpenQuestion]]] = {
    "manifold": open_manifold, "polymarket": open_polymarket, "kalshi": open_kalshi,
}


def manifold_description(market_id: str, fetch=fetch_http_json) -> str:
    data = fetch(f"{MANIFOLD_API_BASE}/market/{urllib.parse.quote(market_id)}")
    return str(data.get("textDescription") or "") if isinstance(data, dict) else ""


def sample_questions(candidates: Dict[str, List[OpenQuestion]], rules: Dict[str, Any]) -> List[OpenQuestion]:
    """Fixed sampling: one market per event (the most liquid), then an equal
    quota per platform drawn by a seeded shuffle, then alternate train/test
    within each platform so both splits have the same mix."""
    rng = random.Random(rules["seed"])
    platforms = [p for p in candidates if candidates[p]]
    if not platforms:
        raise RuntimeError("no open question passed the filters")
    pools: Dict[str, List[OpenQuestion]] = {}
    for platform in platforms:
        qs = sorted(candidates[platform], key=lambda q: (-q.liquidity, q.question_id))
        if rules["one_market_per_event"]:
            seen, kept = set(), []
            for q in qs:
                if q.event and q.event in seen:
                    continue
                seen.add(q.event)
                kept.append(q)
            qs = kept
        qs = sorted(qs, key=lambda q: q.question_id)
        rng.shuffle(qs)
        pools[platform] = qs
    quota = rules["n_questions"] // len(platforms)
    chosen: List[OpenQuestion] = []
    spare: List[OpenQuestion] = []
    for platform in platforms:
        chosen.extend(pools[platform][:quota])
        spare.extend(pools[platform][quota:])
    rng.shuffle(spare)
    chosen.extend(spare[: rules["n_questions"] - len(chosen)])
    by_platform: Dict[str, int] = {}
    for q in chosen:
        i = by_platform.get(q.platform, 0)
        q.split = "train" if i % 2 == 0 else "test"
        by_platform[q.platform] = i + 1
    return chosen


def register(config: Dict[str, Any], run_dir: Path, now: Optional[float] = None,
             fetch=fetch_http_json, log=print) -> Path:
    run_dir = Path(run_dir)
    target = run_dir / "registration.json"
    if target.exists():
        raise RuntimeError(f"{target} exists; a registration is never overwritten")
    now = utc_now() if now is None else now
    rules = config["rules"]
    candidates: Dict[str, List[OpenQuestion]] = {}
    for platform in config["platforms"]:
        candidates[platform] = OPEN_FETCHERS[platform](rules, now, fetch=fetch)
        log(f"  {platform}: {len(candidates[platform])} open questions pass the filters")
    chosen = sample_questions(candidates, rules)
    for q in chosen:
        if q.platform == "manifold" and not q.description:
            q.description = manifold_description(q.market_id, fetch=fetch)
    registration = {
        "name": config["name"],
        "registered_at": iso(now),
        "registered_ts": now,
        "rules": rules,
        "agents": config["agents"],
        "cheap_baseline": config["cheap_baseline"],
        "platforms": config["platforms"],
        "candidates_per_platform": {p: len(v) for p, v in candidates.items()},
        "questions": [dataclasses.asdict(q) for q in chosen],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(registration) + "\n")
    digest = sha256_file(target)
    (run_dir / "registration.sha256").write_text(f"{digest}  registration.json\n")
    n_train = sum(q.split == "train" for q in chosen)
    log(f"Registered {len(chosen)} questions ({n_train} train, {len(chosen) - n_train} test) "
        f"in {target}\nSHA-256 {digest}\nCommit and push the run folder now: "
        "the push time is the public record that the plan came before the forecasts.")
    return target


def load_registration(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "registration.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run register first")
    stored = (Path(run_dir) / "registration.sha256").read_text().split()[0]
    if sha256_file(path) != stored:
        raise RuntimeError(f"{path} changed after registration (hash mismatch); the plan is frozen")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelReply:
    text: str
    input_tokens: int
    output_tokens: int
    status: str = "ok"           # ok | refused
    cost_usd: Optional[float] = None


def call_anthropic(agent: Dict[str, Any], system: str, prompt: str) -> ModelReply:
    import anthropic  # optional dependency: pip install anthropic

    client = anthropic.Anthropic()
    kwargs: Dict[str, Any] = {}
    if agent.get("effort"):
        kwargs["output_config"] = {"effort": agent["effort"]}
    # No refusal fallback on purpose: a fallback would let another model answer
    # under this agent's name. A refusal is recorded as a missing forecast.
    response = client.messages.create(
        model=agent["model"], max_tokens=16000, system=system,
        messages=[{"role": "user", "content": prompt}], **kwargs,
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    status = "refused" if response.stop_reason == "refusal" else "ok"
    return ModelReply(text, response.usage.input_tokens, response.usage.output_tokens, status)


def call_openai(agent: Dict[str, Any], system: str, prompt: str) -> ModelReply:
    from openai import OpenAI  # optional dependency: pip install openai

    kwargs: Dict[str, Any] = {}
    if agent.get("effort"):
        kwargs["reasoning"] = {"effort": agent["effort"]}
    response = OpenAI().responses.create(model=agent["model"], instructions=system, input=prompt, **kwargs)
    usage = response.usage
    return ModelReply(response.output_text or "", usage.input_tokens, usage.output_tokens)


def call_google(agent: Dict[str, Any], system: str, prompt: str) -> ModelReply:
    from google import genai  # optional dependency: pip install google-genai

    response = genai.Client().models.generate_content(
        model=agent["model"], contents=prompt, config={"system_instruction": system})
    meta = response.usage_metadata
    # Thinking tokens are billed as output.
    out = (meta.candidates_token_count or 0) + (getattr(meta, "thoughts_token_count", None) or 0)
    return ModelReply(response.text or "", meta.prompt_token_count or 0, out)


def call_command(agent: Dict[str, Any], system: str, prompt: str) -> ModelReply:
    """Any other model: a command that reads {"system", "prompt", "model"} as JSON
    on stdin and prints {"text", "input_tokens", "output_tokens"} (and optionally
    "cost_usd") as JSON."""
    payload = json.dumps({"system": system, "prompt": prompt, "model": agent["model"]})
    done = subprocess.run(agent["command"], input=payload, capture_output=True, text=True,
                          shell=isinstance(agent["command"], str), timeout=900)
    if done.returncode != 0:
        raise RuntimeError(f"command failed ({done.returncode}): {done.stderr.strip()[:500]}")
    data = json.loads(done.stdout)
    return ModelReply(str(data.get("text", "")), int(data.get("input_tokens", 0)),
                      int(data.get("output_tokens", 0)), str(data.get("status", "ok")),
                      _float(data.get("cost_usd")))


PROVIDERS: Dict[str, Callable[[Dict[str, Any], str, str], ModelReply]] = {
    "anthropic": call_anthropic, "openai": call_openai, "google": call_google, "command": call_command,
}

PROB_RE = re.compile(r"probability\s*\**\s*[:=]\s*\**\s*(\d*\.?\d+)\s*(%)?", re.IGNORECASE)


def parse_probability(text: str) -> Optional[float]:
    """The last 'Probability: x' in the reply; accepts 0.37, .37 and 37%."""
    matches = list(PROB_RE.finditer(text or ""))
    if not matches:
        return None
    number, percent = matches[-1].groups()
    value = float(number) / (100.0 if percent else 1.0)
    return value if 0.0 <= value <= 1.0 else None


def build_prompt(agent: Dict[str, Any], q: Dict[str, Any], now: float) -> str:
    return PROMPTS[agent["prompt"]].format(
        today=iso(now)[:10], title=q["title"],
        description=q["description"].strip() or "(none given)",
        close_date=iso(q["close_time"])[:10])


def reply_cost(agent: Dict[str, Any], reply: ModelReply) -> float:
    if reply.cost_usd is not None:
        return reply.cost_usd
    return (reply.input_tokens * float(agent["price_in"]) + reply.output_tokens * float(agent["price_out"])) / 1e6


# ---------------------------------------------------------------------------
# Forecast collection
# ---------------------------------------------------------------------------

def forecast(run_dir: Path, workers: int = 1, dry_run: bool = False,
             only_agents: Optional[Sequence[str]] = None, limit: Optional[int] = None,
             providers: Optional[Dict[str, Callable]] = None, clock=utc_now, log=print) -> Dict[str, int]:
    """Collect missing forecasts. Safe to re-run: answered pairs are skipped and
    failed ones are retried, up to MAX_ATTEMPTS per pair."""
    run_dir = Path(run_dir)
    reg = load_registration(run_dir)
    providers = providers or PROVIDERS
    log_path = run_dir / "forecasts.jsonl"
    records = read_jsonl(log_path)
    done = {(r["question_id"], r["agent_id"]) for r in records if r["status"] == "ok"}
    attempts: Dict[Tuple[str, str], int] = {}
    for r in records:
        key = (r["question_id"], r["agent_id"])
        attempts[key] = attempts.get(key, 0) + 1

    deadline = reg["registered_ts"] + reg["rules"]["max_forecast_lag_hours"] * 3600.0
    agents = [a for a in reg["agents"] if not only_agents or a["agent_id"] in only_agents]
    todo = []
    for q in reg["questions"]:
        for a in agents:
            key = (q["question_id"], a["agent_id"])
            if key not in done and attempts.get(key, 0) < MAX_ATTEMPTS:
                todo.append((q, a))
    if limit is not None:
        todo = todo[:limit]

    now = clock()
    if dry_run:
        for q, a in todo[:2]:
            log(f"--- {a['agent_id']} on {q['question_id']}\n{build_prompt(a, q, now)}\n")
        log(f"{len(todo)} forecasts to collect from {len(agents)} agents.")
        return {"todo": len(todo)}
    if now > deadline:
        raise RuntimeError(f"the forecast window closed at {iso(deadline)} "
                           f"({reg['rules']['max_forecast_lag_hours']} hours after registration)")

    lock = threading.Lock()
    counts = {"ok": 0, "refused": 0, "parse_error": 0, "error": 0}

    def one(item):
        q, a = item
        started = clock()
        prompt = build_prompt(a, q, started)
        record = {"question_id": q["question_id"], "agent_id": a["agent_id"], "model": a["model"],
                  "provider": a["provider"], "family": a["family"], "prompt": a["prompt"],
                  "requested_at": iso(started), "requested_ts": started}
        try:
            reply = providers[a["provider"]](a, SYSTEM_PROMPT, prompt)
            p = parse_probability(reply.text) if reply.status == "ok" else None
            record.update({"status": reply.status if reply.status != "ok" or p is not None else "parse_error",
                           "probability": p, "text": reply.text,
                           "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                           "cost_usd": round(reply_cost(a, reply), 6)})
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and retried later
            record.update({"status": "error", "probability": None, "error": f"{type(exc).__name__}: {exc}"[:1000]})
        record["answered_at"] = iso(clock())
        record["answered_ts"] = clock()
        with lock:
            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            total = sum(counts.values())
            if total % 10 == 0:
                log(f"  {total} of {len(todo)} done")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))

    if log_path.exists():
        digest = sha256_file(log_path)
        (run_dir / "forecasts.sha256").write_text(f"{digest}  forecasts.jsonl\n")
        log(f"forecasts.jsonl SHA-256 {digest}")
    log(f"Collected: {counts}. Commit and push the run folder to timestamp the forecasts.")
    return counts


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def lookup_market(q: Dict[str, Any], fetch=fetch_http_json):
    """Fetch one market and parse it with the platform's importer. Returns a
    RawMarket when the question resolved YES or NO, otherwise None."""
    mid = urllib.parse.quote(q["market_id"])
    if q["platform"] == "manifold":
        return ManifoldImporter().parse_market(fetch(f"{MANIFOLD_API_BASE}/market/{mid}"))
    if q["platform"] == "polymarket":
        raw = fetch(f"{GAMMA_API_BASE}/markets/{mid}")
        if not isinstance(raw, dict) or not raw.get("closed"):
            return None
        imp = PolymarketImporter()
        imp.require_final_price = False
        return imp.parse_market(raw)
    if q["platform"] == "kalshi":
        data = fetch(f"{KALSHI_API_BASE}/markets/{mid}")
        raw = data.get("market", data) if isinstance(data, dict) else {}
        imp = KalshiImporter()
        imp.require_final_price = False
        return imp.parse_market(raw)
    raise ValueError(f"unknown platform {q['platform']}")


def resolve(run_dir: Path, fetch=fetch_http_json, clock=utc_now, log=print) -> Dict[str, int]:
    run_dir = Path(run_dir)
    reg = load_registration(run_dir)
    path = run_dir / "resolutions.json"
    resolutions = json.loads(path.read_text()) if path.exists() else {}
    counts = {"resolved": 0, "pending": 0, "failed": 0}
    for q in reg["questions"]:
        qid = q["question_id"]
        if qid in resolutions:
            counts["resolved"] += 1
            continue
        try:
            market = lookup_market(q, fetch=fetch)
        except RuntimeError as exc:
            log(f"  {qid}: lookup failed ({exc})")
            counts["failed"] += 1
            continue
        if market is None:
            counts["pending"] += 1
            continue
        resolutions[qid] = {"outcome": market.outcome, "close_time": market.close_time,
                            "resolution_time": market.resolution_time, "checked_at": iso(clock())}
        counts["resolved"] += 1
    path.write_text(canonical_json(resolutions) + "\n")
    log(f"{counts['resolved']} resolved, {counts['pending']} not resolved yet (or N/A), "
        f"{counts['failed']} lookups failed. Re-run later for the rest.")
    return counts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GroupScore:
    label: str
    agents: List[str]
    n: int
    brier: float
    log_loss: float
    confidently_wrong: int
    cost: float  # USD per question: mean actual cost of each member on these questions, summed


@dataclasses.dataclass
class Comparison:
    label: str
    other: str
    diff: float  # Brier of label minus Brier of other; < 0 means label is better
    low: float
    high: float


def usable_forecasts(reg: Dict[str, Any], records: Sequence[Dict[str, Any]],
                     resolutions: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, int]]:
    """Keep the first successful forecast per pair, made within the forecast
    window and before the question closed or resolved."""
    deadline = reg["registered_ts"] + reg["rules"]["max_forecast_lag_hours"] * 3600.0
    questions = {q["question_id"]: q for q in reg["questions"]}
    kept: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dropped = {"late": 0, "after_close": 0}
    for r in records:
        if r.get("status") != "ok" or r.get("probability") is None:
            continue
        key = (r["question_id"], r["agent_id"])
        if key in kept or r["question_id"] not in questions:
            continue
        if r["answered_ts"] > deadline:
            dropped["late"] += 1
            continue
        res = resolutions.get(r["question_id"], {})
        ends = [t for t in (questions[r["question_id"]]["close_time"], res.get("close_time"),
                            res.get("resolution_time")) if t is not None]
        if ends and r["answered_ts"] >= min(ends):
            dropped["after_close"] += 1
            continue
        kept[key] = r
    return kept, dropped


def build_datasets(reg: Dict[str, Any], records: Sequence[Dict[str, Any]], resolutions: Dict[str, Any]):
    """Return (pool dataset, dataset with the market added, per-question costs, notes)."""
    rules = reg["rules"]
    kept, dropped = usable_forecasts(reg, records, resolutions)
    resolved = [q for q in reg["questions"] if q["question_id"] in resolutions]
    notes = [f"{len(reg['questions'])} registered, {len(resolved)} resolved YES or NO."]
    if dropped["late"] or dropped["after_close"]:
        notes.append(f"Forecasts not scored: {dropped['late']} after the forecast window, "
                     f"{dropped['after_close']} after the question closed.")
    agents = []
    for a in reg["agents"]:
        have = sum((q["question_id"], a["agent_id"]) in kept for q in resolved)
        coverage = have / len(resolved) if resolved else 0.0
        if coverage >= rules["min_agent_coverage"]:
            agents.append(a)
        else:
            notes.append(f"Agent {a['agent_id']} dropped: answered {have} of {len(resolved)} resolved questions.")
    qs = [q for q in resolved if all((q["question_id"], a["agent_id"]) in kept for a in agents)]
    if len(qs) < len(resolved):
        notes.append(f"{len(resolved) - len(qs)} resolved questions left out because a kept agent has no forecast.")
    questions = {q["question_id"]: hx.Question(q["question_id"], q["title"],
                                               int(resolutions[q["question_id"]]["outcome"]), q["split"])
                 for q in qs}
    if not any(q.split == "train" for q in questions.values()) or not any(q.split == "test" for q in questions.values()):
        raise ValueError("need resolved questions in both the train and the test split")
    costs = {(q, a["agent_id"]): float(kept[(q, a["agent_id"])].get("cost_usd") or 0.0)
             for q in questions for a in agents}
    train = [q for q, v in questions.items() if v.split == "train"]
    hap_agents = {}
    for a in agents:
        mean_cost = sum(costs[(q, a["agent_id"])] for q in train) / len(train)
        hap_agents[a["agent_id"]] = hx.Agent(a["agent_id"], a["family"], a["provider"],
                                             f"{a['prompt']}_no_tools", mean_cost)
    forecasts = {(q, aid): float(kept[(q, aid)]["probability"]) for q in questions for aid in hap_agents}
    pool = hx.Dataset(questions, hap_agents, forecasts)
    market_agent = hx.Agent("market", "market", "market", "price_at_registration", 0.0)
    prices = {q["question_id"]: q["market_price"] for q in qs}
    with_market = hx.Dataset(questions, {**hap_agents, "market": market_agent},
                             {**forecasts, **{(q, "market"): prices[q] for q in questions}})
    return pool, with_market, costs, notes


def score_group(data: hx.Dataset, label: str, agent_ids: Sequence[str], costs: Dict[Tuple[str, str], float],
                split: str = "test") -> GroupScore:
    qids = data.question_ids(split)
    result = hx.evaluate_pool(data, label, agent_ids, split)
    cost = sum(sum(costs.get((q, a), 0.0) for q in qids) / len(qids) for a in agent_ids)
    return GroupScore(label, list(agent_ids), len(qids), result.brier, result.log_loss,
                      result.confidently_wrong, cost)


def compare(data: hx.Dataset, a: GroupScore, b: GroupScore, split: str = "test") -> Comparison:
    qids = data.question_ids(split)
    diff = [x - y for x, y in zip(hc.per_question_brier(data, a.agents, qids),
                                  hc.per_question_brier(data, b.agents, qids))]
    boots = hc.bootstrap_means(diff, hc.bootstrap_indices(len(qids)))
    return Comparison(a.label, b.label, sum(diff) / len(diff),
                      hc.percentile(boots, 0.025), hc.percentile(boots, 0.975))


def score(run_dir: Path, log=print) -> str:
    run_dir = Path(run_dir)
    reg = load_registration(run_dir)
    rules = reg["rules"]
    path = run_dir / "resolutions.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run resolve first")
    pool, with_market, costs, notes = build_datasets(reg, read_jsonl(run_dir / "forecasts.jsonl"),
                                                     json.loads(path.read_text()))
    k = min(rules["k"], len(pool.agents))
    weights = hx.HAPWeights(**rules["hap_weights"])
    selected: Dict[str, List[str]] = {}
    for name, select in (("hap", lambda d: hx.select_hap(d, k, weights)),
                         ("contribution", lambda d: hx.select_contribution(d, k, rules["contribution_min_gain"])),
                         ("same_model", lambda d: hx.select_same_model(d, k)),
                         ("naive_mix", lambda d: hx.select_naive_mix(d, k))):
        try:
            selected[name] = select(pool)
        except ValueError as exc:
            notes.append(f"{hx.NAMES[name]} not formed: {exc}.")
    req = rules["requirement"]
    buyer = hc.buyer_test(pool, hc.Requirement(req["max_brier"], req["max_confident_wrong"], req["budget_usd"]),
                          max_size=min(3, len(pool.agents)), k=k)

    cheap = reg["cheap_baseline"]
    groups: List[GroupScore] = []
    labels = {"hap": "HAP diversity-aware", "contribution": "HAP contribution-based",
              "same_model": "Same model repeated", "naive_mix": "Naive mix"}
    for name, ids in selected.items():
        groups.append(score_group(with_market, labels[name], ids, costs))
    if buyer.chosen:
        groups.append(score_group(with_market, "HAP buyer test choice", list(buyer.chosen.agents), costs))
    if cheap in pool.agents:
        groups.append(score_group(with_market, f"Cheap single model ({cheap})", [cheap], costs))
    else:
        notes.append(f"Cheap baseline {cheap} was dropped, so it cannot be compared.")
    groups.append(score_group(with_market, "Average of all models", sorted(pool.agents), costs))
    groups.append(score_group(with_market, "Market price at registration", ["market"], costs))
    by_label = {g.label: g for g in groups}

    comparators = [g for g in groups if g.label.startswith(("Cheap", "Average", "Market"))]
    hap_like = [g for g in groups if g.label.startswith("HAP")]
    comparisons = [compare(with_market, h, c) for h in hap_like for c in comparators]

    margin = rules["non_inferiority_margin"]
    verdicts = []
    for h in hap_like:
        for c in comparators:
            cmp_ = next(x for x in comparisons if x.label == h.label and x.other == c.label)
            if cmp_.high < 0:
                verdict = "better"
            elif cmp_.low > 0:
                verdict = "worse"
            elif cmp_.high < margin:
                verdict = "not worse by more than the margin"
            else:
                verdict = "unclear"
            cheaper = h.cost < c.cost
            if verdict == "not worse by more than the margin" and cheaper and c.label != "Market price at registration":
                verdict += ", and cheaper"
            verdicts.append((h.label, c.label, verdict))

    report = format_score_report(reg, pool, groups, comparisons, verdicts, buyer, notes, margin)
    (run_dir / "report.md").write_text(report + "\n")
    log(report)
    return report


def format_score_report(reg, pool, groups, comparisons, verdicts, buyer, notes, margin) -> str:
    n_train = len(pool.question_ids("train"))
    n_test = len(pool.question_ids("test"))
    lines = [
        f"# HAP pilot: {reg['name']}",
        "",
        f"Registered {reg['registered_at']}. {n_train} train questions (used to select), "
        f"{n_test} test questions (used to score). Rules frozen at registration; see registration.json.",
        "",
        "## Test questions",
        "",
        "| Group | Members | Brier | Log loss | Confidently wrong | Cost per question (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for g in groups:
        lines.append(f"| {g.label} | {', '.join(g.agents)} | {g.brier:.3f} | {g.log_loss:.3f} "
                     f"| {g.confidently_wrong} of {g.n} | {g.cost:.4f} |")
    lines += [
        "",
        "## HAP against the comparators",
        "",
        "Difference in Brier (HAP minus comparator; below 0 means HAP is better), "
        "with a 95% paired bootstrap interval over the test questions.",
        "",
        "| HAP group | Comparator | Difference | 95% interval | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c, (_, _, verdict) in zip(comparisons, verdicts):
        lines.append(f"| {c.label} | {c.other} | {c.diff:+.3f} | {c.low:+.3f} to {c.high:+.3f} | {verdict} |")
    lines += [
        "",
        f"Verdicts follow the rule fixed at registration: better if the whole interval is below 0, "
        f"worse if it is above 0, not worse by more than the margin if the upper end is below "
        f"{margin:.2f}, otherwise unclear. Cheaper compares actual cost per question on the test questions.",
        "",
        "## Buyer test (train questions)",
        "",
    ]
    req = buyer.requirement
    lines.append(f"Requirement: Brier at most {req.max_brier:.2f}, confidently wrong at most "
                 f"{req.max_confident_wrong:.0%}, cost at most {req.budget:.3f} USD per question.")
    if buyer.chosen:
        t = buyer.chosen_test
        lines.append(f"Chosen: {buyer.chosen.label} at {buyer.chosen.cost:.4f} USD. On the test questions: "
                     f"Brier {t.brier:.3f} (pessimistic {t.brier_upper:.3f}), "
                     f"meets the requirement: {'yes' if t.meets(req) else 'no'}.")
    else:
        lines.append("No option met the requirement on the train questions.")
    lines += ["", "## Notes", ""] + [f"- {n}" for n in notes] + [
        f"- Forecasts are single answers from models without tools; each agent's cost is what its "
        "calls actually cost. The market comparator is the price when the questions were registered.",
        f"- With {n_test} test questions an interval of a few hundredths of Brier is normal. "
        "A pilot can show a direction and the size of the uncertainty, not settle the case.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("register", help="snapshot open questions and freeze the plan")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("forecast", help="collect model forecasts")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="show prompts and counts, call nothing")
    p.add_argument("--agents", nargs="*", help="only these agent ids")
    p.add_argument("--limit", type=int, help="at most this many calls (for a trial run)")
    p = sub.add_parser("resolve", help="look up outcomes")
    p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("score", help="select on train, compare on test")
    p.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "register":
            register(load_config(args.config), args.run)
        elif args.cmd == "forecast":
            forecast(args.run, workers=args.workers, dry_run=args.dry_run,
                     only_agents=args.agents, limit=args.limit)
        elif args.cmd == "resolve":
            resolve(args.run)
        else:
            score(args.run)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
