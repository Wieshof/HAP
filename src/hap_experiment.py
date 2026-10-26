"""hap_experiment.py - HAP diversity benchmark.

Compares four ways of forming a group of forecasting agents on resolved
binary questions:

  1. Same model repeated: k agents from the model family that scored best on
     the training split.
  2. Naive mix: the best agent from each of the k best families, averaged.
  3. HAP diversity-aware: greedy selection that trades training accuracy
     against error correlation, declared provenance overlap and price.
  4. Contribution-based: greedy selection that adds whichever agent most
     improves the group's averaged forecast, and stops when nothing helps.

Selection sees only the training split. Every metric is reported on the test
split. Group forecasts are the plain average of the members' probabilities.

Metrics (lower is better for all of them):
  * Average error (Brier): mean squared error of the group forecast.
  * Log loss: mean negative log-likelihood of the group forecast.
  * Everyone wrong: questions where every member gave the true outcome < 50%.
  * Confidently wrong: questions where the group gave the wrong outcome >= 75%.
  * Tail error: mean Brier of the worst quarter of questions.
  * Shared error: mean pairwise Pearson correlation of members' per-question
    misses |forecast - outcome|. 1.00 means they miss on the same questions
    by the same amounts; about 0 means unrelated mistakes.

The bundled smoke data are synthetic. They test the plumbing, not the thesis.

Usage:
  python src/hap_experiment.py --generate-smoke     # write smoke data, then run
  python src/hap_experiment.py                      # run on data/
  python src/hap_experiment.py --questions Q.csv --predictions P.csv --out DIR
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-6


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    outcome: int  # 0 or 1
    split: str    # "train" or "test"


@dataclasses.dataclass(frozen=True)
class Agent:
    agent_id: str
    model_family: str
    operator: str
    pipeline: str
    price: float


@dataclasses.dataclass(frozen=True)
class Dataset:
    questions: Dict[str, Question]
    agents: Dict[str, Agent]
    forecasts: Dict[Tuple[str, str], float]  # (question_id, agent_id) -> p

    def question_ids(self, split: str) -> List[str]:
        return sorted(q for q, v in self.questions.items() if v.split == split)

    def forecast(self, question_id: str, agent_id: str) -> float:
        try:
            return self.forecasts[(question_id, agent_id)]
        except KeyError:
            raise ValueError(f"missing forecast for {agent_id} on {question_id}") from None


QUESTION_COLUMNS = ("question_id", "text", "outcome", "split")
PREDICTION_COLUMNS = ("question_id", "agent_id", "model_family", "operator",
                      "pipeline", "price", "probability")


def _read_rows(path: Path, columns: Sequence[str]) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {', '.join(missing)}; "
                f"expected {', '.join(columns)}"
            )
        return list(reader)


def _number(row: Dict[str, str], column: str, where: str, kind: type = float):
    try:
        return kind(row[column])
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {column} must be a number, got {row[column]!r}") from None


def load_dataset(questions_csv: Path, predictions_csv: Path) -> Dataset:
    """Read questions and predictions. Predictions carry the agent's declared
    provenance and price on every row; they must agree across rows.
    Problems are reported as ValueError naming the file and line."""
    questions: Dict[str, Question] = {}
    for line, row in enumerate(_read_rows(questions_csv, QUESTION_COLUMNS), start=2):
        where = f"{questions_csv}, line {line}"
        outcome = _number(row, "outcome", where, int)
        if outcome not in (0, 1):
            raise ValueError(f"{where}: outcome must be 0 or 1")
        if row["split"] not in ("train", "test"):
            raise ValueError(f"{where}: split must be train or test")
        questions[row["question_id"]] = Question(
            row["question_id"], row["text"], outcome, row["split"]
        )

    agents: Dict[str, Agent] = {}
    forecasts: Dict[Tuple[str, str], float] = {}
    for line, row in enumerate(_read_rows(predictions_csv, PREDICTION_COLUMNS), start=2):
        where = f"{predictions_csv}, line {line}"
        agent = Agent(
            row["agent_id"], row["model_family"], row["operator"],
            row["pipeline"], _number(row, "price", where),
        )
        if agents.setdefault(agent.agent_id, agent) != agent:
            raise ValueError(f"{where}: inconsistent metadata for {agent.agent_id}")
        p = _number(row, "probability", where)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{where}: probability must be between 0 and 1")
        if row["question_id"] not in questions:
            raise ValueError(f"{where}: unknown question {row['question_id']}")
        key = (row["question_id"], agent.agent_id)
        if key in forecasts:
            raise ValueError(f"{where}: duplicate forecast {key}")
        forecasts[key] = p

    for qid in questions:
        for aid in agents:
            if (qid, aid) not in forecasts:
                raise ValueError(f"missing forecast for {aid} on {qid}")
    return Dataset(questions, agents, forecasts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def log_loss(p: float, y: int) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -math.log(p if y == 1 else 1 - p)


def is_wrong(p: float, y: int) -> bool:
    """Wrong means giving the true outcome less than 50%. Exactly 50% is not
    wrong for either outcome, so the rule is symmetric."""
    return (p if y == 1 else 1 - p) < 0.5


def is_confidently_wrong(p: float, y: int, threshold: float = 0.75) -> bool:
    return (1 - p if y == 1 else p) >= threshold


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation. Returns 0.0 when either series is constant,
    because no co-movement can be measured."""
    n = len(x)
    if n != len(y):
        raise ValueError("series differ in length")
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def residuals(data: Dataset, agent_id: str, qids: Sequence[str]) -> List[float]:
    return [data.forecast(q, agent_id) - data.questions[q].outcome for q in qids]


def errors(data: Dataset, agent_id: str, qids: Sequence[str]) -> List[float]:
    """Per-question miss: how far the forecast was from the outcome, |p - y|.

    Shared-error measures correlate these, not the signed residuals p - y.
    Signed residuals are negative whenever the outcome is 1 and positive when
    it is 0, so any two forecasters that lean toward the truth look correlated
    even if their mistakes are independent."""
    return [abs(data.forecast(q, agent_id) - data.questions[q].outcome) for q in qids]


def mean_pairwise(values: Dict[str, List[float]]) -> float:
    ids = list(values)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    if not pairs:
        return 0.0
    return sum(pearson(values[a], values[b]) for a, b in pairs) / len(pairs)


def provenance_overlap(a: Agent, b: Agent) -> float:
    """Share of declared provenance fields two agents have in common."""
    fields = ("model_family", "operator", "pipeline")
    return sum(getattr(a, f) == getattr(b, f) for f in fields) / len(fields)


def agent_brier(data: Dataset, agent_id: str, qids: Sequence[str]) -> float:
    return sum(r * r for r in residuals(data, agent_id, qids)) / len(qids)


@dataclasses.dataclass
class PoolResult:
    pool: str
    agents: List[str]
    n_questions: int
    brier: float
    log_loss: float
    everyone_wrong: int
    confidently_wrong: int
    tail_brier: float
    shared_error: float


def evaluate_pool(data: Dataset, pool: str, agent_ids: Sequence[str],
                  split: str = "test") -> PoolResult:
    qids = data.question_ids(split)
    if not qids or not agent_ids:
        raise ValueError("need at least one question and one agent")
    briers, losses = [], []
    everyone_wrong = confidently_wrong = 0
    for q in qids:
        y = data.questions[q].outcome
        probs = [data.forecast(q, a) for a in agent_ids]
        group = sum(probs) / len(probs)
        briers.append(brier(group, y))
        losses.append(log_loss(group, y))
        everyone_wrong += all(is_wrong(p, y) for p in probs)
        confidently_wrong += is_confidently_wrong(group, y)
    worst = sorted(briers, reverse=True)[: max(1, len(briers) // 4)]
    return PoolResult(
        pool=pool,
        agents=list(agent_ids),
        n_questions=len(qids),
        brier=sum(briers) / len(briers),
        log_loss=sum(losses) / len(losses),
        everyone_wrong=everyone_wrong,
        confidently_wrong=confidently_wrong,
        tail_brier=sum(worst) / len(worst),
        shared_error=mean_pairwise({a: errors(data, a, qids) for a in agent_ids}),
    )


# ---------------------------------------------------------------------------
# Pool selection (training split only)
# ---------------------------------------------------------------------------

def _ranked_by_brier(data: Dataset, ids: Iterable[str], qids: Sequence[str]) -> List[str]:
    return sorted(ids, key=lambda a: (agent_brier(data, a, qids), a))


def _family_scores(data: Dataset, qids: Sequence[str]) -> List[Tuple[float, str]]:
    families: Dict[str, List[str]] = {}
    for a in data.agents.values():
        families.setdefault(a.model_family, []).append(a.agent_id)
    return sorted(
        (sum(agent_brier(data, a, qids) for a in ids) / len(ids), fam)
        for fam, ids in families.items()
    )


def select_same_model(data: Dataset, k: int) -> List[str]:
    """k agents from the family with the best mean training Brier."""
    qids = data.question_ids("train")
    for _, fam in _family_scores(data, qids):
        ids = [a.agent_id for a in data.agents.values() if a.model_family == fam]
        if len(ids) >= k:
            return _ranked_by_brier(data, ids, qids)[:k]
    raise ValueError(f"no model family has {k} agents")


def select_naive_mix(data: Dataset, k: int) -> List[str]:
    """The best agent from each of the k best families."""
    qids = data.question_ids("train")
    families = [fam for _, fam in _family_scores(data, qids)][:k]
    if len(families) < k:
        raise ValueError(f"need {k} model families, found {len(families)}")
    return [
        _ranked_by_brier(
            data, [a.agent_id for a in data.agents.values() if a.model_family == fam], qids
        )[0]
        for fam in families
    ]


@dataclasses.dataclass(frozen=True)
class HAPWeights:
    """Fixed before looking at test results. Change them only on training data."""
    correlation: float = 0.05
    provenance: float = 0.05
    price: float = 0.0


def select_hap(data: Dataset, k: int, weights: HAPWeights = HAPWeights()) -> List[str]:
    """Greedy selection minimising
        Brier + w_corr * mean error correlation with chosen members
              + w_prov * mean provenance overlap with chosen members
              + w_price * price / max price
    on the training split."""
    qids = data.question_ids("train")
    if len(data.agents) < k:
        raise ValueError(f"need {k} agents, found {len(data.agents)}")
    res = {a: errors(data, a, qids) for a in data.agents}
    acc = {a: agent_brier(data, a, qids) for a in data.agents}
    max_price = max(a.price for a in data.agents.values()) or 1.0

    chosen: List[str] = []
    while len(chosen) < k:
        def cost(a: str) -> Tuple[float, str]:
            c = acc[a] + weights.price * data.agents[a].price / max_price
            if chosen:
                c += weights.correlation * sum(pearson(res[a], res[s]) for s in chosen) / len(chosen)
                c += weights.provenance * sum(
                    provenance_overlap(data.agents[a], data.agents[s]) for s in chosen
                ) / len(chosen)
            return (c, a)
        chosen.append(min((a for a in data.agents if a not in chosen), key=cost))
    return chosen


def group_brier(data: Dataset, agent_ids: Sequence[str], qids: Sequence[str]) -> float:
    """Brier score of the group's averaged forecast."""
    return sum(
        brier(sum(data.forecast(q, a) for a in agent_ids) / len(agent_ids), data.questions[q].outcome)
        for q in qids
    ) / len(qids)


def select_contribution(data: Dataset, k: int, min_gain: float = 0.002) -> List[str]:
    """Select by measured contribution on the training split.

    Try every group of 1 to k agents and score its averaged forecast. Keep the
    smallest group whose Brier is within `min_gain` of the best group found,
    so members that add (almost) nothing are left out. Searching all groups
    rather than adding one agent at a time matters: a weak but different
    agent often only helps once several others are in the group.

    The number of groups grows quickly with the pool, and comparing many on
    few questions favours lucky groups. Fine for a dozen agents; a larger pool
    needs a cheaper search."""
    qids = data.question_ids("train")
    ids = sorted(data.agents)
    best_by_size: List[Tuple[float, Tuple[str, ...]]] = []
    for size in range(1, min(k, len(ids)) + 1):
        best_by_size.append(min(
            (group_brier(data, combo, qids), combo) for combo in itertools.combinations(ids, size)
        ))
    overall = min(score for score, _ in best_by_size)
    for score, combo in best_by_size:
        if score <= overall + min_gain:
            return list(combo)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Synthetic smoke fixture
# ---------------------------------------------------------------------------

SMOKE_AGENTS = [
    # agent_id, family, operator, pipeline, price
    ("gpt_1", "gpt", "openai", "web_search_a", 1.0),
    ("gpt_2", "gpt", "openai", "web_search_a", 1.0),
    ("gpt_3", "gpt", "openai", "web_search_a", 1.0),
    ("gpt_4", "gpt", "openai", "web_search_a", 1.0),
    ("claude_1", "claude", "anthropic", "web_search_a", 1.2),
    ("claude_2", "claude", "anthropic", "web_search_b", 1.2),
    ("gemini_1", "gemini", "google", "web_search_b", 0.9),
    ("gemini_2", "gemini", "google", "web_search_b", 0.9),
    ("local_1", "llama", "self_hosted", "offline_corpus", 0.3),
    ("local_2", "llama", "self_hosted", "offline_corpus", 0.3),
]


def generate_smoke_data(out_dir: Path, n_questions: int = 24, seed: int = 7) -> Tuple[Path, Path]:
    """Write deterministic synthetic questions and forecasts.

    Forecasts are built on the log-odds scale from shared error components:
    one per question for everyone, one per pipeline, one per model family, and
    individual noise. Agents that share a family or pipeline therefore share
    mistakes. This is a plumbing fixture, not a model of real forecasters."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path, p_path = out_dir / "questions.csv", out_dir / "smoke_predictions.csv"

    families = sorted({a[1] for a in SMOKE_AGENTS})
    pipelines = sorted({a[3] for a in SMOKE_AGENTS})
    skill = {"gpt": 1.6, "claude": 1.5, "gemini": 1.4, "llama": 1.0}

    with open(q_path, "w", newline="") as fq, open(p_path, "w", newline="") as fp:
        wq, wp = csv.writer(fq), csv.writer(fp)
        wq.writerow(["question_id", "text", "outcome", "split"])
        wp.writerow(["question_id", "agent_id", "model_family", "operator",
                     "pipeline", "price", "probability"])
        for i in range(n_questions):
            qid = f"Q{i + 1:02d}"
            y = rng.randint(0, 1)
            split = "train" if i < n_questions // 2 else "test"
            wq.writerow([qid, f"Synthetic question {i + 1}", y, split])
            common = rng.gauss(0, 0.8)
            fam_err = {f: rng.gauss(0, 1.6) for f in families}
            pipe_err = {p: rng.gauss(0, 0.8) for p in pipelines}
            sign = 1 if y == 1 else -1
            for aid, fam, op, pipe, price in SMOKE_AGENTS:
                logit = sign * skill[fam] + common + fam_err[fam] + pipe_err[pipe] + rng.gauss(0, 0.3)
                p = 1 / (1 + math.exp(-logit))
                wp.writerow([qid, aid, fam, op, pipe, price, f"{min(max(p, 0.01), 0.99):.4f}"])
    return q_path, p_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_outputs(results: Sequence[PoolResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Python 3.12 made sum() more precise, so the last digits differ between
    # versions. Ten decimals keep committed outputs identical everywhere.
    rows = [
        {k: round(v, 10) if isinstance(v, float) else v for k, v in dataclasses.asdict(r).items()}
        for r in results
    ]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({**row, "agents": ";".join(row["agents"])})
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "report.md").write_text(format_report(results) + "\n")


NAMES = {
    "same_model": "Same model repeated",
    "naive_mix": "Naive mix",
    "hap": "HAP diversity-aware",
    "contribution": "Contribution-based",
}


def format_report(results: Sequence[PoolResult]) -> str:
    n = results[0].n_questions
    lines = [
        "# HAP diversity experiment results",
        "",
        f"Test questions: {n}. Groups have at most {max(len(r.agents) for r in results)} members. "
        "Exact unrounded values are in `results.csv`.",
        "",
        "| Group | Size | Average error (Brier) ↓ | Everyone wrong ↓ | Confidently wrong ↓ | Shared error ↓ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        shared = f"{r.shared_error:.2f}" if len(r.agents) > 1 else "n/a"
        lines.append(
            f"| {NAMES.get(r.pool, r.pool)} | {len(r.agents)} | {r.brier:.2f} | {r.everyone_wrong} of {n} "
            f"| {r.confidently_wrong} of {n} | {shared} |"
        )
    lines += [""] + [f"- {NAMES.get(r.pool, r.pool)}: {', '.join(r.agents)}" for r in results]
    lines += [
        "",
        "Everyone wrong: every member gave the true outcome less than 50%. "
        "Confidently wrong: the group gave the wrong outcome 75% or more. "
        "Shared error: mean pairwise correlation of members' errors (1.00 = identical mistakes).",
        "",
        f"With {n} test questions, small differences are noise. Synthetic data test the "
        "pipeline, not the hypothesis.",
    ]
    return "\n".join(lines)


def run(data: Dataset, k: int = 4, weights: HAPWeights = HAPWeights()) -> List[PoolResult]:
    pools = [
        ("same_model", select_same_model(data, k)),
        ("naive_mix", select_naive_mix(data, k)),
        ("hap", select_hap(data, k, weights)),
        ("contribution", select_contribution(data, k)),
    ]
    return [evaluate_pool(data, name, ids) for name, ids in pools]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "questions.csv")
    ap.add_argument("--predictions", type=Path, default=ROOT / "data" / "smoke_predictions.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "smoke")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--generate-smoke", action="store_true",
                    help="(re)write the synthetic fixture to data/ before running")
    args = ap.parse_args(argv)

    if args.generate_smoke:
        generate_smoke_data(ROOT / "data")
    try:
        data = load_dataset(args.questions, args.predictions)
    except (OSError, ValueError) as exc:
        ap.exit(2, f"error: {exc}\n")
    results = run(data, args.k)
    write_outputs(results, args.out)
    print(format_report(results))


if __name__ == "__main__":
    main()
