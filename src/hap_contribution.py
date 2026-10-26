"""hap_contribution.py - what does each model actually add?

Two analyses built on hap_experiment:

  1. Contribution cases. Four synthetic fixtures, each constructed to show one
     pattern: redundancy, complementarity, illusory diversity and
     counterproductive diversity. For each, a with/without table on the test
     questions: how much worse the group's averaged forecast gets without each
     agent, with a 95% bootstrap interval, next to the agent's measured error
     correlation and declared provenance overlap. Then the groups picked by the
     diversity-aware selector and the contribution-based selector.

  2. Buyer test. The buyer states a requirement: a maximum Brier score, a
     maximum rate of confidently wrong answers and a budget per forecast. On
     the training questions, every single agent and every group of up to three
     (plus the four benchmark groups) is checked against the pessimistic end of
     its interval. The cheapest candidate that passes is chosen, then checked
     again on the untouched test questions.

The fixtures are built to exhibit each pattern. They show that the
measurements detect it, not that real AI systems behave this way.

Usage:
  python src/hap_contribution.py                    # cases + buyer test on data/
  python src/hap_contribution.py --max-brier 0.08 --max-confident-wrong 0.3 --budget 2
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hap_experiment as hx  # noqa: E402

ROOT = hx.ROOT
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 11


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def bootstrap_indices(n: int, samples: int = BOOTSTRAP_SAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> List[List[int]]:
    """Resamples of question positions, shared by every candidate so that
    comparisons use the same draws."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(samples)]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]."""
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap_means(values: Sequence[float], draws: Sequence[Sequence[int]]) -> List[float]:
    return [sum(values[i] for i in d) / len(d) for d in draws]


def rate_upper_bound(events: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound for a rate. Unlike a bootstrap,
    it does not report 0% when no event was seen: with 0 of 12 it gives 22%."""
    if events >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(events + 1))

    lo, hi = events / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# With/without contribution
# ---------------------------------------------------------------------------

def per_question_brier(data: hx.Dataset, agent_ids: Sequence[str], qids: Sequence[str]) -> List[float]:
    return [
        hx.brier(sum(data.forecast(q, a) for a in agent_ids) / len(agent_ids), data.questions[q].outcome)
        for q in qids
    ]


@dataclasses.dataclass
class Contribution:
    agent: str
    family: str
    price: float
    solo_brier: float
    contribution: float  # Brier without the agent minus Brier with it; > 0 means it helps
    low: float
    high: float
    error_correlation: float  # mean correlation of its misses with the other members'
    most_similar: str          # the member whose misses track its own most closely
    most_similar_correlation: float
    provenance_overlap: float  # mean declared overlap with the other members

    @property
    def verdict(self) -> str:
        if self.low > 0:
            return "helps"
        if self.high < 0:
            return "hurts"
        return "unclear"


def with_without(data: hx.Dataset, group: Sequence[str], split: str = "test",
                 draws: Optional[Sequence[Sequence[int]]] = None) -> List[Contribution]:
    """For each member: the change in the group's Brier when it is removed."""
    if len(group) < 2:
        raise ValueError("need at least two members to remove one")
    qids = data.question_ids(split)
    draws = draws or bootstrap_indices(len(qids))
    full = per_question_brier(data, group, qids)
    res = {a: hx.errors(data, a, qids) for a in group}
    rows = []
    for a in group:
        others = [b for b in group if b != a]
        diff = [w - f for w, f in zip(per_question_brier(data, others, qids), full)]
        boots = bootstrap_means(diff, draws)
        agent = data.agents[a]
        corr = {b: hx.pearson(res[a], res[b]) for b in others}
        twin = max(others, key=lambda b: (corr[b], b))
        rows.append(Contribution(
            agent=a,
            family=agent.model_family,
            price=agent.price,
            solo_brier=hx.agent_brier(data, a, qids),
            contribution=sum(diff) / len(diff),
            low=percentile(boots, 0.025),
            high=percentile(boots, 0.975),
            error_correlation=sum(corr.values()) / len(others),
            most_similar=twin,
            most_similar_correlation=corr[twin],
            provenance_overlap=sum(
                hx.provenance_overlap(agent, data.agents[b]) for b in others) / len(others),
        ))
    return rows


# ---------------------------------------------------------------------------
# Case fixtures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SimAgent:
    """A simulated forecaster. Its log-odds are skill toward the true outcome,
    plus the per-question errors of every hidden source it uses, plus its own
    noise. Agents that use the same source share mistakes, whatever their
    declared provenance says."""
    agent_id: str
    family: str
    operator: str
    pipeline: str
    price: float
    skill: float
    sources: Tuple[str, ...]
    noise: float


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    title: str
    question: str
    sources: Dict[str, float]  # hidden error source -> standard deviation
    agents: Tuple[SimAgent, ...]
    k: int = 3
    ladder: Tuple[str, ...] = ()  # optional: report groups of the first 1..n of these


def simulate(case: Case, n_questions: int = 200, seed: int = 3) -> hx.Dataset:
    """Build a dataset for a case. Half the questions train, half test."""
    rng = random.Random(seed)
    questions: Dict[str, hx.Question] = {}
    forecasts: Dict[Tuple[str, str], float] = {}
    for i in range(n_questions):
        qid = f"Q{i + 1:03d}"
        y = rng.randint(0, 1)
        questions[qid] = hx.Question(qid, f"Synthetic question {i + 1}", y,
                                     "train" if i < n_questions // 2 else "test")
        err = {s: rng.gauss(0, sd) for s, sd in sorted(case.sources.items())}
        sign = 1 if y == 1 else -1
        for a in case.agents:
            logit = sign * a.skill + sum(err[s] for s in a.sources) + rng.gauss(0, a.noise)
            p = 1 / (1 + math.exp(-logit))
            forecasts[(qid, a.agent_id)] = round(min(max(p, 0.01), 0.99), 4)
    agents = {a.agent_id: hx.Agent(a.agent_id, a.family, a.operator, a.pipeline, a.price)
              for a in case.agents}
    return hx.Dataset(questions, agents, forecasts)


def _copy(i: int) -> SimAgent:
    return SimAgent(f"gpt_{i}", "gpt", "openai", "web_search_a", 1.0, 1.6, ("gpt",), 0.1)


CASES: Tuple[Case, ...] = (
    Case(
        name="redundancy",
        title="Redundancy",
        question="Do six near-identical copies of one model improve the result at all?",
        sources={"gpt": 1.4, "claude": 1.4, "gemini": 1.4},
        agents=tuple(_copy(i) for i in range(1, 7)) + (
            SimAgent("claude_1", "claude", "anthropic", "web_search_b", 1.2, 1.5, ("claude",), 0.3),
            SimAgent("gemini_1", "gemini", "google", "web_search_c", 0.9, 1.4, ("gemini",), 0.3),
        ),
        ladder=tuple(f"gpt_{i}" for i in range(1, 7)),
    ),
    Case(
        name="complementarity",
        title="Complementarity",
        question="Can a weaker, cheaper model improve the group because its mistakes are different?",
        sources={"shared_search": 2.0},
        agents=(
            SimAgent("gpt_1", "gpt", "openai", "web_search_a", 1.0, 2.0, ("shared_search",), 0.4),
            SimAgent("claude_1", "claude", "anthropic", "web_search_a", 1.2, 2.0, ("shared_search",), 0.4),
            SimAgent("gemini_1", "gemini", "google", "web_search_a", 0.9, 2.0, ("shared_search",), 0.4),
            # Weaker on its own, but it does not use the shared search source.
            SimAgent("local_1", "llama", "self_hosted", "offline_corpus", 0.3, 0.8, (), 1.0),
        ),
    ),
    Case(
        name="illusory_diversity",
        title="Illusory diversity",
        question="What happens when a model declared as different makes the same systematic errors?",
        sources={"gpt": 1.5, "claude": 1.5},
        agents=(
            SimAgent("gpt_1", "gpt", "openai", "web_search_a", 1.0, 1.6, ("gpt",), 0.3),
            SimAgent("claude_1", "claude", "anthropic", "web_search_b", 1.2, 1.6, ("claude",), 0.3),
            # Declared as an unrelated family, operator and pipeline, but it
            # repeats gpt's hidden error source, for example via the same data.
            SimAgent("mistral_1", "mistral", "mistral_ai", "own_search", 0.8, 1.6, ("gpt",), 0.15),
            SimAgent("llama_1", "llama", "self_hosted", "offline_corpus", 0.3, 1.2, (), 1.0),
        ),
    ),
    Case(
        name="counterproductive_diversity",
        title="Counterproductive diversity",
        question="Can the method spot a model whose answers differ but make the group worse?",
        sources={"gpt": 1.2, "claude": 1.2, "gemini": 1.2},
        agents=(
            SimAgent("gpt_1", "gpt", "openai", "web_search_a", 1.0, 1.8, ("gpt",), 0.3),
            SimAgent("claude_1", "claude", "anthropic", "web_search_b", 1.2, 1.8, ("claude",), 0.3),
            SimAgent("gemini_1", "gemini", "google", "web_search_c", 0.9, 1.8, ("gemini",), 0.3),
            # Independent of everyone, so it looks maximally diverse, but it
            # barely tracks the truth.
            SimAgent("contrarian_1", "other", "other_lab", "other_search", 0.5, 0.2, (), 2.0),
        ),
    ),
)


@dataclasses.dataclass
class CaseResult:
    case: Case
    n_test: int
    pool_brier: float
    table: List[Contribution]
    ladder: List[Tuple[int, float]]
    selections: List[Tuple[str, List[str], float]]  # (name, members, test Brier)


def run_case(case: Case, n_questions: int = 200) -> CaseResult:
    data = simulate(case, n_questions)
    pool = [a.agent_id for a in case.agents]
    test = data.question_ids("test")
    ladder = [(n, hx.group_brier(data, list(case.ladder[:n]), test))
              for n in range(1, len(case.ladder) + 1)]
    selections = []
    for name, ids in (("hap", hx.select_hap(data, case.k)),
                      ("contribution", hx.select_contribution(data, case.k))):
        selections.append((name, ids, hx.group_brier(data, ids, test)))
    return CaseResult(case, len(test), hx.group_brier(data, pool, test),
                      with_without(data, pool), ladder, selections)


def format_case(r: CaseResult) -> str:
    c = r.case
    lines = [
        f"## {c.title}",
        "",
        f"*{c.question}*",
        "",
        f"All {len(c.agents)} agents averaged: Brier {r.pool_brier:.3f} on {r.n_test} test questions.",
        "",
        "| Agent | Declared family | Price | Solo Brier | Contribution | 95% interval | Verdict "
        "| Error correlation with others | Most similar errors | Declared overlap with others |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in r.table:
        lines.append(
            f"| {t.agent} | {t.family} | {t.price:.1f} | {t.solo_brier:.3f} | {t.contribution:+.4f} "
            f"| {t.low:+.4f} to {t.high:+.4f} | {t.verdict} | {t.error_correlation:.2f} "
            f"| {t.most_similar} ({t.most_similar_correlation:.2f}) | {t.provenance_overlap:.2f} |"
        )
    if r.ladder:
        lines += ["", "Averaging the first n copies only:", "",
                  "| Copies | Brier |", "| --- | --- |"]
        lines += [f"| {n} | {b:.3f} |" for n, b in r.ladder]
    lines += ["", f"Groups of up to {c.k} chosen on training questions, scored on test questions:", ""]
    lines += [f"- {hx.NAMES[name]}: {', '.join(ids)} (Brier {b:.3f})" for name, ids, b in r.selections]
    return "\n".join(lines)


def cases_report(results: Sequence[CaseResult]) -> str:
    return "\n".join([
        "# What does each model add? Four synthetic cases",
        "",
        "Each case is a synthetic fixture built to show one pattern, with 100 training and 100 "
        "test questions. Its settings were chosen so the pattern is visible. It shows that the "
        "measurements detect the pattern, not that real AI systems behave this way.",
        "",
        "**Contribution** is how much worse the group's averaged forecast gets (in Brier score) "
        "when that agent is removed. Positive means the agent helps, negative means the group "
        "is better without it. The 95% interval comes from resampling the test questions; "
        "**helps** or **hurts** is shown only when the whole interval is on one side of zero. "
        "A contribution belongs to this group: the same agent can matter more or less elsewhere.",
        "",
        "**Error correlation** is measured from forecasts: how closely an agent's misses track the "
        "others' on the same questions (about 0 means unrelated mistakes). **Most similar errors** "
        "names the member it tracks most closely. **Declared overlap** is the share of model "
        "family, operator and pipeline an agent claims to share with the others.",
        "",
    ] + ["\n\n".join(format_case(r) for r in results)])


# ---------------------------------------------------------------------------
# Buyer test
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Requirement:
    max_brier: float = 0.10
    max_confident_wrong: float = 0.25  # rate; with 12 questions, 0 of 12 bounds at 22%
    budget: float = 3.0                # credits per completed forecast


@dataclasses.dataclass
class Candidate:
    label: str
    agents: Tuple[str, ...]
    cost: float
    brier: float
    brier_upper: float
    confident_wrong: int
    confident_wrong_upper: float
    n: int

    def meets(self, req: Requirement) -> bool:
        return (self.cost <= req.budget and self.brier_upper <= req.max_brier
                and self.confident_wrong_upper <= req.max_confident_wrong)


def score_candidate(data: hx.Dataset, label: str, agents: Sequence[str], qids: Sequence[str],
                    draws: Sequence[Sequence[int]]) -> Candidate:
    briers = per_question_brier(data, agents, qids)
    cw = sum(
        hx.is_confidently_wrong(sum(data.forecast(q, a) for a in agents) / len(agents),
                                data.questions[q].outcome)
        for q in qids
    )
    return Candidate(
        label=label,
        agents=tuple(agents),
        cost=sum(data.agents[a].price for a in agents),
        brier=sum(briers) / len(briers),
        brier_upper=percentile(bootstrap_means(briers, draws), 0.95),
        confident_wrong=cw,
        confident_wrong_upper=rate_upper_bound(cw, len(qids)),
        n=len(qids),
    )


@dataclasses.dataclass
class BuyerResult:
    requirement: Requirement
    candidates: List[Candidate]  # training scores
    chosen: Optional[Candidate]
    most_accurate: Candidate
    chosen_test: Optional[Candidate]
    accurate_test: Candidate


def buyer_test(data: hx.Dataset, req: Requirement, max_size: int = 3, k: int = 4) -> BuyerResult:
    train, test = data.question_ids("train"), data.question_ids("test")
    train_draws = bootstrap_indices(len(train))
    test_draws = bootstrap_indices(len(test), seed=BOOTSTRAP_SEED + 1)
    groups: Dict[Tuple[str, ...], str] = {}
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(sorted(data.agents), size):
            groups[combo] = " + ".join(combo)
    for name, select in (("same_model", hx.select_same_model), ("naive_mix", hx.select_naive_mix),
                         ("hap", hx.select_hap), ("contribution", hx.select_contribution)):
        try:
            ids = select(data, k)
        except ValueError:  # e.g. no family has k agents; skip that benchmark group
            continue
        key = tuple(sorted(ids))
        if key in groups:
            groups[key] += f" (= {hx.NAMES[name]})"
        else:
            groups[key] = f"{hx.NAMES[name]}: {' + '.join(key)}"
    cands = [score_candidate(data, label, ids, train, train_draws) for ids, label in groups.items()]
    passing = [c for c in cands if c.meets(req)]
    chosen = min(passing, key=lambda c: (c.cost, c.brier_upper, c.label)) if passing else None
    accurate = min(cands, key=lambda c: (c.brier, c.cost, c.label))

    def on_test(c: Candidate) -> Candidate:
        return score_candidate(data, c.label, c.agents, test, test_draws)

    return BuyerResult(req, cands, chosen, accurate,
                       on_test(chosen) if chosen else None, on_test(accurate))


def format_buyer(r: BuyerResult) -> str:
    req = r.requirement

    def row(c: Candidate) -> str:
        ok = "yes" if c.meets(req) else "no"
        return (f"| {c.label} | {c.cost:.1f} | {c.brier:.3f} | {c.brier_upper:.3f} "
                f"| {c.confident_wrong} of {c.n} | {c.confident_wrong_upper:.0%} | {ok} |")

    header = ["| Option | Cost | Brier | Brier, pessimistic | Confidently wrong "
              "| Confidently wrong, pessimistic | Meets requirement |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    singles = sorted((c for c in r.candidates if len(c.agents) == 1), key=lambda c: (c.cost, c.label))
    named = [c for c in r.candidates if any(n in c.label for n in hx.NAMES.values())]
    n_train = r.candidates[0].n
    lines = [
        "# Buyer test: the cheapest option that meets the requirement",
        "",
        f"Requirement: Brier at most {req.max_brier:.2f}, confidently wrong at most "
        f"{req.max_confident_wrong:.0%} of questions, cost at most {req.budget:.1f} credits per forecast. "
        "Costs are the fixture's illustrative prices, summed over the members who must answer.",
        "",
        f"Options considered: every single agent, every group of up to three, and the four benchmark "
        f"groups ({len(r.candidates)} in all). Each is judged on {n_train} training questions by the "
        "pessimistic end of its interval: the 95th percentile of a bootstrap for Brier, and a 95% "
        "upper bound for the confidently-wrong rate. The cheapest option that passes is chosen, "
        "then checked on the untouched test questions.",
        "",
    ]
    if r.chosen:
        lines.append(f"**Chosen: {r.chosen.label}**, cost {r.chosen.cost:.1f}.")
    else:
        lines.append("**No option meets the requirement on the training questions.** "
                     "With this few questions the pessimistic bounds are wide; more questions, "
                     "a looser requirement or a bigger budget would be needed.")
    lines.append(f"Most accurate option on training questions: {r.most_accurate.label}, "
                 f"cost {r.most_accurate.cost:.1f}.")
    lines += ["", "## On the training questions (used to choose)", ""] + header
    shown = ([r.chosen] if r.chosen else []) + [r.most_accurate] + singles + named
    seen = set()
    for c in shown:
        if c.agents not in seen:
            seen.add(c.agents)
            lines.append(row(c))
    lines += ["", "## On the test questions (check only)", ""] + header
    tested = ([r.chosen_test] if r.chosen_test else []) + [r.accurate_test]
    seen = set()
    for c in tested:
        if c.agents not in seen:
            seen.add(c.agents)
            lines.append(row(c))
    lines += [
        "",
        f"The pessimistic bound protects each option on its own, not the best of "
        f"{len(r.candidates)}: the more options are compared on the same few questions, the more "
        "likely the winner was partly lucky. The test check is there to catch that.",
        "",
        f"With {r.accurate_test.n} test questions these checks are rough. Synthetic data and "
        "made-up prices show how the decision works, not which real model to buy.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "questions.csv")
    ap.add_argument("--predictions", type=Path, default=ROOT / "data" / "smoke_predictions.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "smoke")
    ap.add_argument("--max-brier", type=float, default=Requirement.max_brier)
    ap.add_argument("--max-confident-wrong", type=float, default=Requirement.max_confident_wrong)
    ap.add_argument("--budget", type=float, default=Requirement.budget)
    args = ap.parse_args(argv)

    try:
        data = hx.load_dataset(args.questions, args.predictions)
    except (OSError, ValueError) as exc:
        ap.exit(2, f"error: {exc}\n")
    args.out.mkdir(parents=True, exist_ok=True)

    cases = cases_report([run_case(c) for c in CASES])
    (args.out / "contribution_cases.md").write_text(cases + "\n")
    req = Requirement(args.max_brier, args.max_confident_wrong, args.budget)
    buyer = format_buyer(buyer_test(data, req))
    (args.out / "buyer_test.md").write_text(buyer + "\n")
    print(cases + "\n\n" + buyer)


if __name__ == "__main__":
    main()
