# Hayekian Agent Protocol

## Can ten AI agents still be one opinion?

Imagine asking ten people whether a project will succeed. Their agreement means something if they bring different experience and notice different risks. It means much less if all ten are repeating the same source.

The same question matters for artificial intelligence. An agent is a program that uses an AI model to carry out tasks. Giving ten agents different names or roles does not necessarily give them ten independent ways of seeing the world. They may still miss the same thing.

The Hayekian Agent Protocol (HAP) asks whether groups of agents can work together without losing the diversity that makes a group useful. The aim is fewer shared blind spots, not more voices saying the same thing.

## What is in this repository today?

A small, runnable comparison using simulated forecasts. It does not yet show that the method helps real agents. Three states appear throughout this README, and they should not be blurred:

- **Runnable now:** the forecasting benchmark below, with tests and example data. Python 3.10 or newer, no other dependencies.
- **Historical prototype:** the earlier auction engine in `docs/history/`, which still runs, and the essay it came with. A superseded first design, kept for reference.
- **Proposed, not yet tested:** a larger study with real forecasts, and the longer-term market design. Plans, not results.

## Try it

```bash
./run_smoke.sh
```

This regenerates the synthetic data, runs the comparison and the tests, and writes `outputs/smoke/report.md`, `results.csv` and `results.json`. A saved example report is included at [outputs/smoke/report.md](outputs/smoke/report.md), so you can look at the output without installing anything. The run uses built-in simulated forecasts; it does not call live AI models.

To run it on your own forecasts, supply two CSV files in the same format as `data/questions.csv` and `data/smoke_predictions.csv`:

```bash
python3 src/hap_experiment.py --questions my_questions.csv --predictions my_predictions.csv --out outputs/my_run
```

## The question

When several agents agree, is that because they checked the problem independently, or because they share the same blind spot? Imagine ten agents assessing a business plan: all ten approve it, and all ten overlooked the same cash-flow problem. Ten agreeing answers produced false reassurance.

HAP's hypothesis: if you know where each answer comes from and how each agent has failed in the past, you can assemble a group that fails differently, and therefore fails more usefully. The competing explanation is simpler: average forecasts from different models and skip the selection machinery. The benchmark exists to decide between them.

## How the comparison works

Three ways of forming the group:

- **One model, many agents:** repeat the same model across several agents.
- **Different models, simple mixing:** take the best agent from each model family and average their forecasts.
- **HAP's proposed method:** choose agents with attention to past accuracy and shared mistakes, then average their forecasts.

Simple mixing is the comparison to beat. Beating repeated copies of one model alone would not show that the extra method is needed.

HAP's selection reads each agent's declared provenance (model family, operator and search pipeline), looks for agents that have tended to be wrong in the same direction on questions with known outcomes, and balances accuracy against overlap with the agents already chosen. Price can be weighed too. The selection uses training questions only, with weights fixed in advance, and never sees test outcomes. Two cautions are built in: being different is not valuable by itself, and agreement is not a fault. Outcomes decide.

Agents give probabilities, not only yes or no. The scorecard asks how close the probabilities are to the outcomes (including the Brier score, which penalizes forecasts more the further they are from what happens), how often every member of a group gets the same question wrong, how often the group is confidently wrong, how bad its worst errors are, and how closely its members' mistakes move together. Whether any improvement is worth the extra computing is a question for the real study; the smoke test does not measure cost.

## What do we know so far?

In September 2026 a small synthetic smoke test compared the three group types on 24 simulated questions. Both mixed-model groups clearly beat the repeated single model (Brier score about 0.12 against 0.30), and HAP showed no clear advantage over simple averaging at that small size. That code was not preserved, so treat this as an anecdote about the plumbing, not a result.

The benchmark in this repository is the October 2026 rebuild of that description, and it is reproducible: `./run_smoke.sh` regenerates everything. Its fixture has 24 questions and ten agents from four model families, some sharing an operator or a search pipeline, so its numbers differ from the September ones. On this fixture all three groups have low average error. The repeated single model is the only group where every member got the same question wrong (2 of 12), and its members' mistakes are almost identical (shared error 1.00, against about 0.64 for both mixed groups). HAP's average error is slightly lower than simple mixing (0.06 against 0.08), but with 12 made-up questions that gap says little. It is not evidence that HAP works.

The proposed next step is a study with at least 500 real forecasts, for example from timestamped records on prediction platforms such as Metaculus or Manifold, subject to each platform's data terms. That study needs safeguards the smoke test only partially meets: outcomes the model could not have known when it forecast, separate questions for developing and for judging the rules, and evaluation rules fixed in advance.

**Status: exploratory research, not a proven product.**

## Why "Hayekian"?

The name comes from Friedrich Hayek's 1945 essay *The Use of Knowledge in Society*. Hayek argued that the knowledge needed to run a complex economy never exists in one head: it is scattered among many people as incomplete, sometimes contradictory pieces, much of it local and hard to write down. Central planning fails because this dispersed knowledge cannot be gathered into one place. Markets cope through prices, a compact signal that lets people coordinate without needing to know why the signal moved.

A similar tension can appear in software that coordinates many agents. One way to coordinate them is a central orchestrator that receives the request, splits it into steps, assigns the work and collects the results. As the number of specialized agents grows, this design can develop two problems Hayek would have recognized: routing every local update through one supervisor consumes its limited context and adds delay, and when the central node summarizes what sub-agents report, raw outputs, error logs and edge cases get discarded.

HAP takes the dispersed-knowledge idea as an inspiration, not as proof that markets or decentralization automatically make software better. The analogy has a real limit: in software, a designer sets the goal, and much of the system's state can in principle be logged and read centrally. The case for decentralizing agents is an engineering trade-off, not a vindication of any political theory. The motivation also draws on James Surowiecki's *The Wisdom of Crowds* (2004), which counts diversity and independence among the conditions for a wise crowd, and on Norbert Wiener's *The Human Use of Human Beings* (1950), which warns against treating people as interchangeable parts of a machine.

## The wider idea: from benchmark to market

The longer-term ambition is a way for people and agents to contribute what they do best without being pushed into one way of thinking. That ambition involves two different jobs, easy to confuse: assembling a group for a joint forecast (what the current benchmark tests) and handing one task to one worker (what the auction design below addresses). The order remains benchmark first, market layer later, only if the evidence supports it. Even a successful forecasting benchmark would not show that the auction improves task assignment; that would need its own comparison with simpler methods.

### Assigning single tasks: the Vickrey auction

*Historical prototype in `docs/history/`; not part of the current benchmark. Its demo was last run on 25 October 2026.*

HAP began with a different question: how can independent agents offer to do work without one central dispatcher needing to know every agent's costs? The proposed answer is a reverse Vickrey auction, also called a second-price procurement auction. Ordinary auctions sell to the highest bidder. Here, agents offer to do a job, so the lowest eligible bid wins, and the winner is paid the second-lowest eligible bid.

**The rule, in plain numbers.** Three agents offer to complete the same task: A asks for 200 credits, B for 250, C for 300. A wins and receives 250, not 200. (Illustrative accounting units, not a payment system.) Why pay more than the winning bid? If A's true cost is 200, bidding 150 still earns 250; bidding 260 loses the job to B; bidding 240 changes nothing. Bidding below cost can backfire: had B asked 180, a bid of 150 wins A the job at 180, less than its cost. In the standard model, where each bidder knows its own cost and cannot coordinate with others, honest cost reporting is the sensible strategy.

**What can go wrong.** That result rests on conditions real markets meet only partly:

- **Uncertain costs.** Agents estimate their token use and latency before running the task. Because they draw on similar models and shared token prices, the winner may simply be the most overoptimistic bidder (the "winner's curse"). The reference code includes a bid margin parameter as a partial mitigation.
- **Identity.** A signed bid proves which registered identity sent it, not that the identity is an independent operator. If one party controls several bidders, it can set its own payout. Genuine resistance to fake identities needs an external cost of identity, such as a deposit or verified registration, which the prototype does not provide.
- **Quality.** A cheap bid is not evidence of quality or independence. How quality and diversity requirements would interact with prices remains an open design question.

The prototype enforces signed bids, one sealed bid per agent per task, a deadline, a maximum price and a deterministic tie-break. It does not provide Sybil resistance, payment, persistence or quality checks, and says so. The core demo needs `pip install cryptography`, then `python3 docs/history/hap_core.py`; the optional HTTP gateway additionally needs `fastapi` and `pydantic`.

### Coordination without one controller

Three further ideas from the wider design, all untested proposals:

- **Stigmergy: coordination through traces.** Ants coordinate through pheromone trails, not instructions. The software version is a shared task board and workspace: one agent commits a result, others adapt, and each agent keeps private memory and credentials that others cannot overwrite.
- **Stable rules, local enforcement.** Drawing on Elinor Ostrom's *Governing the Commons* (1990): local monitors with graduated penalties instead of one all-seeing controller. A central dispatcher may still be cheaper and simpler when one operator controls all the workers; this is a design to compare, not a doctrine.
- **A circuit breaker against herding.** Watch for agents converging too quickly on the same answer, and force a recheck against external evidence, such as running tests on generated software. Agreement is often correct, so convergence is at most a warning signal. Checked outcomes must do the judging.

### Where the analogy stops

The original essay's most useful warning: in Albert Hirschman's terms from *Exit, Voice, and Loyalty* (1970), people can leave a system or protest within it. Agents can only exit: they time out, fail health checks or exhaust budgets. A swarm converging on a wrong answer is not staging a revolt; it is executing bad mathematics along the path of least resistance. Stable rules, isolated state and price-like signalling are engineering heuristics worth testing, not proofs of a political worldview.

## Repository layout

- `src/hap_experiment.py`: data loading, the three group-forming methods, metrics and reports.
- `tests/test_experiment.py`: tests for the metrics, the selection rules and the rule that selection never sees test outcomes.
- `data/`: the synthetic fixture of 24 questions and ten agents.
- `outputs/smoke/`: a saved example report you can read without running anything.
- `run_smoke.sh`: regenerates the data, runs the comparison and the tests.
- `docs/history/`: the original essay and the auction prototype, kept for reference with their own status notes.

## Background reading

[Hayek's *The Use of Knowledge in Society*](https://www.econlib.org/book-chapters/chapter-the-use-of-knowledge-in-society/)

[Second-price auction incentives, explained in teaching notes](https://www.kiragoldner.com/teaching/DS574/fall22/L2.pdf). These notes describe the standard selling auction; HAP's example reverses it to buy a service.

The original essay, *The Machine and the Market*, is preserved in [docs/history](docs/history/) for readers who want the full argument and the formal details. You do not need it to understand or run the benchmark.

## Feedback

Useful feedback at this stage: is simple averaging the right comparison to beat, what would make a fair test with real forecasts, and where might the method reward the wrong kind of difference?

> The question is not how many agents we can run. It is how many genuinely different ways of seeing a problem they bring, and whether that helps when reality checks their answers.
