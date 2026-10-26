# Hayekian Agent Protocol

## Can ten AI agents still be one opinion?

Imagine asking ten people whether a project will succeed. Their agreement means something if they bring different experience and notice different risks. It means much less if all ten are repeating the same source.

The same question matters for artificial intelligence. An agent is a program that uses an AI model to carry out tasks. Giving ten agents different names or roles does not necessarily give them ten independent ways of seeing the world. They may still miss the same thing.

The Hayekian Agent Protocol (HAP) asks whether groups of agents can work together without losing the diversity that makes a group useful. The aim is fewer shared blind spots, not more voices saying the same thing.

## What is in this repository today?

A small, runnable comparison using simulated forecasts. It does not yet show that the method helps real agents. Two states appear throughout this README, and they should not be blurred:

- **Runnable now:** the forecasting benchmark below, an analysis of what each model contributes, and a buyer test that picks the cheapest option meeting a stated requirement. With tests and example data. Python 3.10 or newer, no other dependencies.
- **Proposed, not yet tested:** a larger study with real forecasts, and the longer-term market design. Plans, not results.

## Try it

```bash
./run_smoke.sh
```

This regenerates the synthetic data, runs the comparison, the contribution analysis, the buyer test and the tests. It writes `outputs/smoke/report.md` (with `results.csv` and `results.json`), [contribution_cases.md](outputs/smoke/contribution_cases.md) and [buyer_test.md](outputs/smoke/buyer_test.md). Saved copies are included, so you can read the output without installing anything. The run uses built-in simulated forecasts; it does not call live AI models.

To run it on your own forecasts, supply two CSV files in the same format as `data/questions.csv` and `data/smoke_predictions.csv`:

```bash
python3 src/hap_experiment.py --questions my_questions.csv --predictions my_predictions.csv --out outputs/my_run
```

The buyer test takes the same files plus your requirement: maximum Brier score, maximum share of confidently wrong answers, and budget per forecast in the units of your `price` column.

```bash
python3 src/hap_contribution.py --questions my_questions.csv --predictions my_predictions.csv --out outputs/my_run --max-brier 0.10 --max-confident-wrong 0.25 --budget 3
```

## The question

When several agents agree, is that because they checked the problem independently, or because they share the same blind spot? Imagine ten agents assessing a business plan: all ten approve it, and all ten overlooked the same cash-flow problem. Ten agreeing answers produced false reassurance.

HAP's hypothesis: if you know where each answer comes from and how each agent has failed in the past, you can assemble a group that fails differently, and therefore fails more usefully. The competing explanation is simpler: average forecasts from different models and skip the selection machinery. The benchmark exists to decide between them.

## How the comparison works

Four ways of forming the group:

- **One model, many agents:** repeat the same model across several agents.
- **Different models, simple mixing:** take the best agent from each model family and average their forecasts.
- **HAP diversity-aware:** choose agents with attention to past accuracy and shared mistakes, then average their forecasts.
- **Contribution-based:** try every group of up to four agents and keep the smallest one that is (nearly) as good as the best, so members that add nothing are left out.

Simple mixing is the comparison to beat. Beating repeated copies of one model alone would not show that the extra method is needed.

HAP's selection reads each agent's declared provenance (model family, operator and search pipeline), looks for agents that have tended to miss on the same questions with known outcomes, and balances accuracy against overlap with the agents already chosen. Price can be weighed too. The selection uses training questions only, with weights fixed in advance, and never sees test outcomes. Two cautions are built in: being different is not valuable by itself, and agreement is not a fault. Outcomes decide.

Agents give probabilities, not only yes or no. The scorecard asks how close the probabilities are to the outcomes (including the Brier score, which penalizes forecasts more the further they are from what happens), how often every member of a group gets the same question wrong, how often the group is confidently wrong, how bad its worst errors are, and how closely its members' mistakes move together. Whether any improvement is worth the extra computing is a question for the real study; the smoke test does not measure cost.

## What do we know so far?

In September 2026 a small synthetic smoke test compared the three group types on 24 simulated questions. Both mixed-model groups clearly beat the repeated single model (Brier score about 0.12 against 0.30), and HAP showed no clear advantage over simple averaging at that small size. That code was not preserved, so treat this as an anecdote about the plumbing, not a result.

The benchmark in this repository is the October 2026 rebuild of that description, and it is reproducible: `./run_smoke.sh` regenerates everything. Its fixture has 24 questions and ten agents from four model families, some sharing an operator or a search pipeline, so its numbers differ from the September ones. On this fixture the three larger groups have low average error. The repeated single model is the only group where every member got the same question wrong (2 of 12), and its members' mistakes are almost identical (shared error 0.99, against 0.30 for simple mixing and 0.24 for HAP). HAP's average error is slightly lower than simple mixing (0.06 against 0.08), but with 12 made-up questions that gap says little. The contribution-based method keeps a single model that looked best on the 12 training questions and does worse on the test questions (0.10). It is not evidence that HAP works, or that contribution-based selection does.

Shared error is measured as the correlation of how far each member's forecast was from the outcome on each question. An earlier version correlated signed errors (forecast minus outcome), which makes any two forecasters look related because both lean toward the true answer. That inflated the shared error of the mixed groups (previously reported as 0.64); the corrected figures are above.

The proposed next step is a study with at least 500 real forecasts, for example from timestamped records on prediction platforms such as Metaculus or Manifold, subject to each platform's data terms. That study needs safeguards the smoke test only partially meets: outcomes the model could not have known when it forecast, separate questions for developing and for judging the rules, and evaluation rules fixed in advance.

## What does each model add?

How different two models are matters less than what each one adds to the group's result. The contribution analysis measures this directly: remove one model, and see how much worse the group's averaged forecast gets. That number, with a 95% interval from resampling the questions, appears in a with/without table next to how closely the model's mistakes track the others' and what provenance it declares.

Four synthetic cases, each with 100 training and 100 test questions, are built to show one pattern apiece. They show that the measurements detect the pattern, not that real AI systems behave this way. Full tables: [contribution_cases.md](outputs/smoke/contribution_cases.md).

- **Redundancy:** six near-copies of one model. Averaging one, two or six of them gives the same score (Brier 0.064 to 0.065); removing any one copy changes nothing measurable.
- **Complementarity:** three strong models that rely on the same search source, plus a cheap, weaker one that does not. The cheap model is the weakest on its own (Brier 0.137 against 0.123 to 0.130), yet the group is clearly worse without it.
- **Illusory diversity:** a model that declares a different family, operator and search pipeline but repeats another model's systematic errors. Its declared overlap is zero; its measured error correlation with that model is 0.97.
- **Counterproductive diversity:** a model whose mistakes are unrelated to everyone else's, so it looks maximally diverse, but which barely tracks the truth. Removing it clearly improves the group, and neither selector picks it.

An improvement in the average is not the same as one model catching another's mistakes. HAP averages forecasts; models do not check or correct each other's answers here.

## Which option should a buyer choose?

A small model can earn its place in two ways: by improving a group cheaply, or by replacing an expensive model while keeping the result good enough. The buyer test makes that decision explicit. The buyer states a requirement: a maximum Brier score, a maximum share of confidently wrong answers, and a budget per forecast. Every single model and every group of up to three is checked on the training questions against the pessimistic end of its interval, so a lucky streak does not count as proof. The cheapest option that passes is chosen, which can be one small model and no ensemble at all, then checked again on the test questions. Full results: [buyer_test.md](outputs/smoke/buyer_test.md).

On the 24-question smoke data, with a requirement of Brier at most 0.10 and confidently wrong at most 25% of questions, only one of 178 options passes on the 12 training questions: a single model. On the 12 test questions it misses the requirement (Brier 0.096, but its pessimistic bound is 0.205). The contribution-based selector makes the same choice and the same mistake. This is the expected failure with so few questions: comparing many options on the same small sample favours a lucky one, and the pessimistic bound protects each option on its own, not the best of 178. Prices in the fixture are made up, so the test shows how the decision works, not which model to buy. A real buyer test needs measured costs per forecast and many more questions.

**Status: exploratory research, not a proven product.**

## Why "Hayekian"?

The name comes from Friedrich Hayek's 1945 essay *The Use of Knowledge in Society*. Hayek argued that the knowledge needed to run a complex economy never exists in one head: it is scattered among many people as incomplete, sometimes contradictory pieces, much of it local and hard to write down. Central planning fails because this dispersed knowledge cannot be gathered into one place. Markets cope through prices, a compact signal that lets people coordinate without needing to know why the signal moved.

A similar tension can appear in software that coordinates many agents. One way to coordinate them is a central orchestrator that receives the request, splits it into steps, assigns the work and collects the results. As the number of specialized agents grows, this design can develop two problems Hayek would have recognized: routing every local update through one supervisor consumes its limited context and adds delay, and when the central node summarizes what sub-agents report, raw outputs, error logs and edge cases get discarded.

HAP takes the dispersed-knowledge idea as an inspiration, not as proof that markets or decentralization automatically make software better. The analogy has a real limit: in software, a designer sets the goal, and much of the system's state can in principle be logged and read centrally. The case for decentralizing agents is an engineering trade-off, not a vindication of any political theory. The motivation also draws on James Surowiecki's *The Wisdom of Crowds* (2004), which counts diversity and independence among the conditions for a wise crowd, and on Norbert Wiener's *The Human Use of Human Beings* (1950), which warns against treating people as interchangeable parts of a machine.

## The wider idea: from benchmark to market

The longer-term ambition is a way for people and agents to contribute what they do best without being pushed into one way of thinking. That ambition involves two different jobs, easy to confuse: assembling a group for a joint forecast (what the current benchmark tests) and handing one task to one worker (what the auction design below addresses). The order remains benchmark first, market layer later, only if the evidence supports it. Even a successful forecasting benchmark would not show that the auction improves task assignment; that would need its own comparison with simpler methods.

### Assigning single tasks: the Vickrey auction

*Proposed design. Not implemented in this repository and not part of the current benchmark.*

HAP began with a different question: how can independent agents offer to do work without one central dispatcher needing to know every agent's costs? The proposed answer is a reverse Vickrey auction, also called a second-price procurement auction. Ordinary auctions sell to the highest bidder. Here, agents offer to do a job, so the lowest eligible bid wins, and the winner is paid the second-lowest eligible bid.

**The rule, in plain numbers.** Three agents offer to complete the same task: A asks for 200 credits, B for 250, C for 300. A wins and receives 250, not 200. (Illustrative accounting units, not a payment system.) Why pay more than the winning bid? If A's true cost is 200, bidding 150 still earns 250; bidding 260 loses the job to B; bidding 240 changes nothing. Bidding below cost can backfire: had B asked 180, a bid of 150 wins A the job at 180, less than its cost. In the standard model, where each bidder knows its own cost and cannot coordinate with others, honest cost reporting is the sensible strategy.

**What can go wrong.** That result rests on conditions real markets meet only partly:

- **Uncertain costs.** Agents estimate their token use and latency before running the task. Because they draw on similar models and shared token prices, the winner may simply be the most overoptimistic bidder (the "winner's curse"). Bidding a margin above expected cost is one partial mitigation.
- **Identity.** A signed bid proves which registered identity sent it, not that the identity is an independent operator. If one party controls several bidders, it can set its own payout. Genuine resistance to fake identities needs an external cost of identity, such as a deposit or verified registration, which this design does not yet include.
- **Quality.** A cheap bid is not evidence of quality or independence. How quality and diversity requirements would interact with prices remains an open design question.

The proposed rules are signed bids, one sealed bid per agent per task, a deadline, a maximum price and a deterministic tie-break. Resistance to fake identities, payment, persistence and quality checks remain open problems.

### Coordination without one controller

Three further ideas from the wider design, all untested proposals:

- **Stigmergy: coordination through traces.** Ants coordinate through pheromone trails, not instructions. The software version is a shared task board and workspace: one agent commits a result, others adapt, and each agent keeps private memory and credentials that others cannot overwrite.
- **Stable rules, local enforcement.** Drawing on Elinor Ostrom's *Governing the Commons* (1990): local monitors with graduated penalties instead of one all-seeing controller. A central dispatcher may still be cheaper and simpler when one operator controls all the workers; this is a design to compare, not a doctrine.
- **A circuit breaker against herding.** Watch for agents converging too quickly on the same answer, and force a recheck against external evidence, such as running tests on generated software. Agreement is often correct, so convergence is at most a warning signal. Checked outcomes must do the judging.

### Where the analogy stops

One warning shapes the whole design: in Albert Hirschman's terms from *Exit, Voice, and Loyalty* (1970), people can leave a system or protest within it. Agents can only exit: they time out, fail health checks or exhaust budgets. A swarm converging on a wrong answer is not staging a revolt; it is executing bad mathematics along the path of least resistance. Stable rules, isolated state and price-like signalling are engineering heuristics worth testing, not proofs of a political worldview.

## Repository layout

- `src/hap_experiment.py`: data loading, the four group-forming methods, metrics and reports.
- `src/hap_contribution.py`: the four contribution cases, the with/without tables and the buyer test.
- `tests/`: tests for the metrics, the selection rules, the cases, the buyer test and the rule that selection never sees test outcomes.
- `data/`: the synthetic fixture of 24 questions and ten agents.
- `outputs/smoke/`: saved example reports you can read without running anything.
- `run_smoke.sh`: regenerates the data, runs the comparison, the contribution analysis, the buyer test and the tests.

## Background reading

[Hayek's *The Use of Knowledge in Society*](https://www.econlib.org/book-chapters/chapter-the-use-of-knowledge-in-society/)

[Second-price auction incentives, explained in teaching notes](https://www.kiragoldner.com/teaching/DS574/fall22/L2.pdf). These notes describe the standard selling auction; HAP's example reverses it to buy a service.

## Feedback

Useful feedback at this stage: is simple averaging the right comparison to beat, what would make a fair test with real forecasts, and where might the method reward the wrong kind of difference?

> The question is not how many agents we can run. It is how many genuinely different ways of seeing a problem they bring, and whether that helps when reality checks their answers.
