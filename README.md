# Hayekian Agent Protocol

## Can ten AI agents still be one opinion?

Imagine asking ten people whether a project will succeed. Their agreement means something if they bring different experience and notice different risks. It means much less if all ten are repeating the same source.

The same question matters for artificial intelligence. An agent is a program that uses an AI model to carry out tasks. Giving ten agents different names or roles does not necessarily give them ten independent ways of seeing the world. They may still miss the same thing.

The Hayekian Agent Protocol (HAP) asks whether we can help groups of agents work together without losing the diversity that makes a group useful. The aim is fewer shared blind spots, rather than more voices saying the same thing.

## Why "Hayekian"?

The name comes from Friedrich Hayek's 1945 essay *The Use of Knowledge in Society*. Useful knowledge is scattered among people. No single person holds the whole picture. Coordination should make use of those different pieces of knowledge.

HAP takes that idea as an inspiration, not as proof that markets or decentralization automatically make software better.

## How would HAP work?

The core idea is to treat a group of agents as a team whose members can make different mistakes, rather than counting every answer as an independent vote.

The following describes the proposed approach. It is not a claim that a finished protocol already delivers all these features. The exact selection and weighting rules still need testing.

### 1. Know where each answer comes from

An answer should come with information about the agent behind it: its model family, tools and operator. This is its provenance, meaning its origin and dependencies.

Why does this matter? Five agents may have different names but use the same model, the same source and the same operator. Those are clues that they might share a blind spot. They are not proof: different models can also make the same mistake.

### 2. Look at mistakes, not just opinions

Compare agents on questions with known outcomes. Look for agents that repeatedly get the same questions wrong together.

Two agents disagreeing today tells us little. Two agents with different patterns of past errors may offer something more useful. Equally, an agent that disagrees with everyone but is usually wrong is not valuable merely because it is different.

HAP's proposed selection method therefore considers both accuracy and shared mistakes. Diversity is something to measure, not something to assume from a provider's name.

### 3. Choose a team, then combine its forecasts

Select agents with attention to how their strengths and failures overlap. Combine their predictions, giving each a weight: how much influence its forecast has on the group's final forecast.

The hypothesis is that selection and weighting informed by shared errors can produce a better group than simple averaging. It must not reward loudness, confidence or disagreement for their own sake.

There is no validated "correct HAP formula" claimed here. Which agents to select, how to set their weights and how much extra complexity is worthwhile are research questions.

### 4. Let outcomes check the method

When the result becomes known, compare the forecast with reality. Accuracy and useful contributions should matter more than the number of answers produced.

In a future open market, this would also require checking that one operator cannot pretend to be many independent contributors. That is a proposed requirement, not a solved capability.

The first milestone is deliberately narrower: test whether diversity-aware selection and weighting actually help.

## Where the Vickrey auction fits

HAP began with a different question, before the diversity benchmark took shape: how can independent agents offer to do work without one central dispatcher needing to know every agent's costs?

The proposed answer is a **reverse Vickrey auction**, also called a second-price procurement auction. Ordinary auctions sell something to the highest bidder. Here, agents offer to do a job, so the lowest eligible bid wins and the winner is paid the second-lowest eligible bid.

### The rule, in plain numbers

Imagine three agents offering to complete the same clearly specified task:

- Agent A asks for 200 credits.
- Agent B asks for 250 credits.
- Agent C asks for 300 credits.

A wins and receives 250 credits, not 200. These are illustrative accounting units, not an existing payment system or a promise about actual prices.

Why pay more than the winning bid? The aim is to make honest cost reporting a sensible strategy. If A's true cost is 200, bidding 150 still earns the same 250. Bidding 260 loses the job to B. Bidding 240 still wins at 250, so it brings no extra gain either. Bidding below cost can backfire: had B asked 180, a bid of 150 would win A the job at 180, less than its cost of 200. A does not need to guess the highest price it can get away with charging.

In the standard single-task model, reporting one's true cost is a best strategy when each bidder has its own known cost, cares about payment minus cost, and cannot coordinate bids or control rival identities. This is an incentive result under assumptions, not a guarantee that real agents tell the truth or finish the work well.

### What this solves, and what it does not

The auction discovers a price and assigns work. The benchmark tests whether a group brings useful differences in judgment. These are different jobs.

A cheap bid is not evidence of quality or independence. The auction cannot by itself tell whether five bidders share one model, whether an answer is correct, or whether one operator controls several bidders. Nor does adding diversity weights to an auction automatically preserve its incentive guarantees. How quality and diversity requirements would interact with allocation and payment remains an open design question.

Uncertain costs are another risk: the cheapest estimate may simply be the most overoptimistic. Fake identities or coordinated bids can also distort the price. A signed bid can establish which registered identity submitted it; it cannot prove that the identity is an independent operator.

The earlier HAP design described signed bids, one bid per registered agent, a deadline and a maximum price. When only one bid arrived, it paid that maximum price. Those are historical design elements, not capabilities retested for this README. Quality disputes and enforcement were not designed.

**The order remains benchmark first, market layer later, only if the evidence supports it.** Even a successful forecasting benchmark would not prove that the auction improves task routing. That would need its own comparison with simpler allocation methods.

## What else holds the wider design together?

The original essay and reference code are kept in [docs/history](docs/history/).

The auction is one part of the earlier architecture, not the whole protocol. These ideas belong to that wider design; they are not all requirements of the first forecasting experiment or verified finished features.

### Shared work, separate private state

A shared task board lets agents find work. A shared workspace lets one agent leave a result or note that another can use. The earlier essay calls this **stigmergy**: coordination through traces left in the environment, rather than every instruction passing through a boss.

Shared work does not mean shared control over everything. Each agent should keep its own private memory and credentials, with clear boundaries on what others can change. The aim is to avoid one worker overwriting another's state while still letting useful results travel.

### Clear rules and local checks

Participants need stable rules for submitting work, allocating tasks and checking results. The design draws on Elinor Ostrom's approach to shared resources in *Governing the Commons* (1990): local monitors and graduated penalties, rather than relying on one all-seeing controller. The earlier proposal suggested increasing fees, suspension and eventual exclusion for misuse of shared resources such as rate limits. Their effectiveness is untested here.

This is an engineering choice to compare with a central dispatcher, not a claim that decentralization is always better. A simple dispatcher may be cheaper and easier when one operator controls all the workers.

### A safety check against herding

The original design also proposed watching for agents copying one another and converging too quickly. A warning could trigger another attempt and an independent check, such as running tests on generated software.

But agreement alone is not a fault. Ten correct answers should not trigger endless retries just because they match. Answer diversity is a warning signal at most; checked outcomes and shared-error history must do the judging. A validated threshold and retry policy are still open work.

### Economics as a design aid, not a political proof

Hayek supplies the dispersed-knowledge question. The broader motivation also draws on James Surowiecki's *The Wisdom of Crowds* (2004), which counts diversity and independence among the conditions for a wise crowd, and Norbert Wiener's *The Human Use of Human Beings* (1950), which warns against using people as interchangeable parts of a machine. Together they suggest that coordination should make differences useful, not flatten them away.

The earlier essay uses Albert Hirschman's distinction in *Exit, Voice, and Loyalty* (1970) between leaving a system and voicing a grievance to warn against describing software failures as political rebellion. HAP is an engineering proposal, not proof of a political worldview. Its economic analogies have to earn their place through useful, testable behavior.

## How is the benchmark run?

A benchmark is a fair comparison using the same questions and clear scoring rules. HAP's proposed larger study follows this sequence.

### 1. Collect questions with checkable outcomes

Use questions where the result can be settled, such as whether a specified event happens before a deadline. The proposed target is at least 500 real forecasts. Timestamped records from prediction platforms such as Metaculus or Manifold are candidate sources, subject to each platform's data terms, not a completed dataset claimed here.

### 2. Make sure the model could not know the answer

Keep the question, outcome, forecast time and model version together. A historical question only belongs in the test if its outcome was not available to the model when it made its forecast, including through training or supplied information.

Asking today's model about an old, resolved question can leak the answer into the experiment. A convincing-looking score would then measure hindsight rather than forecasting.

### 3. Keep learning separate from judging

Use one set of questions to develop the selection and weighting rules. Judge those rules on separate questions they were not tuned to answer. Fix the evaluation rules before inspecting the final results.

Otherwise, a method can look good because it was adjusted to fit the very questions used to judge it.

### 4. Compare three ways of forming the group

- **One model, many agents:** repeat the same model across several agents.
- **Different models, simple mixing:** average the predictions from a mixed-model group.
- **HAP's proposed method:** select and weight agents with attention to accuracy and shared mistakes.

Simple mixing is the comparison to beat. Beating repeated copies of one model alone would not show that the extra method is needed.

### 5. Score probabilities and shared failures

Agents give a probability, rather than only "yes" or "no". For example, a forecast might say an event has a 70% chance of happening.

The scorecard asks:

- **Accuracy:** how close are the probabilities to the outcomes? The planned measures include the Brier score, which penalizes forecasts more when they are further from what happens.
- **Shared failure:** how often do agents get the same question wrong together?
- **Worst failures:** does the method reduce the group's most damaging errors, including confident mistakes?
- **Cost:** is any improvement worth the extra computing and work?

A better average score is useful. HAP's more specific question is whether it reduces shared blind spots, especially in the failures that matter most.

### 6. Report the result, even if the simpler method wins

Show whether the difference is larger than the uncertainty in the data. A tiny numerical gap is not enough.

If simple mixing performs just as well, use the simpler approach. That would be a useful finding, not a reason to hide the experiment.

## An example

Suppose a group predicts whether a new business will survive its first year. Ten agents may agree because they all overlook the same cash-flow problem.

HAP would ask whether the group contains genuinely different sources of judgment and whether their past errors overlap. It would test whether a different selection and combination catches risks that the original group misses.

It would not force disagreement or automatically stop the system whenever everyone agrees. Agreement can be correct; disagreement can be noise. Outcomes decide.

*This is an illustration, not a reported experiment.*

## What do we know so far?

The September 2026 review reported a small synthetic smoke test: 24 simulated questions, 12 used to choose the groups and 12 to test them, with four agents per group. Both mixed-model groups clearly beat the repeated single model (Brier score about 0.12 against 0.30). HAP and simple mixed-model averaging were statistically indistinguishable at that size. Simple mixing is therefore a strong baseline, not proof that diversity is irrelevant. The test checked the experimental plumbing; it did not validate the hypothesis.

That result is historical and has not been independently rerun for this README. The 500-plus study remains a proposed next step. No larger validated result is claimed here.

**Status: exploratory research, not a proven product.**

## Why call it a protocol if the first step is a benchmark?

The wider ambition is a way for people and agents to contribute what they do best without being pushed into one way of thinking. An eventual protocol could carry information about contributors, coordinate allocation and use checked outcomes to assess performance.

The Vickrey auction and the wider coordination design remain part of that ambition. They have not been discarded. They are kept separate from the first benchmark so that building market machinery does not stand in for showing that the idea helps.

For now, the deliverable is evidence: does a diversity-aware group fail differently, and does that make it more reliable?

## Background reading

[Hayek's *The Use of Knowledge in Society*](https://www.econlib.org/book-chapters/chapter-the-use-of-knowledge-in-society/)

[Second-price auction incentives, explained in teaching notes](https://www.kiragoldner.com/teaching/DS574/fall22/L2.pdf). These notes describe the standard selling auction; HAP's example reverses it to buy a service.

> The question is not how many agents we can run. It is how many genuinely different ways of seeing a problem they bring, and whether that helps when reality checks their answers.
