# Pilot: real model forecasts on open questions

The importers give HAP real questions and real market prices, but every signal
they produce comes from a market price. What is missing is independent
forecasts from AI models, recorded before the answers are known. This pilot
collects them and asks one question:

> Does HAP select something better, or something cheaper that is still good
> enough?

A negative answer is useful too. With 50–100 questions this is a pilot: it can
show a direction and how uncertain it is, not prove the case.

## The plan, fixed before any forecast is collected

All of the following is written into `registration.json` when you register.
The file is hashed, and `forecast`, `resolve` and `score` refuse to run if it
has changed. Change the plan by editing your config **before** registering. A
new plan means a new run.

### Questions

- Open binary questions from Manifold, Polymarket and Kalshi. Metaculus is
  left out because its API needs a token.
- Closing 14 to 75 days after registration, so the pilot finishes within about
  three months.
- Market price between 5% and 95%. Questions the market already treats as
  settled tell us little.
- Enough trading to make the price meaningful:
  - Manifold: at least 20 bettors.
  - Polymarket: at least $10,000 volume.
  - Kalshi: at least 1,000 contracts.
  - On Polymarket and Kalshi, a bid–ask spread of at most 0.10. The market
    price is the midpoint.
- At most one market per Polymarket or Kalshi event, the most liquid one. This
  avoids twenty near-copies of one question.
- 60 questions by default, an equal share per platform. They are drawn with a
  seeded shuffle, and the seed is recorded.
- Half are train questions (used to select groups) and half are test questions
  (used to score them), alternating within each platform so both halves have
  the same mix. The split is random rather than by resolution date, because
  questions that resolve early are systematically different (shorter and
  often easier).

### Forecasts

- Every agent answers every question once. The same prompt goes to every
  agent, in one of two variants: `direct`, or `base_rate`, which asks for
  reference classes first. It includes the question, its resolution details,
  the close date and today's date.
- No tools and no web access. Models never see the market price; otherwise
  they would copy it rather than forecast.
- Each answer is stored with its UTC time, the full reply, tokens used and
  actual cost: tokens times the price per million tokens in the config, or
  the cost a `command` agent reports.
- Forecasts must be collected within 48 hours of registration. Later answers,
  and any answer given after its question closed or resolved, are not scored.
- Failed calls are retried, up to three attempts per question and agent.
  Refusals count as missing; there is no fallback to another model, because
  that would put another model's answer under this agent's name.

### Selection (train questions only)

- **HAP diversity-aware:** `select_hap`, k = 3, with the weights in the
  registration.
- **HAP contribution-based:** `select_contribution`, k = 3.
- **HAP buyer test:** the cheapest single agent or group of up to three whose
  pessimistic train scores meet the requirement. The default requirement is
  Brier ≤ 0.20, confidently wrong ≤ 25%, cost ≤ $0.05 per question.
- For reference: same model repeated, and naive mix.

An agent is a model plus a prompt. Its price in selection is its mean actual
cost per question on the train questions.

### Comparators

- **A cheap single model**, named in the config (`cheap_baseline`).
- **Simple averaging**: the plain average of all agents.
- **The market**: the market price when the questions were registered, which
  is the information available when the models answered.

### Scoring (test questions only)

- Main metric: Brier score of the group's averaged forecast. Also reported:
  log loss, confidently wrong (≥ 75% on the wrong side), and actual cost per
  question.
- For each HAP group against each comparator: the Brier difference, with a
  95% paired bootstrap interval over the test questions. The verdict:
  - **better** if the whole interval is below 0;
  - **worse** if the whole interval is above 0;
  - **not worse by more than the margin** if the upper end is below 0.02;
  - **unclear** otherwise.

  "Cheaper" means lower actual cost per question.
- Agents that answered fewer than 90% of resolved questions are dropped.
  After that, questions missing a forecast from any remaining agent are
  dropped.
- Questions that resolve N/A, are cancelled or stay unresolved are left out.

## Running it

Install the SDKs for the providers you use (only those):

```bash
pip install anthropic openai google-genai
```

Each SDK reads its key from the environment, for example `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY` and `GEMINI_API_KEY`. Keys never go into the config or the
run folder.

1. **Configure.** Copy `config.example.json` to `config.json`. Fill in the
   OpenAI and Gemini model names and their prices per million tokens, or
   delete those agents. Check the Anthropic prices against the current price
   list. Set the requirement and margin now; they cannot change later.
   Any other model can join as a `command` agent. That is a program that
   reads `{"system", "prompt", "model"}` as JSON on stdin and prints
   `{"text", "input_tokens", "output_tokens"}`, optionally with `"cost_usd"`.

2. **Register**, then commit and push the run folder. The push time on GitHub
   is the public record that the plan came first.

   ```bash
   hap-pilot register --config pilot/config.json --run pilot/runs/2026-11
   ```

3. **Forecast** within 48 hours. Start with a dry run, which shows the prompts
   and calls nothing, and a few real calls:

   ```bash
   hap-pilot forecast --run pilot/runs/2026-11 --dry-run
   ```

   ```bash
   hap-pilot forecast --run pilot/runs/2026-11 --limit 8
   ```

   ```bash
   hap-pilot forecast --run pilot/runs/2026-11 --workers 4
   ```

   Re-running continues where it stopped. Commit and push again when it is
   done.

4. **Resolve** after the questions close. Run it again until (nearly) all have
   resolved:

   ```bash
   hap-pilot resolve --run pilot/runs/2026-11
   ```

5. **Score**:

   ```bash
   hap-pilot score --run pilot/runs/2026-11
   ```

   This writes `report.md` in the run folder.

## Cost

The pilot spends real money on model calls. With 60 questions and the eight
example agents it is 480 calls. Rough estimate: a reply with reasoning uses
1,000–4,000 output tokens, which is about $0.02–0.08 per Opus call and far
less for small models. Ten to twenty dollars in total is a reasonable
expectation. The report shows what it actually cost. Try `--limit` first to
see real numbers.

## What this pilot can and cannot show

- **Small samples.** About 30 test questions give intervals of several
  hundredths of Brier. Most differences will come out "unclear". That is an
  honest result for a pilot, and it tells you how many questions a real study
  needs.
- **Correlated questions.** Questions are not fully independent: several may
  depend on the same election or the same price.
- **Model knowledge.** Models answer from training data that ends at their
  knowledge cutoff, while the market keeps learning. Beating the market is not
  expected; the interesting comparisons are among the model groups, and how
  close they get to the market at what cost.
- **One question at a time.** One forecast per agent and question, at one
  moment. A model's answers vary from run to run, and that noise is part of
  the result.
