# Buyer test: the cheapest option that meets the requirement

Requirement: Brier at most 0.10, confidently wrong at most 25% of questions, cost at most 3.0 credits per forecast. Costs are the fixture's illustrative prices, summed over the members who must answer.

Options considered: every single agent, every group of up to three, and the four benchmark groups (178 in all). Each is judged on 12 training questions by the pessimistic end of its interval: the 95th percentile of a bootstrap for Brier, and a 95% upper bound for the confidently-wrong rate. The cheapest option that passes is chosen, then checked on the untouched test questions.

**Chosen: claude_2 (= Contribution-based)**, cost 1.2.
Most accurate option on training questions: claude_2 (= Contribution-based), cost 1.2.

## On the training questions (used to choose)

| Option | Cost | Brier | Brier, pessimistic | Confidently wrong | Confidently wrong, pessimistic | Meets requirement |
| --- | --- | --- | --- | --- | --- | --- |
| claude_2 (= Contribution-based) | 1.2 | 0.052 | 0.088 | 0 of 12 | 22% | yes |
| local_1 | 0.3 | 0.198 | 0.329 | 2 of 12 | 44% | no |
| local_2 | 0.3 | 0.193 | 0.322 | 1 of 12 | 34% | no |
| gemini_1 | 0.9 | 0.149 | 0.226 | 0 of 12 | 22% | no |
| gemini_2 | 0.9 | 0.120 | 0.189 | 0 of 12 | 22% | no |
| gpt_1 | 1.0 | 0.123 | 0.246 | 1 of 12 | 34% | no |
| gpt_2 | 1.0 | 0.139 | 0.260 | 1 of 12 | 34% | no |
| gpt_3 | 1.0 | 0.137 | 0.258 | 1 of 12 | 34% | no |
| gpt_4 | 1.0 | 0.140 | 0.267 | 1 of 12 | 34% | no |
| claude_1 | 1.2 | 0.072 | 0.127 | 0 of 12 | 22% | no |
| Same model repeated: gpt_1 + gpt_2 + gpt_3 + gpt_4 | 4.0 | 0.134 | 0.256 | 1 of 12 | 34% | no |
| Naive mix: claude_2 + gemini_2 + gpt_1 + local_2 | 3.4 | 0.095 | 0.153 | 0 of 12 | 22% | no |
| HAP diversity-aware: claude_1 + claude_2 + gemini_2 + gpt_1 | 4.3 | 0.073 | 0.119 | 0 of 12 | 22% | no |

## On the test questions (check only)

| Option | Cost | Brier | Brier, pessimistic | Confidently wrong | Confidently wrong, pessimistic | Meets requirement |
| --- | --- | --- | --- | --- | --- | --- |
| claude_2 (= Contribution-based) | 1.2 | 0.096 | 0.205 | 1 of 12 | 34% | no |

The pessimistic bound protects each option on its own, not the best of 178: the more options are compared on the same few questions, the more likely the winner was partly lucky. The test check is there to catch that.

With 12 test questions these checks are rough. Synthetic data and made-up prices show how the decision works, not which real model to buy.
