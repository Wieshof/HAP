# HAP diversity experiment results

Test questions: 12. Groups have at most 4 members. Exact unrounded values are in `results.csv`.

| Group | Size | Average error (Brier) ↓ | Everyone wrong ↓ | Confidently wrong ↓ | Shared error ↓ |
| --- | --- | --- | --- | --- | --- |
| Same model repeated | 4 | 0.08 | 2 of 12 | 0 of 12 | 0.99 |
| Naive mix | 4 | 0.08 | 0 of 12 | 0 of 12 | 0.30 |
| HAP diversity-aware | 4 | 0.06 | 0 of 12 | 0 of 12 | 0.24 |
| Contribution-based | 1 | 0.10 | 1 of 12 | 1 of 12 | n/a |

- Same model repeated: gpt_1, gpt_3, gpt_2, gpt_4
- Naive mix: claude_2, gpt_1, gemini_2, local_2
- HAP diversity-aware: claude_2, claude_1, gemini_2, gpt_1
- Contribution-based: claude_2

Everyone wrong: every member gave the true outcome less than 50%. Confidently wrong: the group gave the wrong outcome 75% or more. Shared error: mean pairwise correlation of members' errors (1.00 = identical mistakes).

With 12 test questions, small differences are noise. Synthetic data test the pipeline, not the hypothesis.
