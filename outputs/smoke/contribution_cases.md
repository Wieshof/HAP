# What does each model add? Four synthetic cases

Each case is a synthetic fixture built to show one pattern, with 100 training and 100 test questions. Its settings were chosen so the pattern is visible. It shows that the measurements detect the pattern, not that real AI systems behave this way.

**Contribution** is how much worse the group's averaged forecast gets (in Brier score) when that agent is removed. Positive means the agent helps, negative means the group is better without it. The 95% interval comes from resampling the test questions; **helps** or **hurts** is shown only when the whole interval is on one side of zero. A contribution belongs to this group: the same agent can matter more or less elsewhere.

**Error correlation** is measured from forecasts: how closely an agent's misses track the others' on the same questions (about 0 means unrelated mistakes). **Most similar errors** names the member it tracks most closely. **Declared overlap** is the share of model family, operator and pipeline an agent claims to share with the others.

## Redundancy

*Do six near-identical copies of one model improve the result at all?*

All 8 agents averaged: Brier 0.065 on 100 test questions.

| Agent | Declared family | Price | Solo Brier | Contribution | 95% interval | Verdict | Error correlation with others | Most similar errors | Declared overlap with others |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_1 | gpt | 1.0 | 0.064 | +0.0007 | -0.0004 to +0.0017 | unclear | 0.72 | gpt_4 (0.99) | 0.71 |
| gpt_2 | gpt | 1.0 | 0.067 | +0.0003 | -0.0008 to +0.0012 | unclear | 0.72 | gpt_4 (0.99) | 0.71 |
| gpt_3 | gpt | 1.0 | 0.066 | +0.0004 | -0.0007 to +0.0014 | unclear | 0.72 | gpt_4 (0.99) | 0.71 |
| gpt_4 | gpt | 1.0 | 0.065 | +0.0005 | -0.0006 to +0.0014 | unclear | 0.72 | gpt_1 (0.99) | 0.71 |
| gpt_5 | gpt | 1.0 | 0.065 | +0.0005 | -0.0006 to +0.0015 | unclear | 0.72 | gpt_4 (0.99) | 0.71 |
| gpt_6 | gpt | 1.0 | 0.064 | +0.0006 | -0.0004 to +0.0015 | unclear | 0.72 | gpt_1 (0.99) | 0.71 |
| claude_1 | claude | 1.2 | 0.128 | -0.0006 | -0.0036 to +0.0025 | unclear | 0.05 | gpt_3 (0.08) | 0.00 |
| gemini_1 | gemini | 0.9 | 0.131 | +0.0002 | -0.0041 to +0.0053 | unclear | -0.02 | gpt_2 (0.01) | 0.00 |

Averaging the first n copies only:

| Copies | Brier |
| --- | --- |
| 1 | 0.064 |
| 2 | 0.065 |
| 3 | 0.065 |
| 4 | 0.065 |
| 5 | 0.065 |
| 6 | 0.065 |

Groups of up to 3 chosen on training questions, scored on test questions:

- HAP diversity-aware: gpt_4, claude_1, gemini_1 (Brier 0.079)
- Contribution-based: claude_1, gemini_1, gpt_4 (Brier 0.079)

## Complementarity

*Can a weaker, cheaper model improve the group because its mistakes are different?*

All 4 agents averaged: Brier 0.109 on 100 test questions.

| Agent | Declared family | Price | Solo Brier | Contribution | 95% interval | Verdict | Error correlation with others | Most similar errors | Declared overlap with others |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_1 | gpt | 1.0 | 0.125 | -0.0013 | -0.0064 to +0.0035 | unclear | 0.64 | gemini_1 (0.95) | 0.22 |
| claude_1 | claude | 1.2 | 0.123 | -0.0011 | -0.0058 to +0.0031 | unclear | 0.66 | gemini_1 (0.95) | 0.22 |
| gemini_1 | gemini | 0.9 | 0.130 | -0.0033 | -0.0081 to +0.0015 | unclear | 0.65 | gpt_1 (0.95) | 0.22 |
| local_1 | llama | 0.3 | 0.137 | +0.0143 | +0.0020 to +0.0276 | helps | 0.06 | claude_1 (0.09) | 0.00 |

Groups of up to 3 chosen on training questions, scored on test questions:

- HAP diversity-aware: gemini_1, local_1, gpt_1 (Brier 0.108)
- Contribution-based: claude_1, gemini_1, local_1 (Brier 0.108)

## Illusory diversity

*What happens when a model declared as different makes the same systematic errors?*

All 4 agents averaged: Brier 0.078 on 100 test questions.

| Agent | Declared family | Price | Solo Brier | Contribution | 95% interval | Verdict | Error correlation with others | Most similar errors | Declared overlap with others |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_1 | gpt | 1.0 | 0.090 | +0.0028 | -0.0031 to +0.0080 | unclear | 0.35 | mistral_1 (0.97) | 0.00 |
| claude_1 | claude | 1.2 | 0.127 | +0.0024 | -0.0058 to +0.0112 | unclear | 0.04 | mistral_1 (0.05) | 0.00 |
| mistral_1 | mistral | 0.8 | 0.086 | +0.0034 | -0.0022 to +0.0084 | unclear | 0.37 | gpt_1 (0.97) | 0.00 |
| llama_1 | llama | 0.3 | 0.108 | +0.0025 | -0.0044 to +0.0106 | unclear | 0.05 | mistral_1 (0.09) | 0.00 |

Groups of up to 3 chosen on training questions, scored on test questions:

- HAP diversity-aware: llama_1, gpt_1, claude_1 (Brier 0.081)
- Contribution-based: claude_1, gpt_1, llama_1 (Brier 0.081)

## Counterproductive diversity

*Can the method spot a model whose answers differ but make the group worse?*

All 4 agents averaged: Brier 0.077 on 100 test questions.

| Agent | Declared family | Price | Solo Brier | Contribution | 95% interval | Verdict | Error correlation with others | Most similar errors | Declared overlap with others |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_1 | gpt | 1.0 | 0.062 | +0.0138 | +0.0082 to +0.0202 | helps | 0.12 | gemini_1 (0.22) | 0.00 |
| claude_1 | claude | 1.2 | 0.055 | +0.0208 | +0.0141 to +0.0281 | helps | 0.01 | contrarian_1 (0.10) | 0.00 |
| gemini_1 | gemini | 0.9 | 0.068 | +0.0146 | +0.0082 to +0.0211 | helps | 0.03 | gpt_1 (0.22) | 0.00 |
| contrarian_1 | other | 0.5 | 0.288 | -0.0310 | -0.0388 to -0.0231 | hurts | 0.06 | gpt_1 (0.20) | 0.00 |

Groups of up to 3 chosen on training questions, scored on test questions:

- HAP diversity-aware: gemini_1, claude_1, gpt_1 (Brier 0.046)
- Contribution-based: claude_1, gemini_1 (Brier 0.049)
