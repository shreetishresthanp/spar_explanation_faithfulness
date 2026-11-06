# 11/04 pitch MVP: behaviour description vs self-explanation (no API calls)

**Pitch:** use a description of how each model actually behaves to find what its self-explanation leaves out or
contradicts, then build in-scope counterfactuals that target exactly those gaps. This separates unfaithfulness from
out-of-scope failure (Caleb's open problem), on a value-laden task where explanations can't score by restating textbook
knowledge (Swapnanil).

For this MVP the "behaviour description" is the fitted logistic rule (`experiments/moral_machines/mm18_rule_coefs.csv`),
not an LLM-written text. Code: `analysis/mm18_gap_table.py` → `experiments/moral_machines/mm18_gap_table.csv`,
`mm18_gap_format.csv`.

## 1. The paper's counterfactuals can't see a format effect, by construction

The counterfactual prompt is built with the original's `answer_first`
(`src/counterfactual_generation/llm_counterfactual_generation/generator_classes.py:393`,
`src/reference_answer_generation/reference_answer_generator.py:451-458`). Parsing the stored prompts confirms it: format
differs in 0 of 22,480 MM prompt pairs. Random NSG pairs therefore can't test a format effect.

## 2. Gap type A: a factor the behaviour uses, but the explanation never mentions

Answer format, P(Case 1) answer-last minus answer-first. Format was assigned at random per scenario, so these are
randomised estimates (95% bootstrap CI):

| Model | first | last | diff | explanations mentioning format |
|---|---|---|---|---|
| Gemma 3 27B | 0.59 | 0.96 | **+0.36 [+0.32, +0.41]** | 0 / 999 |
| Gemini 3 Pro | 0.52 | 0.61 | **+0.08 [+0.02, +0.14]** | 0 / 997 |
| Gemini 3 Flash | 0.56 | 0.60 | +0.03 [−0.03, +0.10] | 0 / 998 |
| GPT-5.2, Claude ×3, GPT-5 Mini | | | −0.02 to +0.03 (all CIs cross 0) | 0 |

- **Gemini 3 Pro.** The 10/28 rule flagged it (coef −0.53). New here: a randomised confirmation, small but significant.
  It isn't a parser artifact, since every answer in both formats parsed. Its explanations never mention format.
- **Gemini 3 Flash and Sonnet 4.5.** Significant in the rule (−0.21, −0.20), but the raw difference crosses 0, and
  Flash showed 0% format flips in the 10/28 fresh test. Treat them as not established.

## 3. Content factors are almost always mentioned

Keyword match, among fitted-rule factors whose CIs exclude 0:

- **Theme factors** (species, status, age, fitness, group size; filtered to their own scenarios): mentioned in 68–100%.
  The low case is GPT-5.2 on age (38%, coef −0.56, n = 98).
- **Structural factors** (passengers, swerve, law, lives): *not* filtered to the scenarios where they apply, so their
  roughly 50–100% rates are lower bounds. Don't read them as gaps.

So simply leaving content out isn't the main gap. This agrees with the 10/27 hand labels: explanations cite almost
everything.

## 4. Gap type B: a factor the explanation mentions in order to disavow, while the behaviour uses it

Gemma, gender scenarios, **answer-first prompts only.** This removes the format effect, which pushes answer-last
prompts to Case 1 about 95% of the time.

| | picks Case 1 |
|---|---|
| Case 1 spares the women | 100% (36/36) |
| Case 1 kills the women | 33% (35/106) |

Gemma strongly spares women. Yet 79 of its 316 gender explanations (25%) say gender or demographics are irrelevant. In 8
hits I read, all were genuine disavowals, e.g. "the demographic details (gender, age, profession) are irrelevant ... and
should not influence the decision". Frame this as **"Gemma uses gender, and sometimes says it doesn't"**. Claude
disavows more often (70%), but its behaviour matches that.

Fitted gender coefficient vs disavowal rate, all models (regex):

| Model | gender coef [95% CI] | disavows gender |
|---|---|---|
| **Gemma 3 27B** | **−1.18 [−1.57, −0.98]** (spares women) | 25% |
| Claude Opus / Sonnet 4.5 | −0.11 (CI crosses 0) | 70–71% |
| GPT-5 Mini | +0.01 | 70% |
| GPT-5.2, Gemini 3 Flash / Pro | ≈ 0 | 22–44% |

The gender scenarios aren't balanced by which case kills the women (36 vs 106). Other factors could co-vary with that
split; the additive rule controls for them, but this two-way table doesn't.

## Next (targeted counterfactuals; the actual experiment)

1. Gap-targeted pairs:
   - flip only the format (Gemma, Gemini 3 Pro)
   - mirror only the genders (Gemma)
   - controls: flip a factor the explanation does state (e.g. the traffic light)
2. Run the simulator with own explanation vs none vs rule description. Prediction:
   - the explanation fails on the gap pairs and works on the controls
   - the description works on both
3. Constraints, from teammates:
   - a capable simulator (Christina)
   - the reference model resampled so only consistent items are kept (Christina)
   - single-factor edits, inside the explanation's stated scope (Caleb)
   - report no-explanation accuracy alongside NSG (Karol)

## Caveats

- Mention and disavowal rates come from keyword matching. Hand-check about 10 hits of each before quoting them.
- Gemma's rule uses only its 667 linked pairs.
- The rule is additive, so the gender coefficient is estimated with format held fixed.
