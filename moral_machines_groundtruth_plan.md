# Moral Machines: ground-truth explanations and re-paired counterfactuals (draft plan)

*Draft, 2026-10-27. Both experiments need `OPENROUTER_API_KEY`; nothing here has called an API. Items marked
**[REVIEW]** need your decision; each has a proposed default. Reply "go", or give changes.*

## Why

- **Experiment 1 (Harry's suggestion):** if a plain-English description of how the model *actually* decides helps a
  simulator more than the model's own explanation, the gap measures information missing from the model's
  explanation. That's a direct measure of incompleteness. It's also WS1's setup (a known rule given as the
  explanation) applied to real models.
- **Experiment 2 (WS2):** counterfactuals chosen on purpose, near-tie vs. clear and cited vs. uncited factor, instead
  of the generator's random pairs. Each model has already answered about 1,700–2,000 scenarios, so any two of them
  form a new pair with a known reference answer. Only simulator calls are needed.

## Experiment 1: ground-truth explanations

**Models:** GPT-5.2 (medium), Gemini 3 Flash (medium), and Gemma 3 27B as the control.

**Split.** For each model, hold out **100 pairs, 20 per pair type**, sampled with a fixed seed. The rest are training
pairs: about 740 for GPT-5.2, 900 for Gemini 3 Flash and 570 for Gemma. Every type has at least 59 pairs per model, so
stratifying is feasible. The decision rule is **refit on training pairs only** for this experiment, so nothing about
the held-out pairs leaks into the ground-truth text.
- **[REVIEW]** 100 held-out pairs per model, stratified by pair type? *(Proposed: yes.)*

**Writing the ground-truth explanation.** Same procedure and template for every model. I (Claude) see only:
1. the training-only coefficient table, with plain-language feature labels and 95% intervals
2. 40 training decisions sampled across pair types, each showing the scenario and the model's answer, but **not its
   explanation**, so the text describes behaviour rather than paraphrasing the model

The output is at most 150 words: a ranked list of what decides the choice, with tie-breakers and stated exceptions,
plus a 3–5 item factor list. The rule is written as an "if X then Y, otherwise Z" procedure, so it's usable on any
counterfactual. The three texts are saved to `experiments/moral_machines/mm18_gt_explanations.json` before any
simulator call.
- **[REVIEW]** Is showing the coefficients allowed, or only raw decisions? Coefficients make the text more accurate but
  less "explanation-like". *(Proposed: allowed; it's the best available description of behaviour.)*
- **Sanity check:** Gemma's text must mention answer position. If it doesn't, I revise it before running.

**Simulator conditions** (on the held-out pairs, original scenario → counterfactual):

| Condition | Reference block shown to the simulator |
|---|---|
| own | the model's own explanation + its factor list (the paper's prompt) |
| ground truth | the ground-truth text + its factor list |
| ground truth + own | the model's own explanation, then "How this evaluator decides across many scenarios:" + the ground-truth text; the model's own factor list |
| none | the paper's no-explanation prompt (reference scenario + answer only) |

**Prompt format.** Built with `MoralMachines.create_counterfactual_prompt` / `create_counterfactual_prompt_no_explanation`
(`src/templates/moral_machines.py`), passing the ground-truth text in the `explanation` field. The template stays
identical, so the "own" and "none" prompts match the paper's exactly.
- **[REVIEW] Answer-position note.** The paper's predictor prompt doesn't show the counterfactual's answer format, so
  a simulator can't use a rule like Gemma's "picks Case 1 when asked to answer last". Proposed: add one line under the
  counterfactual scenario in **all four** conditions, "(The evaluator was asked to give its answer after / before its
  reasoning.)". Every condition sees it, so the comparison stays fair. The cost is that the "own" condition no longer
  exactly matches the paper's prompt. *(Proposed: add it. The alternative is to drop Gemma's position sanity check.)*

**Simulators:** `openrouter/openai/gpt-oss-20b` (reasoning medium, temperature 0.6, seed 666) and
`openrouter/google/gemini-3-flash-preview` (provider default reasoning). Both keys exist in `src/vllm_configs.py`.
- **[REVIEW] Re-run the "own" and "none" conditions, or reuse the paper's stored predictions?** Both simulators are
  among the paper's 5 predictors, but the stored gpt-oss-20b answers probably came from the local vLLM config
  (`openai/gpt-oss-20b`), not OpenRouter. The answer-position note also changes the prompt. *(Proposed: re-run all
  four conditions in one session. Then compare the stored answers with the new "own" answers as a drift check.)*

**Metrics.** Accuracy on the counterfactual answer and NSG for each condition, pooled over the two simulators and
shown separately for each, with 95% pair-bootstrap intervals. **Key quantity: ground-truth NSG minus own-explanation
NSG**, per model. Also report the fitted rule's held-out accuracy on the same 100 pairs, as the ceiling.
- **Expected result** (it can fail): ground truth > own for all three models, with the gap largest for Gemma (whose
  real cause is never mentioned) and smallest for GPT-5.2. If ground truth + own > ground truth alone, the model's own
  explanation adds pair-specific information.

## Experiment 2: re-paired counterfactuals

Code: `analysis/mm18_repair.py` (already run, no API). It writes three files to `experiments/moral_machines/`:
- `mm18_repair_candidates.csv`: all 275,891 candidates, with group labels
- `mm18_repair_counts.csv`: the counts below
- `mm18_repair_sample.csv`: the proposed sample, 226 pairs

**Construction.**
- **Pool:** 5,010 answered scenarios (both sides of each model's existing pairs).
- **Pairing:** a new pair (A → B) needs the same model, the same theme, the same car/swerve structure (`is_in_car`,
  `is_interventionism`) and the same answer position. A must have a parsed explanation, and the pair must not already
  exist.
- **Near-tie:** the fitted rule's held-out P(Case 1 | B) is within 0.2 of 0.5.
- **Cited:** the changed factor appears in A's own `most_important_factors` list, by an **approximate** keyword map.
  For example, "pedestrian" counts as the passengers factor, and "rule" counts as law.

**Candidate pairs** (change size = how many rule features differ between A and B; "counts only" means head counts
change but the net difference doesn't):

| Model | Change size | Clear, cited | Clear, uncited | Near-tie, cited | Near-tie, uncited |
|---|---|---|---|---|---|
| GPT-5.2 | counts only | 18,050 | 2,841 | 1,001 | 25 |
| | one rule feature | 42,440 | 2,417 | 3,378 | **53** |
| | several | 14,024 | 351 | 2,389 | 9 |
| Gemini 3 Flash | counts only | 16,604 | 9,876 | 3,385 | 1,591 |
| | one rule feature | 51,271 | 5,922 | 9,119 | 630 |
| | several | 17,060 | 220 | 4,764 | 90 |
| Gemma 3 27B | counts only | 2,693 | 11,984 | 707 | 2,081 |
| | one rule feature | 21,572 | 10,759 | 3,913 | 939 |
| | several | 10,075 | 2,687 | 727 | 244 |

**Reference flip rate** on single-feature changes: GPT-5.2 35% clear vs. 50% near-tie; Gemini 3 Flash 52% vs. 51%;
Gemma 16% vs. 48%.

**Proposed sample.**
- Single-feature changes only, with 20 pairs per near-tie × cited cell for each model.
- Each original scenario is used at most once, and each cell is half flips, half non-flips where possible.
- That gives **226 pairs**: Gemini 3 Flash and Gemma 80 each, and GPT-5.2 **66**.
- GPT-5.2's near-tie, uncited cell has only 6 distinct originals. GPT-5.2's factor lists name almost everything, which
  matches your hand labels.

- **[REVIEW] GPT-5.2's near-tie, uncited cell (6 pairs).** Accept it as small, or relax the cell to "several rule
  features" too? *(Proposed: accept, and report it as small.)*
- **[REVIEW] Factor balance.** With car/swerve structure fixed, single-feature changes are mostly the traffic light
  (154 of 226 pairs change law), then gender (41) and social status (16). Stratify by changed factor as well?
  *(Proposed: cap law at half of each cell.)*
- **[REVIEW] Keyword-based "cited".** It's approximate. Hand-check 20 sampled pairs in the report before running?
  *(Proposed: yes, about 10 minutes.)*

**Conditions:** own and none, on both simulators, using the same prompts as experiment 1, including the
answer-position note.
- **[REVIEW]** Add the ground-truth condition here too? It costs 50% more calls, and it shows whether the ground truth
  helps most on near-ties. *(Proposed: no for now; add it if experiment 1 shows a clear gap.)*

**Metrics:** NSG for each near-tie × cited cell with pair-bootstrap intervals; flip rate for each cell.

## Cost and run plan

**Prompt size.** Paper predictor prompts on Moral Machines average about 4,000 characters (about 1,000 tokens) with
the explanation and about 3,000 (about 750 tokens) without. Ground truth + own adds about 200 tokens. The stored
predictor reasoning averages about 1,500 characters, so assume **300–1,000 output tokens per call**. The stored
token-count columns are empty.

| Step | Calls | Input tokens | Output tokens (≈500 per call) |
|---|---|---|---|
| Smoke test: 5 held-out GPT-5.2 pairs × ground truth × 2 simulators | 10 | ~10k | ~5k |
| Exp. 1: 3 models × 100 pairs × 4 conditions × 2 simulators | 2,400 | ~2.4M | ~1.2M |
| Exp. 2: 226 pairs × 2 conditions × 2 simulators | 904 | ~0.8M | ~0.45M |
| **Total** | **~3,300** | **~3.2M** | **~1.7M** |

- **Prices:** fill in from the OpenRouter model pages for `openai/gpt-oss-20b` and `google/gemini-3-flash-preview`.
  The smoke test's `usage` fields give the real token counts and cost per call, which we then scale up.
- **[REVIEW] Budget.** That's about 3,300 calls, more than the 1,000–2,000 I estimated earlier, because experiment 1
  re-runs all four conditions. To halve it, use only gpt-oss-20b, or reuse the stored "own"/"none" answers.
  *(Proposed: both simulators, if the smoke test prices the total under the key's credit limit.)*

**Script.** A new `analysis/mm18_gt_run.py`:
- It builds prompts with the `MoralMachines` template, calls `src.utils.get_messages` for each simulator config, and
  parses the answers with `parse_response` against `VALID_ANSWERS`.
- It saves after each batch, so an interrupted run can resume, and it writes
  `experiments/moral_machines/mm18_gt_predictions.csv`.
- **Why not `generate_predictor_answers`:** it reads the prompts stored in a parquet and can't vary the explanation
  block without building a custom parquet.

`--smoke` runs the 10-call test and prints the parse rate, tokens and cost. The full run waits for your go-ahead
after the smoke test.

## Review checklist
- [ ] 100 held-out pairs per model, stratified by pair type
- [ ] Ground-truth text may use the training-only coefficients plus 40 training decisions (no model explanations)
- [ ] Add the answer-position note to all conditions
- [ ] Re-run all four conditions (no reuse of stored predictions); drift check against stored answers
- [ ] GPT-5.2's near-tie, uncited cell stays small (6 pairs)
- [ ] Cap law changes at half of each cell
- [ ] Hand-check 20 pairs labelled cited by keyword
- [ ] No ground-truth condition in experiment 2 for now
- [ ] Both simulators, subject to the smoke-test price
