# Moral Machines: next experiments (with the OpenRouter key)

*Working doc, 2026-10-27. Midterm deadline 2026-11-02. This is the run order and the checklist. The detailed design
(splits, prompts, candidate counts, token estimates) is in `moral_machines_groundtruth_plan.md`; this doc doesn't
repeat it.*

**Legend:**
- **[YOU]** = an action item for you: a decision, a review, or a hand check.
- **[CLAUDE]** = an implementation step I do.
- Nothing paid runs until you tick the gate before it.

---

## What we're testing

| # | Experiment | Question | Calls (est.) |
|---|---|---|---|
| 0 | Smoke test | Does the pipeline work through OpenRouter, and what does a call cost? | 10 |
| 1 | Ground-truth explanations | How much does each model's own explanation leave out, compared with a description of how it actually decides? | ~2,400 |
| 2 | Re-paired counterfactuals | Is explanation gain lower on near-ties, and does it depend on whether the change hits a factor the explanation cites? | ~900 |
| 3 | Resampling near-ties *(optional)* | Are near-tie failures coin flips by the reference model, or missing information? | ~400 (reference model calls) |

Experiments 1 and 2 answer the two open points in `moral_machines_writeup.md` (§5, §7, §10). Experiment 3 overlaps
with Christina's non-determinism work: coordinate before running it.

---

## Step 0: set up the key and run the smoke test

**[YOU]**
- [ done ] Put the key in a file named `.env` at the repo root, with this single line (git already ignores `.env`):
      ```
      OPENROUTER_API_KEY=sk-or-...
      ```
      *Current status: the key is in `.env` and loads (checked 2026-10-27).*
- [ ] Check with your collaborator that the key has a **credit limit**, and note it here: $_40___
- [ done] Tell me when it's in place. Don't paste the key into the chat.

**[CLAUDE]**
1. Check that the key loads, without printing it.
2. Write `analysis/mm18_gt_run.py` (spec in the ground-truth plan, "Script"):
   - builds prompts with the `MoralMachines` template, so they match the paper's
   - calls the two simulators through `src.utils`
   - parses answers with `parse_response`
   - saves after each batch so a run can resume
   - `--smoke` mode
3. Run `--smoke`: 5 held-out GPT-5.2 pairs × the ground-truth condition × 2 simulators = 10 calls.
4. Report:
   - the parse rate (target: 10/10)
   - tokens per call
   - the **actual cost per call**, scaled to the full runs
   - one example prompt and answer for you to read

**Smoke test results (2026-10-27, done)**

- **Change from step 3:** the ground-truth texts don't exist yet, because they wait for Gate 1a. So the smoke test ran
  the **"own" condition** on 5 GPT-5.2 pairs (one per pair type, seed 0), with the answer-position line, × 2
  simulators = 10 calls.
- **The key loads** from `.env`.
- **The prompts match the paper's.**
  - The rebuilt "own" and "none" prompts are identical to the paper's stored prompts, apart from blank lines, on every
    pair: GPT-5.2 (840), Gemini 3 Flash (998) and Gemma 3 27B (999).
  - The script checks this before every smoke run and stops if it fails.
  - The only addition is one line under the counterfactual scenario: "(Note: for this scenario the evaluator was
    asked to state its answer before / after its reasoning.)"
- **Parse rate: 10 of 10.** 9 of 10 predictions were right.

| Simulator | Prompt tokens per call | Output tokens per call | Note |
|---|---|---|---|
| `openai/gpt-oss-20b` | ~900 | ~375 | Reasons (medium effort, as in the paper's config) |
| `google/gemini-3-flash-preview` | ~860 | ~14 | No reasoning at the provider default, the same config as the paper's stored predictor |

- **Cost:**
  - The key's usage counter still read $0.00 right after the run, so the cost below is estimated from OpenRouter list
    prices (fetched from `/api/v1/models`). Re-check the key's usage later to confirm.
  - Gemini 3 Flash costs $0.50 / $3 per M input / output tokens; gpt-oss-20b costs $0.018 / $0.09; GPT-5.2 costs
    $1.75 / $14.
  - Per call: about **$0.0005** for Gemini 3 Flash and **$0.00005** for gpt-oss-20b. The smoke test cost about $0.003.
  - Projected, both simulators:
    - experiment 1 (~2,400 calls): about $0.70
    - experiment 2 (~900 calls): about $0.25
    - experiment 3 (GPT-5.2 at medium reasoning, about $0.02 per call): about $5–10 for 200–400 calls
- **Credit limit:** the key reports a limit of **$50**, not the $40 noted above.
- **Code and output:**
  - Script: `analysis/mm18_gt_run.py --smoke`
  - Prompts, answers and reasoning: `experiments/moral_machines/mm18_gt_smoke.csv`

**[YOU] Gate 0**
- [ done] Read the example prompt. Does it look like the paper's predictor prompt, plus the answer-position line?
- [ done] Is the projected total cost under the credit limit? If not, choose:
  - gpt-oss-20b only (halves the cost)
  - drop the "ground truth + own" condition (−25% in experiment 1)
  - fewer held-out pairs

## Step 1: experiment 1, ground-truth explanations

**[YOU] Decisions** (proposed defaults in brackets; the ground-truth plan explains each):
- [yes ] Hold out 100 pairs per model, 20 per pair type *(yes)*
- [ yes] The ground-truth text may use the training-only coefficients plus 40 training decisions, but not the model's
      own explanations *(yes)*
- [yes] Add the answer-position line to all 4 conditions *(yes; without it, no simulator can use a rule like Gemma's)*
- [yes] Re-run all 4 conditions rather than reuse the stored predictions, and check them against the stored answers
      for drift *(yes)*

**[CLAUDE] Implementation**
1. **Split.** Refit the decision rule on training pairs only, for GPT-5.2, Gemini 3 Flash and Gemma 3 27B. Save the
   held-out pair ids.
2. **Write the three ground-truth explanations** from the training coefficients and 40 training decisions. Each is
   ≤150 words, an "if X then Y, otherwise Z" procedure with a factor list. Save them to
   `experiments/moral_machines/mm18_gt_explanations.json`.
3. **Sanity check:** Gemma's text must mention answer position.

**[YOU] Gate 1a: read the three ground-truth texts before any calls (about 10 minutes)**
- [ ] Is each one a fair, plain description of how that model decides?
- [ ] Is any one so specific that it gives away individual held-out answers? (It shouldn't be; it's written from
      training pairs only.)

**[CLAUDE] Run**
4. Run 3 models × 100 pairs × 4 conditions × 2 simulators (about 2,400 calls), resumable.
   - The 4 conditions: the model's own explanation, the ground truth, ground truth + own, and none.
5. Analyse in `analysis/mm18_gt.py`:
   - accuracy and NSG per condition, with pair-bootstrap CIs, pooled and per simulator
   - **ground-truth NSG minus own-explanation NSG**, per model
   - the fitted rule's accuracy on the same pairs, as the ceiling
   - the drift check against the stored predictions
6. Add an "Experiments" tab to `analysis/mm18_report.html`.

**[YOU] Gate 1b: review the results**
- [ ] Does the ground truth beat the model's own explanation? Is the gap largest for Gemma?
- [ ] Does the drift check look OK? If the new "own" accuracy is far from the stored accuracy, we report it and don't
      mix stored and new numbers.

## Step 2: experiment 2, re-paired counterfactuals

Candidates are already computed (`analysis/mm18_repair.py`, no API): about 276,000 new pairs built from scenarios each
model has already answered. The proposed sample is 226 single-feature changes in 4 groups: near-tie vs. clear,
crossed with whether the changed factor is cited or uncited.

**[YOU] Decisions**
- [ ] Accept that GPT-5.2's near-tie, uncited group is small (6 distinct original scenarios) *(yes, report it as
      small)*
- [ ] Cap traffic-light changes at half of each group *(yes; they're 154 of the 226 pairs as drawn)*
- [ ] No ground-truth condition in experiment 2 for now *(add it only if experiment 1 shows a clear gap)*

**[CLAUDE] Implementation**
1. Apply the traffic-light cap, redraw `experiments/moral_machines/mm18_repair_sample.csv`, and print the new group
   counts.
2. Add the sample to the report's tagging tab as a small "Check re-pairs" list showing original, counterfactual, the
   changed factor and the keyword verdict, so you can check pairs with one click each.

**[YOU] Gate 2a: hand-check 20 pairs (about 10 minutes)**
- [ ] For pairs the keyword match marks "cited", does the original's factor list really cite the changed factor?
      If more than 4 of 20 are wrong, I'll tighten the keyword map before running.

**[CLAUDE] Run**
3. Run 226 pairs × 2 conditions (own, none) × 2 simulators (about 900 calls).
4. Analyse:
   - NSG and flip rate per group, with CIs
   - **headline: near-tie vs. clear**, with the same answer-position line and prompts as experiment 1
5. Add the results to the "Experiments" tab.

**[YOU] Gate 2b**
- [ ] Is near-tie NSG below clear NSG on these purpose-built pairs too? This is the main replication of the write-up's
      §5.
- [ ] Is there a cited vs. uncited difference? It's fine if there isn't: the hand labels already suggested citation
      doesn't separate failures from successes.

## Step 3 (optional): resampling near-ties

Only do this if Christina isn't already resampling Moral Machines, and only if the budget allows.

**[YOU] Decisions**
- [ ] Ask Christina whether this overlaps her work.
- [ ] Choose the reference model: GPT-5.2 at medium reasoning *(proposed)*. These are reference-model calls, which
      cost more than simulator calls.

**[CLAUDE]**
1. Pick 40 GPT-5.2 pairs: 20 near-tie and 20 clear, including the 12 near-tie hand-labelled failures.
2. Resample the counterfactual scenario 5 times (200 calls), and the original if you want consistency on both sides
   (400 calls).
3. Report per pair: how often the model repeats its answer, and the "noise ceiling" (the best accuracy any predictor
   could reach) for near-tie vs. clear.

**[YOU] Gate 3**
- [ ] If near-tie answers are mostly unstable, the near-tie result is about non-determinism, and it goes in
      Christina's section. If they're stable, the explanations are missing information near the boundary.

## Step 4: write-up and report

**[CLAUDE]**
- Add experiments 1 and 2 (and 3 if run) to `moral_machines_writeup.md` §10, replacing "next experiments" with the
  results.
- Export the PNGs you choose for the midterm.

**[YOU]**
- [ ] Clear the remaining [CHECK] items in `moral_machines_writeup.md`.
- [ ] Take the write-up to Caleb and the team to decide placement in the shared report.

---

## Suggested timeline (to Nov 2)

| Day | You | Me |
|---|---|---|
| Mon Oct 27 | Key in `.env`; Gate 0 | Run script + smoke test |
| Tue Oct 28 | Gate 1a (texts), Gate 2a (hand check) | Ground-truth texts; experiment 1 and experiment 2 runs |
| Wed Oct 29 | Gates 1b, 2b | Analysis + report tab |
| Thu Oct 30 | Talk to Caleb / Christina about placement | (Experiment 3 if agreed) |
| Fri–Sat Oct 31–Nov 1 | Edit the write-up | Fold in results, export figures |
| Sun Nov 2 | Submit | — |

## Files

| What | Where |
|---|---|
| Detailed design | `moral_machines_groundtruth_plan.md` |
| Re-pair candidates and sample | `analysis/mm18_repair.py`, `experiments/moral_machines/mm18_repair_*.csv` |
| Run script (to write) | `analysis/mm18_gt_run.py` → `experiments/moral_machines/mm18_gt_predictions.csv` |
| Analysis (to write) | `analysis/mm18_gt.py` |
| Write-up | `moral_machines_writeup.md` |
