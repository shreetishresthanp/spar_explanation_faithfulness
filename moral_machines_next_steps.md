# Moral Machines experiment: plan using `data_parquet/` (no GPU)

**Goal:** Investigate the moral machines dataset to establish ground truth rules, then find counterfactuals near the decision boundary that show where models fail.

**Can `data_parquet/` alone support this goal? Yes.** I checked the numbers below on 14 reference-model configurations (as of 2026-10-02).
- **Ground truth rules:** For every model, a depth-4 decision tree on a few scenario features predicts its choices with 72–89% cross-validated accuracy, against 56–70% for a majority-class baseline. The top rules are consistent across models.
- **Counterfactuals near the decision boundary:** 666 of the existing pairs change only the head counts. In **191** of them at least one model flips its answer, and in **56** at least three models do. Those are boundary cases you already have.
- **Failure cases:** Predictor answers with and without the explanation are already in the files. When the reference model flips its answer, predictors match it only 50–79% of the time even with the explanation, compared with 78–90% overall.

The only part that `data_parquet/` can't cover is **testing new, hand-built boundary scenarios**. That's an optional final phase (Phase C) that uses OpenRouter, which needs no GPU.

Items marked **☐ REVIEW** need a decision from you.

---

## 1. The data

`data_parquet/` holds 8 complete parquet files (1.6 GB) in the repo schema (89 columns). Read only the moral machines rows with:
```python
pq.read_table(f, columns=[...], filters=[('original_dataset', '=', 'moral_machines')])
```

| File | Moral machines reference models (rows) | Pair set |
|---|---|---|
| `claude_4_5_predictions.parquet` | Haiku 4.5 (997), Opus 4.5 (1000), Sonnet 4.5 (1000), all at medium reasoning | A |
| `sonnet_itc_predictions.parquet` | Sonnet 4.5 at reasoning none/low/medium/high (1000 each) | A |
| `gemini_3_predictions.parquet` | Gemini 3 Flash (1000), Gemini 3 Pro (998) | A |
| `gpt_5_predictions.parquet` | GPT-5-mini (967), GPT-5-nano (365), GPT-5.2 medium (907; same as in the ITC file) | A |
| `gpt_5_2_itc_predictions.parquet` | GPT-5.2 at low/medium/high reasoning (878 / 907 / 891) | A |
| `qwen_3_predictions_latest.parquet` | Qwen3 0.6B–32B (747–1000 each) | B |
| `gemma_3_predictions.parquet` | Gemma-3 1B–27B (514–1000 each) | B |
| `extra_three_ref_models_predictions.parquet` | GPT-4o-mini, Gemini 2.0 Flash, GPT-5.2 with no reasoning | B |

What each row contains:
- the original scenario and its counterfactual
- the reference model's answer, explanation and reasoning trace on the original (`original_reference_response_parsed_response`, `_cot`)
- the reference model's answer on the counterfactual, which is the label the predictors try to match
- `counterfactual_predictor_response_{with,without}_explanation_predictor_answers` and `_predictor_names`: answers from 5 predictors (gpt-oss-20b, Gemma-3-27B, Qwen3-32B, Gemini 3 Flash, GPT-5-mini)

There are no `match_*` columns. Compute matches by comparing each predictor answer with `counterfactual_reference_response_answer`.

**There are two pair sets.** Compare models only within one set.
- **Set A** matches the committed `data/natural_counterfactuals/moral_machines_counterfactual_dataset_balanced.parquet`. Every row joins to `data/raw/moral_machines_raw.csv` by question index, and the raw CSV holds the structured features: the counts per character type, whether the victims are in the car, whether the car swerves, and the traffic lights.
- **Set B** (Qwen3, Gemma-3 and the extra three) matches `combined_dataset.parquet`. It comes from an older generation of the raw CSV, so only about 69% of its rows join. For set B, parse the features from the prompt text. The generator's wording is deterministic, and you can check the parser against set A.
- The two sets share only 142 pairs.

How the scenarios are built (in `tabular_utils.generate_moral_machine_scenarios`):
- Case 1 always kills group 1 (`count_dict_1`) and Case 2 always kills group 2.
- `is_in_car=True` means Case 1 kills the car's passengers.
- `is_interventionism=True` means Case 1 is the swerve option.

---

## 2. Preliminary findings (set A, 14 reference-model configurations)

| Model | Pairs | Majority baseline | Depth-4 tree (cross-validated) | Flip rate | Flip, counts-only pairs | Flip, other pairs | Predictor accuracy with explanation | Predictor accuracy without | Predictor accuracy with explanation, on flips |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.2 high | 891 | .700 | .880 | .22 | .02 | .40 | .899 | .839 | .77 |
| GPT-5.2 medium | 907 | .696 | .875 | .22 | .03 | .39 | .904 | .849 | .76 |
| GPT-5.2 low | 878 | .696 | .876 | .23 | .03 | .41 | .892 | .842 | .75 |
| GPT-5-mini | 967 | .698 | .848 | .25 | .09 | .40 | .850 | .820 | .67 |
| GPT-5-nano | 365 | .570 | .723 | .35 | .16 | .54 | .783 | .743 | .52 |
| Claude Opus 4.5 | 1000 | .614 | .855 | .31 | .05 | .55 | .884 | .839 | .73 |
| Claude Sonnet 4.5 medium | 1000 | .608 | .888 | .31 | .07 | .54 | .876 | .828 | .72 |
| Claude Sonnet 4.5, no reasoning | 1000 | .558 | .863 | .28 | .08 | .46 | .840 | .807 | .66 |
| Claude Haiku 4.5 | 997 | .643 | .849 | .28 | .15 | .40 | .778 | .762 | .50 |
| Gemini 3 Flash | 1000 | .573 | .867 | .34 | .05 | .62 | .888 | .841 | .79 |
| Gemini 3 Pro | 998 | .557 | .826 | .36 | .08 | .61 | .872 | .819 | .76 |

Sonnet 4.5 at low and high reasoning looks like Sonnet medium. The predictor accuracies average all 5 predictors for each pair.

### Finding 1: A few structural rules explain most of the choices
A depth-4 tree on structural features gets 72–89% accuracy for every model. For GPT-5.2, the splits rank the factors in this order:
1. **Don't kill people crossing legally.** In pedestrian-only scenarios with traffic lights, red-light jaywalkers are the ones killed about 93% of the time.
2. **Save more humans.**
3. **Sacrifice passengers rather than pedestrians**, unless the passengers outnumber the pedestrians or the pedestrians are pets.
4. **Prefer not to swerve** when the other factors are tied.

Gender, age, social status and fitness barely show up, even though they define about 64% of the scenarios. Confirm this ordering for each model in Phase A.

### Finding 2: Most existing pairs aren't controlled boundary probes
The generator pairs two scenarios when they have the same set of feature names but different counts. It doesn't control for:
- which side has the red light
- which group is in Case 1

In pairs that change only the counts, models flip **2–16%** of the time. In pairs that also swap a light or swap which group is in Case 1, they flip **39–62%** of the time. Most flips therefore come from large, uncontrolled changes.

The counts-only flips are the genuine near-boundary cases. There are 191 such pairs, mostly social-value (86) and gender (54) scenarios.

### Finding 3: Predictors fail mostly when the reference model flips
Overall, predictors match the reference model 78–90% of the time with the explanation, but only 50–79% of the time on pairs where the reference model flipped its answer. For each model there are also:
- 55–202 pairs where most predictors are wrong even with the explanation
- 9–34 pairs where the explanation makes the predictors worse (most right without it, most wrong with it)

These are the failure cases to study.

### Finding 4: The moral machines prompt conflicts with itself, and some GPT-5.2 rows are missing
The raw scenario text says *"Respond with one word only"*, but the template asks for `[EXPLANATION]`, `[ANSWER]` and so on. GPT-5.2 sometimes replied with a bare answer, and the pipeline dropped those rows. Only GPT-5.2 is affected: it has 878–907 pairs instead of 1000, and the missing rows come almost entirely from answer-first prompts. GPT-5.2 has about 400 answer-first pairs, compared with 489 for Claude.

---

## 3. Recommended approach

### Phase A: Establish ground truth rules (laptop only, free)
1. Build a scenario-level feature table. For set A, join to `moral_machines_raw.csv` by index. For set B, write a prompt-text parser and check it on set A.
2. For each reference model, fit interpretable rules (a shallow tree or an ordered list of rules) and write them out as code, for example `mm_rules.py: predict(model, scenario) -> CASE 1/2, margin`.
3. Compare the rules across models and with the human preferences reported by Awad et al. (2018).
4. Check the fitted rules against what each model says in its explanation (`[MOST_IMPORTANT_FACTORS]` in the parsed response).

### Phase B: Mine boundary cases and failures from the existing data (laptop only, free)
1. Start with the 191 counts-only pairs where a model flips, particularly the 56 where three or more models flip.
2. For each, label the rule that was crossed and its margin under the Phase A rules.
3. Measure failures on these pairs:
   - predictors wrong even with the explanation
   - the explanation makes predictors worse
   - the reference model's counterfactual answer contradicts the rule stated in its own explanation
4. Report separately for the uncontrolled pairs (a light or group swap). They mix in large changes and shouldn't count as boundary evidence.

### Phase C (optional): New hand-built boundary scenarios (no GPU, OpenRouter cost)
Use this only if Phase B's 191 pairs aren't enough, for example because utilitarianism and species have only 5 and 4 pairs.
1. Write a **deterministic** scenario builder: a variant of `generate_moral_machine_scenarios` that takes its settings as arguments instead of sampling them.
2. Generate single-factor edits around the near-tie scenarios.
3. Run them through `generate_reference_answers` → `generate_predictor_answers` → `analyze_simulatability` with `openrouter/...` models, for example `openrouter/openai/gpt-5.2-low`.

---

## 4. Items that need your review

- ☐ **REVIEW: How to define "ground truth."** Options:
  - (a) each model's empirical rules from Phase A (recommended)
  - (b) a normative rule set you write yourself, such as the Awad et al. preferences
  - (c) both, and treat the differences as a finding
- ☐ **REVIEW: Which models to study.** Set A (frontier models; matches the committed moral machines data) is recommended. Set B (scaling within Qwen3 and Gemma-3) needs the prompt parser. You can also do both, analyzed separately.
- ☐ **REVIEW: The GPT-5.2 skew (Finding 4).** Accept it, or limit cross-model comparisons to answer-last pairs?
- ☐ **REVIEW: The definition of "near the boundary."** The current definition is counts-only pairs where some model flips (191). Alternatives are pairs where the rule margin is small in Phase A, or pairs where the 5 predictors disagree.
- ☐ **REVIEW: Is Phase C needed?** Decide after Phase B. It needs an `OPENROUTER_API_KEY` in `.env` (none is set). 500 new pairs × (1 reference + 2 prompts × 5 predictors) comes to roughly 5,500 calls.
- ☐ **REVIEW: The one-word prompt conflict in new runs.** If you run Phase C, removing "Respond with one word only" improves the explanations, but the new results won't be exactly comparable with `data_parquet/`.
- ☐ **REVIEW: Uncommitted data changes.** All 7 `*_balanced.parquet` files in `data/natural_counterfactuals/` have uncommitted changes. The moral machines question pairs are identical to the committed version, but check the others before you commit.
- ☐ **REVIEW: Keep `data_parquet/` and `data_parquet.zip` out of git.** They total about 3 GB and are untracked but not ignored. Add them to `.gitignore`?

## 5. Reproducing these numbers
The tables above came from ad-hoc scripts that aren't in the repo. Phase A step 1 should turn them into a notebook or script under `analysis/`, so the numbers can be regenerated from `data_parquet/`.
