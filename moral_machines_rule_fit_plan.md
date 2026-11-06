# Moral Machines: per-model decision-rule fit (draft plan)

*Draft, 2026-10-27. Part of `moral_machines_6h_plan.md` (the "in parallel" block). Nothing has been run yet.
Items marked **[REVIEW]** need your decision or a check before I execute. Reply "go", or give changes.*

## Why

The rule makes Harry's "ground-truth explanation" suggestion quantitative, and it gives WS1 an empirical "known rule" for
each real model. It answers three questions without any new API calls:

1. **What actually drives each model's choices?** Coefficients on the scenario factors, AMCE-style.
2. **Rule ceiling:** how much of the model's predictable behaviour does its explanation recover? Compare the rule's
   accuracy on counterfactual answers with predictor accuracy with and without the explanation.
3. **Distance to the decision boundary:** a formal "too close / near-tie" axis. Plot NSG and flip rate by the rule's
   predicted P(flip), next to the existing pair-type breakdown.

## Data

- **Source:** existing predictions in `data_parquet/` (set A), loaded through `analysis/mm18.py` (`load_mm`), and
  scenario factors from `data/raw/moral_machines_raw.csv` (every set-A row links to it).
- **Unit:** one decision per scenario. Each pair gives two (original + counterfactual), and no scenario appears in more
  than one pair for a given model. So GPT-5.2 has about 1,680 decisions and Gemini 3 Flash about 2,000.
- **Target:** y = 1 if the model chose **Case 1**.
- **Models:**
  - the 8 proprietary models: GPT-5 Nano, GPT-5 Mini, GPT-5.2 (medium), Claude Haiku/Sonnet/Opus 4.5 and Gemini 3
    Flash/Pro, all at medium reasoning
  - Gemma 3 27B as the known-unfaithful control (667 linked pairs, from set B)
- **[REVIEW]** GPT-5 Nano has only 172 pairs (344 decisions). Keep it, but flag its fits as low-power? *(Proposed: keep and flag.)*

## Features (main-effects model)

All features describe **Case 1 relative to Case 2**, so a positive coefficient means "more likely to choose Case 1". By
the generator's construction, Case 1 kills set 1 and spares set 2.

| Feature | Encoding | Explanation factor it matches |
|---|---|---|
| `lives_saved` | (people killed in Case 2) − (people killed in Case 1); animals counted separately in `pets_saved` | Number of lives |
| `c1_swerves` | +1 if Case 1 is the swerve, −1 if it's staying in lane | Intervention |
| `c1_kills_passengers` | +1 if Case 1 kills the car's passengers (`is_in_car`), else 0 | Passengers vs. pedestrians |
| `c1_spares_lawful` | light of Case 2's victims minus light of Case 1's victims; green = +1, red = −1, no light = 0 | Law / traffic light |
| `theme_<dim>` (one per theme) | +1 if Case 1 kills the theme's first group (e.g. the younger, the female, the humans), −1 if the second; 0 for other themes | Age, gender, species, social status, fitness, utilitarianism |
| `answer_first` | prompt format (+1 answer first, −1 answer last) | none, a hidden cause |
| intercept | baseline Case 1 preference | none (position bias) |

- **[REVIEW] Theme sign convention:** for each theme, the first group is the first entry in the generator's
  `scenario_dimension_group_types` (species: human; social value: lower; gender: female; age: younger; fitness: lower;
  utilitarianism: less). Only the sign of the coefficients flips if you'd rather use "the group a person would usually
  save", so it's a readability choice. *(Proposed: generator order, with plain-language labels in the plots.)*
- **[REVIEW] Character-level detail:** an extended model adds the count difference for each of the 20 character
  types (baby, criminal, doctor, …) with stronger L2 regularisation. More accurate, but harder to read and closer to
  overfitting with about 1,700 decisions. *(Proposed: report main effects as the headline, and the extended model only
  as a robustness check on the ceiling.)*
- **To verify before fitting:** whether the counterfactual prompt uses the same answer position as the original.
  `original_answer_first` is the only flag in the data. If they can differ, I'll parse the position from the prompt text.

## Fit and validation

- **Model:** L2-regularised logistic regression (scikit-learn 1.7.2; statsmodels isn't installed, so there are no
  p-values). Standardise the features.
- **Cross-validation:** 5-fold `GroupKFold`, grouped by pair, so a pair's two scenarios are never split across train
  and test.
- **Coefficient uncertainty:** 95% intervals from a 200-replicate pair-level bootstrap.
- **Held-out accuracy** against two baselines: always choose the model's majority case, and intercept + `answer_first`
  only (position alone).
- **Sanity checks (pass/fail):**
  - the `answer_first` coefficient is about 0 for GPT-5.2 and Gemini 3 Flash, and large for Gemma 3 27B
  - the position-only baseline is close to the full model for Gemma 3 27B and far below it for the proprietary models
- **[REVIEW]** If a sanity check fails, I'll stop and report before making figures. OK?

## Outputs

### 1. Rule ceiling (per model)
For each pair, predict the counterfactual answer with the rule fitted on folds that don't contain that pair. Compare:

| | Accuracy on the counterfactual answer |
|---|---|
| Predictors without explanation | existing |
| Predictors with explanation | existing |
| **Fitted rule** (scenario factors only) | new |
| **Rule + original answer** (the rule's P(flip), combined with the known original answer) | new |

Also a **rule NSG** = (rule accuracy − accuracy without) / (1 − accuracy without), on the same scale as explanation NSG.

- **[REVIEW] Caveat to accept:** this is not a strict ceiling. The rule is fitted on about 1,300 of the model's own
  decisions, and predictors see one explanation. Read it as "how predictable the model is from scenario factors alone".
  If the explanation scores near the rule, it carries most of the factor-level information; if it scores well below,
  information is missing from the explanation.

### 2. Boundary figure
- For each pair: **predicted P(flip)** = the rule's probability that the counterfactual answer differs from the
  original answer, plus the **counterfactual margin** = |P(Case 1 | counterfactual) − 0.5|, where small means a near-tie.
- Plot NSG and flip rate by P(flip) quintile (5 bins), with pair-level bootstrap CIs, for GPT-5.2, Gemini 3 Flash and
  all proprietary models pooled. Put the existing pair-type breakdown beside it.
- **[REVIEW] Near-tie threshold:** a counterfactual margin below 0.2, i.e. P(Case 1) between 0.3 and 0.7. *(Proposed;
  I'll also report how much the results change at 0.1 and 0.3.)*
- **Expected result** (it can fail): counts-only pairs fall in the low P(flip) bins, and NSG rises with P(flip)
  regardless of pair type.

### 3. Join to the coverage sample
Add `rule_p_flip`, `rule_margin` and `near_tie` to the 60 coverage-sample rows. Then, once your labels arrive, the 2×2
(in scope × crosses the boundary) in the 6-hour plan needs no extra work.

## Where things go

- **Code:** `analysis/mm18_rule.py`, which reuses `mm18.load_mm`.
- **Data:** `experiments/moral_machines/mm18_rule_coefs.csv`, `mm18_rule_ceiling.csv`, `mm18_rule_pairs.csv` (one row
  per (model, pair) with P(flip) and margin).
- **[REVIEW] Figures:** a new **"Decision rule"** tab in `analysis/mm18_report.html` (coefficients, ceiling table,
  boundary figure), or a separate page? *(Proposed: a new tab.)*
  - Rebuilding the report **keeps your coverage labels.** The row ids are fixed by seed, and the labels live in your
    browser keyed by those ids, as long as you open the file from the same path.
  - Still, download your labels CSV before I rebuild, just in case.

## Time
About 45–60 minutes to run and check, plus about 15 minutes to add the report tab.

## Review checklist
- [ ] Keep GPT-5 Nano, flagged as low-power
- [ ] Theme sign convention: generator order
- [ ] Main effects as the headline; character-level model as a robustness check only
- [ ] Stop and report if a sanity check fails
- [ ] Accept the "not a strict ceiling" caveat
- [ ] Near-tie threshold 0.2 (with 0.1/0.3 sensitivity)
- [ ] Figures in a new report tab
