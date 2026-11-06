# Moral Machines × WS2: status report and plan

*Editable working doc. Snapshot of 2026-10-07; midterm / blog draft due **2026-11-02**. Tick boxes, fill in `> Notes:` blocks, and strike out what you drop.*

- **Scope:** the paper's **18 reference models** (Fig. 2), one config each:
  - **10 open-weight:** Qwen 3 ×6, Gemma 3 ×4
  - **8 proprietary:** GPT-5 Nano, GPT-5 Mini, GPT-5.2, Claude Haiku/Sonnet/Opus 4.5, Gemini 3 Flash/Pro
- **Predictors:** all 18 models are scored by the same 5 (gpt-oss-20b, Qwen3-32B, Gemma-3-27B, Gemini 3 Flash, GPT-5-mini).
- **Reasoning level:** GPT-5.2 and Sonnet 4.5 use **medium**, an assumption, because the paper doesn't say.
- **Out of scope:** the other reasoning levels, GPT-4o-mini, Gemini 2.0 Flash and GPT-5.2-none.
- **Provenance:** all numbers come from `analysis/mm18.py` on `data_parquet/` (2026-10-07). The interactive version is `analysis/mm18_report.html`.
- **Older material:** the earlier 13/26-config analyses are in `analysis/_archive/`. `moral_machines_README.md` holds the older detail and literature.

---

## 1. Where this fits

- **WS1:** "Toy Models of Simulatability". They need a principled account of what makes a counterfactual "too far away", and of how counterfactual simulatability fails.
- **WS2 (me):** counterfactual construction, i.e. building counterfactuals *from the explanation*.
- **Why Moral Machines (MM):**
  - Its generator is fully known. Every scenario is a handful of named factors: law, passengers vs. pedestrians, intervention, head counts and theme.
  - So a counterfactual can change *exactly* the factor an explanation cites. That makes MM a semi-toy setting: real behaviour, controlled inputs.

### How MM counterfactuals are built
1. 15,000 random scenarios are generated.
2. They're grouped when they have the same character types and yes/no settings (car, swerve, lights shown).
3. Two with different head counts are paired.

**Not checked:** which side has the red light, which group is on which side, and the theme (10% of pairs change theme, against the paper's Appendix A.8). So a "counterfactual" ranges from +1 character to lights flipped and sides swapped. There's no Hamming-distance limit.

### Two pair sets
- The **proprietary** models were run on one set of 1,000 MM pairs (**set A**), and the **open-weight** models on another (**set B**).
- They share only **142** pairs, so compare the groups model by model, not pair by pair.

---

## 2. Key results (18 models)

### 2.1 The low MM NSG comes from the open-weight models
| Reference models | MM NSG | Same models, tabular mean |
|---|---|---|
| Paper, 18-model average (reported) | 6.0% | — |
| Ours, 18-model average | 7.2% | 32.3% |
| 8 proprietary | **23.9%** | 32.9% |
| 10 open-weight | **−6.1%** | 31.7% |

- For GPT-5.2, Opus 4.5 and Gemini 3 Flash/Pro, MM NSG is within 1 point of the model's tabular mean. MM isn't an outlier for them.
- **Larger gaps among proprietary models:** Haiku 4.5 (−24 pts), GPT-5 Nano and Mini (−20) and Sonnet 4.5 (−7).
- **Every open-weight model drops:** Qwen 3 by 23–29 points, Gemma 3 by 42–69.
- ☐ **Inspect:** the remaining gap between 7.2% and 6.0%. The likely causes are a different reasoning level for GPT-5.2 or Sonnet, or different parse handling.

### 2.2 Raw counterfactual distance is the wrong axis (core WS2 evidence)
| What changed | Share of pairs | Proprietary: flip / NSG | Open-weight: flip / NSG |
|---|---|---|---|
| counts only (closest) | 48% | 8% / **−8%** | 20% / **−17%** |
| side swapped | 18% | 21% / **46%** | 23% / 6% |
| light swapped | 17% | 79% / 25% | 48% / 5% |
| light + side swapped | 6% | 63% / 32% | 39% / 0% |
| theme changed | 10% | 48% / 23% | 44% / 8% |

- **The closest pairs carry no signal.** The model barely flips, so there's nothing for an explanation to predict.
- **For proprietary models the gain comes from changes that touch factors explanations discuss:** sides (passengers vs. pedestrians, intervention) and the law.
- **So what matters is whether the change hits a cited factor and crosses the decision boundary, not its raw size.**
- The open-weight figures use the ~2/3 of their rows that link to the raw generator CSV.

### 2.3 Failure modes (a taxonomy for WS1's toy model)
| Failure mode | Evidence |
|---|---|
| **Too close:** no flip, so no signal | counts-only NSG < 0 for both groups (§2.2) |
| **Near-tie / noise** | Of the 11 proprietary shared-failure pairs, 73% split the models (30–70% flip), against 13% of other pairs (§2.5) |
| **Out of scope:** the change isn't covered by the explanation | theme-changed pairs; needs a validated tagger |
| **Hidden cause** | Gemma 3 27B picks Case 1 on 96% of answer-last prompts vs. 59% answer-first. Its explanations argue from content. Not a parser artifact. Proprietary models show no format effect. |
| **Genuine principle violation** | Spot-checked GPT-5.2 and Opus cases (e.g. "age is not a tie-breaker", then age decides); the paper's Fig. 20 |
| **Construction artifact** | 10% of pairs change theme; some pairs put a cat *inside the car* |

### 2.4 Misleading explanations
- **Definitions:**
  - *Egregious*: every predictor is wrong **with** the explanation. This is the paper's definition; it ignores the without-explanation condition.
  - *Explanation-misled*: egregious **and** 3+ of 5 predictors right **without** the explanation.
- **Proprietary:** 339 egregious rows (4.9%), of which **39 are misled** (0.6%; 12%). That's 0–7 per model, and 0 for GPT-5 Nano.
- **Open-weight:** 1,028 egregious rows (12.4%), of which **85 are misled** (1.0%; 8%).
- **Most egregious rows are hard rather than misleading:** for 65% of them, every predictor was also wrong *without* the explanation.
- **Direction:**
  - Proprietary is balanced: in 18 rows the explanation predicted a flip that didn't happen, and in 21 it missed one.
  - Open-weight is mostly invented flips (57 of 85). That's what you get when answers ignore content while explanations argue from it.
- **Where (proprietary):** side swaps (22 of 39) and light + side swaps (8). Counts-only pairs have almost none (2).
- **Predictors:** for proprietary models, every predictor is helped more than harmed (net +3.6% to +4.9%). For open-weight models, harm exceeds help for 3 of 5 predictors.

### 2.5 Shared failures: no pair fails for most models
| View | Pairs | Never egregious | Fail for 3+ models | Worst pair | Shared pairs where models split |
|---|---|---|---|---|---|
| 8 proprietary (set A) | 1,000 | 746 | **11** | 4 of 7 | 73% (vs. 13% of others) |
| 10 open-weight (set B) | 1,000 | 370 | 107 | 5 of 8 | 92% (vs. 34%) |
| All 18, on the 142 shared pairs | 142 | 47 | 15 | 6 of 17 | 33% (vs. 15%) |

- For proprietary models, shared failures are rare (11 pairs) and look like near-ties.
- The open-weight failures are many, and they overlap with the position bias.

### 2.6 Data coverage
- **Usable MM pairs per model (of 1,000):** GPT-5 Nano has only **172**; Gemma 3 4B 514; Gemma 3 12B 533; Qwen 3 0.6B 612 and 1.7B 634; GPT-5.2 840; GPT-5 Mini 931. The rest have 982–1,000.
- Treat the small-sample models' numbers with care.

---

## 3. Flags: things I need to inspect

- ☐ **Spot-check the 11 proprietary shared-failure pairs.** HTML › *Shared failures* › 8 proprietary view. Save with *Download my labels*.
  > Notes:
- ☐ **Spot-check the 39 proprietary explanation-misled rows.** HTML › *Misleading explanations* › Group: proprietary.
  > Notes:
- ☐ **Tag 100 explanations for cited factors.** HTML › *Tag explanations*. The sample is in `experiments/moral_machines/mm18_tag_sample.csv`. Download the labels as `mm18_tag_labels.csv` into the same folder. These become the answer key for the automatic tagger (κ). The tab also has a **Failure cases** set (70 explanations behind the proprietary misled rows and shared-failure pairs), with buttons that open each one in the other tabs. Use it to check the failures yourself; it doesn't count towards κ. The old regex flagged "law" in 89% of explanations. *This blocks the §2.2 claim.*
- ☐ **The 7.2% vs. 6.0% gap** (§2.1).
- ☐ **Questions for HM:**
  - Are the theme-changing pairs (10%) intended?
  - Why are Gemma answer-first rows missing (4B has none, 12B has 19)?
  - Which reasoning level do GPT-5.2 and Sonnet 4.5 use in the main results?
- ☐ **Dimension skew in the pairs** (gender 32%, species 5%) *[unverified]*. Re-check it.
- ☐ **Housekeeping:** 7 `*_balanced.parquet` files have uncommitted changes. Add `data_parquet/` to `.gitignore`.

---

## 4. Research agenda

**Central question (WS2):** what makes a counterfactual *informative* for testing an explanation? Do counterfactuals built from the explanation give a sharper faithfulness signal than generic perturbations?

- **H1, explanation-relative distance.** NSG and flip rate depend on (a) whether the counterfactual changes a factor the explanation *cites*, and (b) whether it crosses the model's decision boundary. Raw distance predicts them poorly.
- **H2, explanation-derived counterfactuals.**
  - Changing a *cited* factor should give higher NSG than random changes.
  - Changing a *non-cited* factor tests critical omissions.
  - Both expose principle violations more efficiently.
- **H3, the noise ceiling.** Near-boundary counterfactuals are partly coin flips. Resampling (K=5) bounds the reachable NSG.
- **Controls:**
  - The proprietary models are the main subjects; they share one pair set, so comparisons are clean.
  - Gemma 3 is a known-unfaithful control.
- **Bridge to WS1:** a per-model fitted decision rule over MM factors (logistic / AMCE) is an interpretable "toy reference model". It defines "too far" formally, e.g. P(crossing the boundary).

> Notes / changes to the agenda:

---

## 5. Plan to the midterm (2026-11-02)

### Week 1 (Oct 7–13): free, existing data
- ☐ Spot-check the proprietary shared-failure pairs and misled rows (§3).
- ☐ Validate or replace the tagger: 100 hand labels, or a small LLM tagger.
- ☐ Fit a per-model decision rule (logistic / AMCE) for the 8 proprietary models.
- ☐ For each pair: cited factor changed? × crosses the fitted boundary?
- ☐ **Key figure:** NSG and flip rate on these two axes, next to raw distance (pair type). *This tests H1.*
- ☐ Meet WS1: can the fitted rule serve as their toy reference model? What do they need from WS2?

### Week 2 (Oct 14–20): build the WS2 method
- ☐ A deterministic MM generator: all factors as arguments, counterbalanced case order, and the "one word only" prompt conflict fixed.
- ☐ Pipeline: explanation → cited factors (LLM extraction) → counterfactuals that:
  - (i) change a cited factor
  - (ii) change a non-cited factor
  - (iii) make a random change, as in the paper
- ☐ Set up `OPENROUTER_API_KEY`. Add a `--reference-repeats` option, or run K times with different seeds. The generator deduplicates prompts, and the Gemma config pins `seed=666`.

### Week 3 (Oct 21–27): pilot (about 1.5–2k API calls)
- ☐ About 50 anchors × 3 counterfactual types.
- ☐ Reference models: GPT-5.2 (medium) and Claude Sonnet 4.5 (medium), plus Gemma 3 27B as the control. K=5 samples each.
- ☐ 2–3 predictors.
- ☐ Compare NSG across counterfactual types (H2) against the noise ceiling (H3).

### Week 4 (Oct 28 – Nov 2): write
- ☐ WS2 section:
  - the construction explainer
  - the H1 figure
  - pilot results
  - the failure taxonomy with examples
- ☐ Send HM the §3 questions.

> Notes / schedule changes:

---

## 6. Decisions needed from you
- ☐ Is H1/H2 (explanation-relative distance, and explanation-derived counterfactuals) the WS2 midterm contribution?
- ☐ API budget for the week-3 pilot: about 2k calls.
- ☐ Pilot reference models: GPT-5.2 + Sonnet 4.5 + Gemma 3 27B?
- ☐ Should the fitted decision rule be WS1's toy model? Agree with WS1.

---

## 7. Where things are
| What | Where |
|---|---|
| Data, definitions and metrics for the 18 models | `analysis/mm18.py` (run it to print a summary; writes `experiments/moral_machines/mm18_*.csv`) |
| Interactive report: Overview, What drives low MM NSG, Misleading explanations, Shared failures (with spot-check browsers) | `analysis/mm18_report.html`; rebuild with `python analysis/mm18_report.py` |
| Earlier analyses (13/26 configs, raw-data plots, old report) | `analysis/_archive/`, `experiments/moral_machines/_archive/`, `figures/moral_machines/_archive/` |
| Older detailed notes and literature | `moral_machines_README.md` |
| Paper | Mayne et al. (2026), [arXiv 2602.02639](https://arxiv.org/abs/2602.02639) |
