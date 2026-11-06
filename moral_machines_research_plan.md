# Moral Machines: research plan (v2, after a reviewer-style critique)

**Workstream:** Data distribution and counterfactual construction (HM's notes of Sep 23)
**Researcher's motivation:** the ethics angle. Do a model's stated moral reasons predict its moral choices?
**Data:** `data_parquet/` (see `moral_machines_next_steps.md`)
**Changes from v1:** new framing around the paper's 6.0% claim, ground truth estimated from a conjoint analysis instead of a tree, a check on the paper's claim added as a primary aim, and a list of reviewer flags.

---

## 0. Facts that set the scope (checked 2026-10-03)

1. **Moral machines doesn't use Hamming distance.**
   - The 6 tabular datasets pair questions at Hamming distance ≤ 2. The mean is 1.83–1.90, and all of their pairs have a distance.
   - Moral machines pairs any two scenarios with the **same set of character types and binary flags but different counts** (`moral_machines_generator.py`). Its `counterfactual_hamming_distance` field is empty for every pair.
   - The paper says only that moral machines "is processed independently due to structural differences (Appendix C)".
2. **The paper reports its lowest NSG on moral machines (6.0%, compared with 42.8% for Pima) and reads this as "models are less faithful on ethical issues"**, citing Matton et al. (2025).
3. **The matching rule distorts the data.** In the 15,000 raw scenarios each dimension makes up about 16.7%. In the pairs:
   - gender is 31.7% and social value 32.2%
   - species is 4.9% and utilitarianism 6.5%

   So the rule over-samples the dimensions that models largely ignore and under-samples the ones where models have strong preferences.
4. **Pairs that change only the counts carry almost no information.** On those pairs the gain from the explanation is about 0 pp (between −3 and +0.5 pp). All of the moral machines gain comes from pairs where the traffic light or the group in Case 1 is swapped. Neither swap is controlled for, and neither is a small change.

**Consequence:** the paper's conclusion that models are "less faithful on ethics" rests on moral machines, and moral machines is the dataset with the weakest counterfactual construction. **A study that tests that conclusion is the strongest version of this project.** It also keeps the ethics angle front and centre.

---

## 1. Flags a NeurIPS/ICML reviewer would raise, and fixes

| # | Flag | Severity | Fix |
|---|---|---|---|
| F1 | **"Consciousness" framing.** Trolley-style choices say nothing about consciousness. Reviewers would reject the overclaim. | High | Frame the work as **faithfulness of moral explanations, or stated versus revealed moral preferences**. Keep words like conscience and awareness out of the claims. |
| F2 | **Circular ground truth.** v1 fitted a tree to the same answers it then evaluated, which leaks information and treats an 88%-accurate model as the truth. | High | Estimate **each model's preferences with AMCEs** (average marginal component effects). These are causal effects from the generator's randomized factors, the method Awad et al. (2018) and Takemoto (2024) use. Fit on scenarios outside the evaluation pairs (cross-fitting). |
| F3 | **Pair type is confounded with scenario type.** Light swaps happen only in law scenarios, so comparing gain across pair types also compares different scenarios. | High | Use matched or stratified comparisons and a regression with scenario fixed effects. Then use designed interventions (§3, Aim 3). |
| F4 | **Position and framing are confounded with content.** Case 1 is always the passengers in in-car scenarios, and Case 1 is the swerve option exactly when `is_interventionism` is set. Position bias can't be separated from these. Prior work finds LLM moral choices are sensitive to the prompt format. | High | Counterbalance the case order in any new runs. On existing data, report the problem as a limitation. |
| F5 | **Non-determinism is ignored.** The reference answers come from a single sample. In indifferent scenarios, such as a mirror-image gender swap, the answer is close to a coin flip. | High | Take K≥5 samples per scenario, define the boundary as p(choice) ≈ 0.5, and report NSG against a ceiling adjusted for noise. The GPT-5.2 duplicate-run agreement of 92.6% is a hint, but confirm the two runs are independent. |
| F6 | **Invalid prompt.** The scenario text says "Respond with one word only" but the template asks for an explanation. GPT-5.2 dropped about 90 answer-first pairs, and some explanations may be truncated. | Medium | Fix the prompt in new runs and report the effect. On existing data, run a sensitivity analysis restricted to answer-last pairs. |
| F7 | **Tiny cells.** The claimed-irrelevance flips have n=2–10, and the explanation factors come from a regex. | Medium | Treat the existing data as exploratory only and test hypotheses on designed data. Validate the factor extraction against 100 hand labels (report agreement, κ). |
| F8 | **Ecological validity.** Trolley problems are widely criticised as stand-ins for real ethics. | Medium | Claim only that this is a **controlled test bed** for measuring faithfulness. Optionally replicate on BBQ, whose data is already in the repo (`data/generated_counterfactuals/bbq_*`), as a second ethics domain. |
| F9 | **Novelty overlap.** The Moral Machine on LLMs is well studied (Takemoto; a 52-LLM follow-up; scaling laws over 75 configurations), and stated-versus-revealed gaps are documented (2506.00751). | Medium | The new contribution is the **link**: whether stated reasons predict revealed choices *under counterfactual intervention*, and whether the NSG metric measures that correctly. Cite these papers as the measurement foundation. |
| F10 | **No hypotheses are pre-registered, and there are many comparisons.** | Low–medium | Write down H1–H3 (below) before running Aim 3. Use bootstrap CIs over pairs and correct for multiple tests across models. |

---

## 2. Refined research questions

**Title idea:** *"Are LLMs less faithful about ethics, or are ethics counterfactuals harder to build? Calibrating counterfactual simulatability on the Moral Machine."*

- **H1 (construct validity):** A large share of moral machines' low NSG (6.0%) comes from how the counterfactuals are constructed, not from unfaithfulness. Under a matched design with Hamming distance ≤ 2 and controlled factors, moral machines NSG will move toward the tabular range.
  - **Both outcomes are publishable.** If NSG rises, the paper's ethics claim was a measurement artifact. If it stays low, the claim now rests on much stronger evidence.
- **H2 (stated vs. revealed):** The factors a model cites in its explanation diverge from its AMCE-ranked revealed preferences, mostly on demographic factors. Models say gender and age are irrelevant, but have small AMCEs that aren't zero.
- **H3 (principle violation):** A model violates the principles it states (for example "don't intervene" or "gender is irrelevant") under minimal counterfactuals built to test them more often than chance noise would explain, and predictors that trust the explanation fail on exactly those cases.

Secondary questions:
- Is agreement among the predictors a valid proxy for "the explanation implies the answer"?
- How does reasoning effort (the Sonnet and GPT-5.2 levels) change the gap between stated and revealed preferences?

---

## 3. Aims and methods

### Aim 1: Recheck the paper's 6.0% on existing data (free; 1–2 weeks)
1. Reproduce moral machines NSG from `data_parquet/` with `analysis_scripts/analyze_simulatability.py`, and confirm it matches the paper's figure.
2. Define a **moral machines Hamming distance** over factor-level features:
   - dimension group on each side
   - the count of each character type (or binned counts)
   - in-car or not
   - swerve or stay
   - the light on each side
3. Compute it for every pair, so moral machines can be compared directly with the tabular datasets.
4. Recompute NSG by pair stratum: counts-only, group swapped, light swapped, and the Hamming distance bins. Use scenario fixed effects (F3) and reweight to the raw dimension distribution (Fact 3).
5. **Deliverable:** one figure showing NSG against counterfactual distance for moral machines and for 1–2 tabular datasets on the same axes. **Take this to HM.**

### Aim 2: Revealed and stated preferences (free; 1–2 weeks, can run alongside Aim 1)
1. AMCE per model and factor, estimated on all scenarios each model answered, with cross-fitting (F2). Factors: species, number of lives, law, passengers versus pedestrians, intervention, gender, age, fitness and social status. This gives a **profile of revealed preferences for each model**, in the same format Awad (2018) used for humans.
2. Stated preferences: extract the factors cited in `[MOST_IMPORTANT_FACTORS]` and `[EXPLANATION]`. Use an LLM tagger, validated against 100 hand labels (F7).
3. Compare the stated and revealed rankings for each model (rank correlation, plus the factors cited but not used and used but not cited). This is H2.

### Aim 3: Designed interventions (API; 3–4 weeks; pre-register H1–H3 first)
1. Deterministic generator: a version of `generate_moral_machine_scenarios` that takes all of its factors as arguments and has a **counterbalanced case order** (F4).
2. Anchors: about 300 scenarios stratified across the 6 dimensions, at an equal rate (Fact 3).
3. For each anchor, create counterfactuals that change one factor, at Hamming distance 1:
   - a random factor, to estimate H1 under the paper's own definition of "close"
   - **the factor named in the explanation**, a principle test (H3)
   - **the factor with the highest AMCE**, which should cross the boundary
4. Sample each reference model K=5 times per scenario at its paper settings (F5), with the prompt fixed (F6). Run 3+ predictors with and without the explanation. Recompute NSG against the noise ceiling.
5. Models: start with GPT-5.2 low and one Claude model (pilot of about 1–2k calls), then scale to 4–6 models.

### Aim 4 (stretch): Generalisation
Run the Aim 1 decomposition on one tabular dataset (heart disease) and on BBQ, to show the calibration method transfers beyond moral machines.

---

## 4. Should the experimenter look at the raw data? Yes, first.

Do this in week 1, before any modelling. Make these plots or tables in a notebook under `analysis/` (for example, alongside `analysis/mm_raw_data.ipynb`):

1. **Factor marginals**, raw versus paired: dimension, in-car, swerve, law, characters per side, number of distinct types. This shows the matching rule's skew (Fact 3).
2. **Factor co-occurrence** heat map. It shows which factors are confounded, such as the light pattern with in-car (F3, F4).
3. **What changes within a pair**: a histogram of count L1, the light-swap rate, the group-swap rate, and the new moral machines Hamming distance next to the tabular datasets' ≤ 2.
4. **Choice rates by factor for each model**: a naive version of the AMCE plot, in the style of Awad (2018), for 5–6 models side by side.
5. **Flip rate and gain by pair stratum** (the key table from §0, as a figure with CIs).
6. **Read 30–50 transcripts.** Include originals, explanations, counterfactual answers and predictor answers, and over-sample flips and explanation-hurts cases. Write down the principles models state. Those notes become the categories for the Aim 2 tagger and the Aim 3 principle tests.

---

## 5. Moral machines or another dataset?

**Stay with moral machines as the main dataset.** It's where the paper's most interesting and least supported claim sits, and it is the only one of the 7 datasets with an ethics angle. Its known generator also makes exact interventions possible, which the tabular datasets can't offer because their pairs are found by search rather than built.

Add one tabular dataset as a control (Aim 4). Without a control, a reviewer can't tell whether a finding is about ethics or about this one generator. If you need a second ethics domain, BBQ is already in the repo, and Matton et al. and Turpin et al. used it for findings about social-bias faithfulness.

---

## 6. Immediate next steps, in order

| When | Task | Output |
|---|---|---|
| Days 1–2 | Read the NSG paper's §2.3, its results and Appendix C; Awad (2018); Takemoto (2024); Matton (2025); Hong & Roth (2026); the stated-versus-revealed paper (2506.00751) | One page of notes: what each measured and how |
| Days 2–4 | EDA plots 1–5 and transcript reading (§4) | EDA notebook and a list of stated principles |
| Days 4–6 | Reproduce the 6.0% NSG for moral machines from `data_parquet/`, then compute stratified NSG and the moral machines Hamming distance (Aim 1) | **The key figure** |
| Day 7 | Meet HM with the figure and the H1 framing. Ask whether they knew about the matching rule and what Appendix C intended. | Go/no-go on the H1 framing |
| Week 2 | AMCE profiles (Aim 2) and validation of the tagger | Profiles of stated and revealed preferences |
| Week 3+ | Pre-register H1–H3, build the deterministic generator, run the Aim 3 pilot | Pilot results |

---

## 7. Items that need your review

- ☐ **REVIEW: Lead with H1?** Directly testing the paper's own "less faithful on ethics" claim is sensitive because HM is an author. Raise it as "calibrating our own metric", not "the paper is wrong".
- ☐ **REVIEW: Drop the consciousness framing (F1).** Is "stated versus revealed moral preferences" acceptable as the ethics angle?
- ☐ **REVIEW: Control dataset.** Use heart disease, income or BBQ?
- ☐ **REVIEW: API budget** for Aim 3. The pilot is about 1–2k calls; a full run with 6 models and K=5 is about 30–50k calls.
- ☐ **REVIEW: GPT-5.2 duplicate runs.** Were they independent? Ask whoever produced `data_parquet/`.
- ☐ **REVIEW: Coordination.** Aim 3's predictor variations overlap the simulator workstream (Christina, Maneet, Karol).

---

## 8. Sources
- Mayne et al. (2026), *A Positive Case for Faithfulness* (NSG; moral machines NSG 6.0%; Hamming ≤ 2): https://arxiv.org/abs/2602.02639
- Takemoto (2024), *The Moral Machine Experiment on LLMs* (AMCE/conjoint): https://arxiv.org/pdf/2309.05958
- *Large-scale moral machine experiment on LLMs* (52 LLMs): https://arxiv.org/pdf/2411.06790
- *Decoding Multilingual Moral Preferences* (Moral Machine, AIES): https://arxiv.org/pdf/2407.15184
- *Scaling laws for moral machine judgement in LLMs*: https://doi.org/10.1098/rsos.260202
- *Robustness of LLMs in moral judgements*: https://royalsocietypublishing.org/rsos/article/12/4/241229/235626/Robustness-of-large-language-models-in-moral
- *Are LLMs Consistent in Stated and Revealed Preferences?* (2025): https://arxiv.org/pdf/2506.00751
- *From Stability to Inconsistency: Moral Preferences in LLMs* (2025): https://arxiv.org/pdf/2504.06324
- Matton et al. (2025), *Walk the Talk?*: https://arxiv.org/abs/2504.14150
- Hong & Roth (2026), *Pragmatic Perturbations*: https://arxiv.org/abs/2601.03775
- *LLMs Don't Know Their Own Decision Boundaries* (2025): https://pith.science/paper/2509.09396
