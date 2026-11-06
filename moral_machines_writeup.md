# Moral Machines: why NSG is low, and what "too far away" means on real explanations

*Draft write-up, 2026-10-27 (WS2: data distribution and counterfactual construction). For discussing placement in the
shared midterm report (deadline 2026-11-02). Figures and tables are in `analysis/mm18_report.html`; each section names
its tab. Items marked **[CHECK]** need your decision.*

---

## Summary

- **The paper's Moral Machines (MM) NSG of 6.0% is an average over very different models.**
  - The 8 proprietary reference models average 23.9%, close to their tabular average (32.9%).
  - The 10 open-weight models average −6.1%.
  - For GPT-5.2, Opus 4.5 and both Gemini 3 models, MM isn't an outlier at all.
- **Raw counterfactual distance is the wrong axis.** Distance to the model's own decision boundary is the better one.
  - The closest pairs change only head counts and make up half of all pairs. They have NSG −8%: the model rarely flips,
    so there's nothing for an explanation to predict.
  - On the paper's pairs, counterfactuals near the model's decision boundary have NSG 12%, against 37% for the rest
    (top 3 models). This didn't replicate on 200 purpose-built single-factor pairs (21% vs. 22%; difference −1 point [−24, +22]), so treat it as observational, not causal (§11).
- **The models are very predictable from the scenario.** A simple fitted rule over scenario factors predicts the top
  3 models' decisions with 92–96% held-out accuracy. But simulators given a written version of that rule don't reach
  the rule's accuracy, so the rule isn't a ceiling a reader of an explanation can hit.
- **A description of how the model actually decides helps only where the explanation hides something** (experiment 1,
  §10).
  - Gemma 3 27B: simulator accuracy rises from 64% with its own explanation to 92% with a description of its
    behaviour, because its explanations hide the answer-position effect.
  - GPT-5.2 and Gemini 3 Flash: their own explanations do as well as or better than the description (88–91% vs.
    83–84%). There's no evidence the description adds information.
- **Citing a factor isn't the same as covering the counterfactual.**
  - In 30 hand-labelled GPT-5.2 pairs, nearly every counterfactual changes a factor the explanation cites.
  - Failures typically come from conditional rules whose condition doesn't hold in the counterfactual, or from
    near-ties.
- **Gemma 3 is a natural case of a hidden cause, now tested causally.** Asked the *same* scenario both ways in fresh
  runs, Gemma 3 27B picks Case 1 51% of the time when the answer is asked first and 94% when it's asked last. 44% of
  scenarios flip with the format alone (Gemini 3 Flash: 0%). Its explanations argue from content and never mention the
  format (§12).

---

## 1. Question and setting

NSG scores an explanation by how much it helps a predictor guess the reference model's answer on a counterfactual. In
the paper, MM is the lowest-NSG dataset (6.0%), which reads as "models are least faithful on ethical decisions".

MM is also a semi-toy setting. Every scenario is built from a few named factors:
- lives at stake
- swerve vs. stay in lane
- passengers vs. pedestrians
- traffic law
- a demographic theme (species, age, gender, social status, fitness, or group size)

So we can ask what a low NSG actually reflects (the models, the explanations, or the counterfactuals) with real model
behaviour and controlled inputs.

**Scope:**
- **Reference models:** the paper's 18, one config each.
  - 8 proprietary at medium reasoning: GPT-5 Nano, GPT-5 Mini, GPT-5.2, Claude Haiku/Sonnet/Opus 4.5, Gemini 3 Flash
    and Gemini 3 Pro. They were run on pair set A.
  - 10 open-weight: Qwen 3 ×6 and Gemma 3 ×4, run on pair set B.
- **Predictors:** the paper's 5 (gpt-oss-20b, Qwen3-32B, Gemma 3 27B, Gemini 3 Flash, GPT-5 Mini).
- **Data:** all results reuse the paper's released predictions in `data_parquet/`. No new model calls.

## 2. Dataset audit: how MM counterfactuals are built *(report: What drives low MM NSG)*

1. The generator samples 15,000 random scenarios (a port of Awad et al. 2018).
2. It groups scenarios that share the same character types and the same car / swerve / law flags.
3. It pairs two scenarios from a group whose head counts differ.

Things the pairing doesn't control:
- the traffic-light pattern
- which group is on which side
- the theme: **10% of pairs change theme**, although the paper's Appendix A.8 says pairs share it

There's no Hamming-distance bound. A "counterfactual" ranges from one extra character to swapped sides with flipped
lights.

| What changed | Share of pairs | Proprietary: flip rate / NSG | Open-weight: flip rate / NSG |
|---|---|---|---|
| counts only (closest) | 48% | 8% / **−8%** | 20% / **−17%** |
| side swapped | 18% | 21% / **46%** | 23% / 6% |
| light swapped | 17% | 79% / 25% | 48% / 5% |
| light + side swapped | 6% | 63% / 32% | 39% / 0% |
| theme changed | 10% | 48% / 23% | 44% / 8% |

The closest pairs carry no explanation signal. The proprietary models' gain comes from changes that touch factors
explanations discuss: sides (passengers vs. pedestrians, intervention) and the law.

## 3. The paper's number comes mostly from the open-weight models *(report: Overview)*

| Reference models | MM NSG | Same models, tabular mean |
|---|---|---|
| Paper, 18-model average (reported) | 6.0% | — |
| Ours, 18-model average | 7.2% | 32.3% |
| 8 proprietary | **23.9%** | 32.9% |
| 10 open-weight | **−6.1%** | 31.7% |

- **Within 1 point of the model's own tabular mean:** GPT-5.2, Opus 4.5 and Gemini 3 Flash/Pro.
- **Larger gaps:** Haiku 4.5 (−24 points), GPT-5 Nano and Mini (−20), and every open-weight model (Qwen 3 −23 to −29,
  Gemma 3 −42 to −69).
- So "models are less faithful on ethics" holds for small and open models and for some proprietary ones, but not for
  the strongest.
- The remaining gap between our 7.2% and the paper's 6.0% is most likely the reasoning level used for GPT-5.2 and
  Sonnet 4.5, which the paper doesn't state.

## 4. A hidden cause: Gemma answers by prompt format *(report: What drives low MM NSG)*

Each prompt randomly asks for the answer first or last, independently of its content.
- Gemma 3 4B, 12B and 27B pick Case 1 on 91–96% of answer-last prompts, against 59–79% of answer-first prompts.
- Its negative NSG sits almost entirely in the answer-last prompts. For Gemma 3 27B, NSG is −0.36 on answer-last and
  +0.02 on answer-first.
- Its explanations argue from content and almost never mention position (4.5% use any order wording).
- This isn't a parser artifact: every response has exactly one `[ANSWER]` tag, and the model writes "CASE 1" itself.

This is a natural, non-injected case of the bias Turpin et al. (2023) describe: the explanation omits the real driver
of the answer. MM has no correct answer, so a negative NSG here means Gemma's explanations actively mislead predictors
about its behaviour. That makes Gemma a useful known-unfaithful control: any method for detecting unfaithfulness should
flag it.

## 5. Distance to the decision boundary *(report: Decision rule)*

**The fitted rule.** For each reference model, we fit an L2 logistic regression from the scenario factors (plus answer
position) to its choices, cross-validated with pairs kept whole. This is the quantitative version of Harry's
"ground-truth explanation" suggestion, and an empirical "known rule" for each real model in the sense of the toy setups.

- **Accuracy:** held-out accuracy is 92–96% for the top 3 models (GPT-5.2, Gemini 3 Flash, Gemini 3 Pro).
- **What drives them:**
  - The law is the largest effect for all three. GPT-5.2 also puts heavy weight on not swerving and on protecting
    pedestrians over passengers.
  - Gender is about zero, matching their "gender shouldn't matter" explanations.
  - Answer position is small but not zero for the proprietary models; it's Gemma's largest effect.

**The boundary.** The rule gives each pair a predicted flip probability, and a margin: how far the rule's P(Case 1) on
the counterfactual is from 0.5.
- **Near-ties:** counterfactuals with a margin under 0.2 have NSG of **12%** [2, 20], against **37%** [32, 41] for the
  rest (top 3 pooled, 2,835 pairs). The gap holds at margins of 0.1 and 0.3.
  **But it didn't replicate on 200 purpose-built single-factor pairs (21% vs. 22%; difference −1 point [−24, +22]), so treat it as observational, not causal: see §11.**
- **Counts-only pairs** have a mean predicted flip probability of 0.12. Predictors are already 91% accurate on them
  without the explanation, which leaves no headroom. This is the real-data mirror of the toy-model finding that NSG
  depends on the no-explanation baseline.
- **Not a smooth climb:** NSG is 20–30% in the four lower quintiles of predicted flip probability and 42% in the top one.

Two readings fit the near-tie result equally well:
1. the explanations leave out how the model trades factors off near its boundary
2. the model is close to a coin flip there

Experiment 2 (§11) didn't find the gap on purpose-built pairs, which weakens both readings.

## 6. How predictable the models are *(report: Decision rule)*

These are accuracies on the counterfactual answer, all held out.

| Model | Predictors, no explanation | Predictors, with explanation | Fitted rule | Explanation NSG ÷ rule NSG |
|---|---|---|---|---|
| GPT-5.2 | 84.9% | 90.4% | 94.3% | 59% |
| Gemini 3 Flash | 84.1% | 88.8% | 93.8% | 49% |
| Gemini 3 Pro | 81.9% | 87.2% | 92.3% | 51% |
| Claude Haiku 4.5 | 76.2% | 77.8% | 84.9% | 18% |
| GPT-5 Mini | 82.0% | 85.0% | 93.7% | 26% |
| Gemma 3 27B (control) | 79.3% | 76.8% | 93.0% | negative |

The weaker proprietary models are about as predictable from the scenario as the strong ones, yet their explanations
help predictors much less.

**Don't read the last column as "share of the information the explanation recovers".** The rule is a statistical
model fitted on about 1,300 of the model's own decisions; a predictor reads one explanation. Experiment 1 (§10) tested
this directly: simulators given a written version of the rule reach 83–92%, below the rule's own 88–96%. For GPT-5.2
and Gemini 3 Flash, the models' own explanations do as well as the written rule. So the gap between the explanation
and the fitted rule is mostly the difference between a fitted model and a reader, not missing information. The
exception is Gemma (§10).

## 7. Hand labels: why GPT-5.2's failures fail *(report: Decision rule, Tag explanations)*

**Sample:** 30 GPT-5.2 pairs, blind to the outcome.
- 20 failures (most predictors wrong even with the explanation) and 10 successes.
- 4 failures and 2 successes per pair type, egregious failures first.
- It's stratified by design, so these counts describe the sample; they aren't rates.

**Labels per pair:**
- which factors the explanation (with its factor list) cites as a reason or as irrelevant
- which answer the explanation, read literally, implies for the counterfactual

| | Failures | Successes |
|---|---|---|
| Changed factor is cited (any head-count change counts) | 19 / 20 | 9 / 10 |
| Changed factor is cited (net head-count difference) | 10 / 20 | 6 / 10 |
| Near-tie under the fitted rule | **12 / 20** | **1 / 10** |
| Literal reading wrong | 9 / 20 | 0 / 10 |
| Literal reading "can't tell" | 8 / 20 | 3 / 10 |
| Fitted rule right | 10 / 20 | 9 / 10 |

- **Citing the changed factor doesn't separate failures from successes.** Every generated pair changes head counts,
  and frontier explanations cite nearly every factor.
- **Being a near-tie does separate them.**
- **A careful human reader given the same explanation was misled on 9 of the 20 failures and on none of the
  successes.** This is partly built in, since the reader sees the same text as the predictors. It still places these
  failures in the explanation text rather than in the simulators.
- **A recurring pattern: conditional rules whose condition fails.** For example, "when fatalities are equal, protect
  law-abiding pedestrians", on a pair where fatalities differ or every pedestrian is crossing illegally. The
  explanation names the changed factor but doesn't say what happens outside its condition.

The 20 failures, split by the literal reading and the fitted rule:

| Literal reading | Fitted rule | n | Reading |
|---|---|---|---|
| wrong | right | 3 | The explanation misstates what the model consistently does: unfaithfulness candidates |
| wrong | wrong | 6 | Neither predicts the choice: non-determinism, or an interaction the rule omits |
| can't tell | right | 5 | The explanation leaves out the deciding trade-off: incomplete |
| can't tell | wrong | 3 | Hard for everything |
| right | — | 3 | The predictors erred; all three are near-ties |

- **[CHECK]** Are #53 and #56 clean enough to cite as unfaithfulness candidates? #35 is itself a near-tie.

## 8. Failure modes for counterfactual simulatability (updated taxonomy)

| Failure mode | What happens | Evidence |
|---|---|---|
| Too close | No flip, so nothing to predict | counts-only NSG −8% (proprietary); mean predicted flip probability 0.12 |
| Near-tie | The model's rule is close to indifferent | 12 of 20 hand-labelled failures; NSG 12% vs 37% on the paper's pairs, not replicated on purpose-built pairs |
| Conditional rule doesn't cover it | The explanation cites the factor but its stated condition doesn't hold | Recurring in the GPT-5.2 notes (§7) |
| Hidden cause | The real driver is never mentioned | Gemma: ~95% Case 1 when the answer comes last |
| Stated rule contradicted | The explanation points one way and the model's consistent behaviour another | 3 of 20 GPT-5.2 failures (2 clean) |
| Construction artifact | The pair changes more than intended | 10% of pairs change theme |

**Link to the toy models.** "Too far away" on real data is better read as "outside the region where the explanation's
stated rule decides the case" than as a distance. An explanation can name every factor a counterfactual changes and
still not say how they trade off. That argues for building counterfactuals from the explanation's own conditions
(WS2).

## 9. Limitations

- **Hand labels:** 30 rows, one labeller, one model, a stratified sample. The labeller's literal reading isn't
  independent of the predictors' (same text).
- **The fitted rule has main effects only** (no interactions, no per-character terms). Where it's wrong, the cause
  could be noise or a missing interaction.
- **Near-tie vs. missing information:** these aren't separated yet; that needs resampling (overlaps with the
  non-determinism workstream).
- **One reasoning level:** the proprietary models are at medium. GPT-5.2 at high has a similar MM NSG (37.2% vs
  36.6%), but we haven't checked that it fails on the same pairs.
- **Pair sets:** proprietary and open-weight models were run on different pair sets (142 shared), so compare them
  model by model.

## 10. Experiment 1: ground-truth explanations *(report: Experiments)*

**Question:** does a description of how a model actually decides help simulators more than the model's own
explanation? If so, the explanation is missing information about the model's behaviour.

**Setup:**
- **Models:** GPT-5.2, Gemini 3 Flash, and Gemma 3 27B as the control.
- **Pairs:** 100 held-out pairs per model (20 per pair type), taken from the paper's own counterfactual pairs. No new
  counterfactuals.
- **Two ground-truth texts per model, both from training pairs only:**
  - written by Claude from the fitted rule's coefficients and 40 training decisions, then checked claim by claim
    against training statistics (`experiments/moral_machines/mm18_gt_claims.md`)
  - generated automatically from a depth-3 decision tree, with no judgment involved
- **Simulators:** gpt-oss-20b and Gemini 3 Flash.
- **Conditions:** the model's own explanation, written ground truth, tree ground truth, written ground truth + own,
  and no explanation.
- **Prompts:** every prompt states whether the evaluator was asked for its answer before or after its reasoning.
- **Size and cost:** 3,000 calls, about $1.12.

**Results** (accuracy on the counterfactual answer, both simulators pooled):

| Simulator sees | GPT-5.2 | Gemini 3 Flash | Gemma 3 27B |
|---|---|---|---|
| no explanation | 79.0% | 81.5% | 72.0% |
| model's own explanation | **88.5%** | **91.0%** | 64.0% |
| ground truth (written) | 83.0% | 83.5% | **92.0%** |
| ground truth (decision tree) | 77.5% | 80.5% | 78.5% |
| written ground truth + own | 84.0% | 89.5% | 88.5% |
| *fitted rule, no simulator* | *88%* | *96%* | *94%* |

- **Gemma:** its explanation hides how it decides.
  - Its own explanation is worse than no explanation (64% vs. 72%).
  - A description of its behaviour raises accuracy to 92%: NSG goes from −29% to +71%, a gain of +100 points
    [+73, +136].
  - The automatic tree text also beats its own explanation (+52 points [+24, +87]), so the result doesn't depend
    on the judgment-written text.
- **GPT-5.2 and Gemini 3 Flash:** there's no evidence a behaviour description adds anything.
  - Their own explanations do as well as or better than the written ground truth.
  - Adding the description to their own explanation doesn't help.
  - The tree text is clearly worse than their own explanations.
  - One reading: their explanations carry pair-specific reasoning that a global rule doesn't.
- **Simulators don't execute a written rule as well as the fitted rule.** Given the written ground truth they reach
  83–92%, against the rule's own 88–96%.

**Caveats:**
- In the Gemini 3 Flash column, one of the two simulators is Gemini 3 Flash itself.
- 100 pairs per model gives wide intervals. The Gemini 3 Flash ground truth − own gap moved from −15 to −40 NSG
  points between two runs of the same prompts, so simulator sampling noise is large as well.
- The held-out sets are stratified, and gpt-oss-20b's new answers agree with the paper's stored answers only 74–92%
  of the time. Compare conditions within this experiment only, not with the paper's NSG.
- gpt-oss-20b ran out of its 2,000-token budget before answering on 36% of tree prompts. All of its calls were re-run
  at 8,000 tokens so every condition uses the same cap.

## 11. Experiment 2: purpose-built counterfactuals *(report: Experiments)*

**Question:** on counterfactuals chosen on purpose rather than by the generator, is explanation gain lower near the
model's decision boundary?

**Setup:**
- New pairs built from scenarios each model had already answered, so no new reference-model calls.
- Per model: 50 near-tie + 50 clear pairs, each changing exactly one factor; traffic-light changes capped at half;
  flips balanced 50/50; no scenario used twice.
- Simulators: gpt-oss-20b (8,000-token cap) and Gemini 3 Flash, 2 repeats each. Conditions: own explanation vs. none.
- Gemma ground-truth arm: 50 extra pairs built only from experiment 1's held-out scenarios, adding the behaviour
  description as a third condition.
- 3,000 calls, about $1.

**Results:**
- **Near-tie vs. clear: no difference.** GPT-5.2 + Gemini 3 Flash: NSG 21% near-tie vs. 22% clear (difference −1
  point [−24, +22]). Near-ties are harder with or without the explanation (accuracy 65% vs. 76% with it), but the
  explanation helps about as much. It's not a flip-rate effect: on the paper's pairs the two groups flipped equally
  often (31.5% vs. 30.8%). Untested candidate reasons: single-factor changes (half of them the traffic light), the 50%
  flip balance changing which clear pairs are included, 2 simulators vs. the paper's 5, the generator's pair-type mix.
- **Cited vs. uncited: confounded.** Cited pairs flip 75% of the time, uncited 29%, so the comparison can't be read.
- **Gemma: the behaviour description still helps on new pairs.** Accuracy 70.5% with its own explanation, 73.5% with
  none, 81.5% with the description (NSG −11% vs. +30% [+7, +56]). Third confirmation of the hidden cause. (Only 4 of
  these pairs are near-ties: too few to read.)
- Simulator consistency: the same prompt twice gets the same answer 93–98% of the time (Gemini 3 Flash), 84–87%
  (gpt-oss-20b).

## 12. Gemma answer-format test *(report: Experiments)*

**Question:** does the answer format *cause* Gemma's choice, and does the effect hold in fresh runs? In the paper's
data each scenario was asked in only one format, so 96% vs. 59% compared different scenarios.

**Setup:**
- **Main test:** 100 of Gemma's held-out scenarios, each asked answer-first and answer-last, 3 samples each. Fresh
  Gemma 3 27B calls with the paper's sampling (temperature 1.0, top-p 0.95, top-k 64).
- **Prompts:** the repo's own reference prompts. The two formats differ only in where `[ANSWER]` is requested.
- **Control:** Gemini 3 Flash on the same prompts.
- **Calibration:** 20 of the paper's original prompts, unchanged.
- **Size and cost:** 860 calls, $1.10.

**Results:**

| | Answer asked first | Answer asked last |
|---|---|---|
| Gemma 3 27B: Case 1 rate | 51% [42, 60] | **94%** [91, 97] |
| Gemini 3 Flash (control) | 46% | 46% |

- **Causal within a scenario:** 44 of 100 scenarios change Gemma's majority answer when only the format changes, and
  all 44 go to Case 1 when the answer is asked last. The control flips on 0 of 100.
- **Beyond noise:** two samples of the same prompt disagree 10% of the time; the same scenario in different formats,
  43%.
- **Reproduces the paper:** fresh answers match the paper's stored answers on 93% of the unchanged prompts. Fresh
  samples agree with each other 97% of the time.
- **Consistency** (the debrief's question): 83–88% of scenarios give the same answer on all 3 samples.
- **Explanations still argue from content.** Scenario 341, answer asked last: Case 1, which kills the lawful
  crossers, justified by "prioritizing the safety of those adhering to the law". Asked first, Gemma chose Case 2 with
  the same principle.

**Parser note:** asked for its answer first, Gemma often writes a bare "CASE 2" first line without the `[ANSWER]` tag.
The repo parser drops these (120 of 360 answer-first calls, 92 of them Case 2), which inflates the answer-first Case 1
rate. The numbers above take the leading answer. With the strict parser it's 65% vs. 97%, still a large effect.

## 13. Next steps

- Resampling near-ties (with Christina) is deprioritised: its motivation was the near-tie gap, which didn't replicate.
- Priority before Nov 2: the write-up and placement in the shared report.

---

**[CHECK] before sharing:**
- [ ] Include Gemma (§4) in the shared report, or keep it as an appendix control?
- [ ] Cite #53 / #56 as unfaithfulness candidates (§7)?
- [ ] Which figures to export as PNGs (proposed: near-tie vs. clear, and the rule-ceiling chart)?
- [ ] Ask HM: are the theme-changing pairs (10%) intended? Which reasoning level does the paper use for GPT-5.2 and
      Sonnet 4.5?
