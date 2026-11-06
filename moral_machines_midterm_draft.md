# Moral Machines: draft text for the shared midterm report

*Draft, 2026-10-27, for the "Counterfactual distance" section of `debrief/collaborators.txt`, or as a short real-data
subsection after it. Placement is to agree with Caleb. Figures are in the **Decision rule** tab of
`analysis/mm18_report.html`. Numbers come from `analysis/mm18_rule.py` and `analysis/mm18_coverage.py`.
Items marked **[CHECK]** need your confirmation before this goes to the team.*

---

## Methods (add to §2, after the distance setup)

We repeat the coverage question on real explanations using Moral Machines (MM), where every scenario is a known
combination of a few factors: lives at stake, swerve vs. stay, passengers vs. pedestrians, traffic law, and a
demographic theme. For each reference model we fit an L2 logistic regression from these factors (plus prompt answer
order) to its decisions, cross-validated with original–counterfactual pairs kept whole. This gives an empirical "known
rule" for each real model. From it, each pair's predicted flip probability and its distance to the model's decision
boundary follow. To read the explanations directly, one of us hand-labelled 30 GPT-5.2 pairs blind to the outcome
(20 where most predictors failed with the explanation and 10 where they succeeded, balanced across pair types). Each
label records which factors the explanation cites and which answer the explanation, read literally, implies for the
counterfactual.

## Results (add to §3, "Counterfactual distance")

**On real explanations, citing the changed factor is not the same as covering the counterfactual.**
- The fitted rules predict held-out decisions with 92–96% accuracy for the three strongest MM reference models
  (GPT-5.2, Gemini 3 Flash and Gemini 3 Pro). But simulators given a written version of that rule don't reach it.
  On 100 held-out pairs per model:
  - GPT-5.2 and Gemini 3 Flash: their own explanations do as well as or better than a description of their behaviour
    (88–91% vs. 83–84% accuracy).
  - Gemma 3 27B: the description raises accuracy from 64% to 92% (NSG −29% → +71%), because its explanations hide
    that it answers by prompt format. An automatically generated description reproduces the gain.
  So a gap between an explanation and a fitted rule doesn't by itself show missing information. A behaviour
  description beating the explanation does.
- Explanation gain is lowest near the model's decision boundary, not at the largest raw distances:
  - The closest pairs change only head counts and make up about half of all pairs. For the top 3 models their mean
    predicted flip probability is 0.12. Across all 8 proprietary models, the reference model flips on 7.5% of them,
    predictors are already 91% accurate without the explanation, and NSG is −8% [−14, −3]. This is the mirror image
    of Fig. 6: the baseline leaves no headroom.
  - On the paper's pairs, counterfactuals the rule places near a tie (P(Case 1) between 0.3 and 0.7) have NSG of
    12% [2, 20], against 37% [32, 41] for the rest (Fig. X). But this didn't replicate on 200 purpose-built single-factor pairs (21% vs. 22%; difference −1 point [−24, +22]), so treat it as observational, not causal.
  - Two readings fit the near-tie result equally well: the explanations leave out how the model trades factors off
    near its boundary, or the reference model is close to a coin flip there. Resampling separates them (see
    Non-determinism).
- The hand labels show that the obvious coverage test doesn't work on these explanations. Whether the counterfactual
  changes a factor the explanation cites does not separate failures from successes:
  - 19 of 20 failures and 9 of 10 successes are "in scope" when any change in head counts counts as changing the
    number of lives. It's near-constant because every generated pair changes head counts.
  - On net differences in head counts it's 10 of 20 vs. 6 of 10.
- What does separate them is being a near-tie under the fitted rule: 12 of 20 failures vs. 1 of 10 successes.
- A careful reader given the same explanation, blind to the outcome, was misled on 9 of the 20 failures and on none
  of the successes. This is partly built in, since the reader sees the same text as the predictors. It still places
  these failures in the explanation text rather than in the simulators, consistent with the capability results above: capable
  simulators are reliable instruments.
- A recurring pattern in the labeller's notes is a **conditional rule whose condition fails in the counterfactual**:
  "when fatalities are equal, protect law-abiding pedestrians", applied to a pair where fatalities differ or every
  pedestrian is crossing illegally. The explanation names the changed factor but doesn't say what happens outside its
  stated condition.
- In 3 of the 20 failures the explanation pointed the wrong way while the fitted rule was right, i.e. the explanation
  misstates what the model consistently does. One of the three is itself a near-tie. In 6 more, both the explanation
  and the rule were wrong. That is consistent with non-determinism, or with interactions the main-effects rule omits.

**Fig. X.** *Moral Machines, top 3 reference models pooled (2,835 pairs). NSG on near-tie vs. clear counterfactuals,
where a near-tie is a fitted-rule P(Case 1) on the counterfactual within 0.1, 0.2 or 0.3 of 0.5 (95% pair-bootstrap
intervals). Explanation gain is lowest where the model's own rule is closest to a tie.* (Report tab: "Near-ties carry
little explanation signal". A companion figure, NSG and flip rate by quintile of predicted flip probability, can go in
an appendix. NSG there is 20–30% in Q1–Q4 and 42% in Q5.)

## Discussion (one or two sentences for §4)

On a real dataset, "too far away" is better read as "outside the region where the explanation's stated rule decides
the case" than as a distance. An explanation can name every factor a counterfactual changes and still not say how
they trade off. This argues for building counterfactuals from the explanation's own conditions (WS2): change a factor
inside the stated rule's condition, and separately cross the condition's boundary.

## Non-determinism hook (for Christina's section, optional)

Of the 20 hand-labelled GPT-5.2 failures, 12 are near-ties under the fitted rule. In 6, neither the explanation nor
the rule predicted the choice. Resampling these pairs would separate a near-random reference model from explanations
that leave out how it trades factors off.

---

## [CHECK] before sending

- [ ] **Placement:** inside Caleb's distance section, or a separate real-data subsection? *(Ask Caleb.)*
- [ ] **Sample size:** the hand-label counts come from a stratified sample of 30. The text says "of 20 failures"; keep it
      descriptive and don't convert it to rates.
- [ ] **"Fig. 6":** refers to the draft's mirrored-rule figure. Check the number after the draft is renumbered.
- [ ] **Gemma control:** leave it out here, or add one clause: "a model whose largest effect is answer order, which its
      explanations never mention, has NSG −12% despite a rule NSG of 66%"?
- [ ] **Model scope:** the top 3 only, or mention that the weaker proprietary models (Haiku 4.5, GPT-5 Mini) are as
      predictable from the scenario but their explanations help predictors much less? (Not tested in experiment 1.)
- [ ] **The 3 "contradiction" cases (#35, #53, #56):** read them once more before calling them unfaithfulness candidates in
      a shared document. #35 is itself a near-tie (P(flip) 0.64), so only #53 and #56 are clean.
- [ ] **Static figure:** the midterm needs a PNG of Fig. X. Ask, and I'll export it from the report.
