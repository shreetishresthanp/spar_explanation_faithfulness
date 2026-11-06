# Moral Machines: findings since the 10/08 debrief

*WS2 (data distribution and counterfactual construction). Each section: research question, setup, finding. Figures:
`debrief/10_27_figures/`; interactive results: `analysis/mm18_report.html` (Decision rule, Experiments); slides:
`debrief/10_27_deck.html`.*

---

## 0. Recap of the 10/08 debrief

**Findings:**
- Low MM NSG comes mostly from the open-weight models: −6% on average, against 24% for proprietary models (close to
  their tabular 33%). Gemma 3 is the worst family.
- Qwen 3 roughly improves with model size (−8% at 0.6B to +15% at 32B).
- Gemma 3 27B picks Case 1 on 96% of prompts that ask for the answer last, against 59% when the answer comes first.
  Its explanations never mention this. *To-do: does this hold in fresh runs?* (answered in §7)
- Half (48%) of counterfactuals change only head counts. The model almost never flips on those, so explanations add
  nothing (NSG −8% for proprietary models). The gain comes from pairs that swap sides or traffic lights.
- Failures fall into six modes: too close, near-tie, out of scope, hidden cause, stated principle violated,
  construction artifact.

**Feedback / to-dos:**
- Prioritise generating better counterfactuals over probing consistency (same prompt about 10 times).
- Focus on proprietary models at medium or high reasoning. We chose GPT-5.2 (highest MM NSG) and Gemini 3 Flash (the
  simulator our collaborators use).
- Look at flip rates and simulator bias behind the counts-only result.
- Look into the egregious and misleading cases, where the explanation misled all 5 predictors.
- Harry: have Claude Code study each model's examples, write a "ground truth" account of how it decides, and use it.

---

## Finding the ground truth

Follows Harry's debrief suggestion: get a "ground truth" for how each model decides, and use it. We built it in two
forms: a fitted rule (in numbers) and plain-English descriptions (in words).

### 1. Fitted rule (ground truth in numbers)

- **RQ:** Can we write down how each model decides?
- **Setup:**
  - For each reference model, a simple decision rule (logistic regression) from the scenario's factors to its choices.
  - Factors: law (traffic light), passengers vs. pedestrians, swerving vs. staying in lane, number of lives, the
    demographic groups (age, gender, status, fitness, species), and answer format (answer asked first or last).
  - Tested on pairs it wasn't fitted on, with each original–counterfactual pair kept together.
- **Finding:**
  - The rule predicts the top 3 models' choices (GPT-5.2, Gemini 3 Flash, Gemini 3 Pro) with 92–96% accuracy.
  - GPT-5.2 is driven by law, passengers vs. pedestrians, not swerving, and number of lives.
  - Gemini 3 Flash puts the law first, then humans over animals, group size, and status.
  - Gemma 3 27B's largest factor is answer format, which its explanations never mention.

### 2. Descriptions of model behaviour (ground truth in words)

- **RQ:** Can the rule be turned into an explanation a simulator can use?
- **Setup:**
  - For GPT-5.2, Gemini 3 Flash and Gemma 3 27B (control), a short "how this evaluator decides" text.
  - Each was written from training pairs only, so the 100 held-out pairs per model stay unseen.
  - Two versions:
    - **Written:** by Claude, from the training-only fitted rule and 40 training decisions, then checked claim by
      claim against training statistics.
    - **Automatic:** from a small decision tree fitted on the same training decisions, turned into
      "situation → usual choice" lines. No judgment is involved; this version checks that results don't depend on
      Claude's writing.
- **Finding:**
  - The written descriptions match the fitted rule's main drivers, organised by situation. For example, for GPT-5.2,
    passengers vs. pedestrians comes first in car scenarios, and law comes first between pedestrian groups.
  - The decision-tree descriptions are coarser. For example, GPT-5.2's tree never mentions passengers vs. pedestrians.
- **Used for:** the explanations simulators read in Experiment 1.

---

## Where explanations stop helping

### 3. Distance to the decision boundary (paper's pairs, observational)

- **RQ:** Which counterfactuals carry little explanation signal?
- **Setup:** Use the fitted rule to measure how close each counterfactual is to a 50/50 call for the model. Call it a
  near-tie when the rule's P(Case 1) is between 0.3 and 0.7. Top 3 models, 2,835 pairs.
- **Finding:**
  - **Too close:** counts-only pairs leave no headroom. Predictors are 91% right without the explanation, and NSG is
    −8%.
  - **Near-ties:** NSG 12% [2, 20] on near-tie counterfactuals, against 37% [32, 41] on clear ones.
  - ⚠️ The near-tie gap **didn't replicate** on purpose-built pairs (§6), so treat it as observational, not causal.

### 4. Hand labels: why GPT-5.2's failures fail

- **RQ:** Do predictors fail when the counterfactual changes a factor the explanation doesn't mention?
- **Setup:** 30 GPT-5.2 pairs (20 where most predictors failed, 10 where they succeeded), labelled blind to the
  outcome: which factors the explanation cites, and what it predicts if read literally.
- **Finding:**
  - **No: citing a factor ≠ covering the counterfactual.** Almost every counterfactual changes a cited factor
    (failures 95%, successes 90%).
  - What separates them:
    - being a near-tie: 60% of failures vs. 10% of successes
    - the explanation pointing the wrong way: 45% vs. 0%. This is partly built in, since the labeller reads the same
      text as the predictors.
  - **The typical failure is a conditional rule whose condition doesn't hold.** For example, "protect pedestrians
    obeying traffic laws" applied to pedestrians crossing on red: all 5 predictors were wrong with the explanation,
    against 3 of 5 right without it.
  - 8 of 20 failures are explanations that can't decide the counterfactual, e.g. "minimise deaths" on a pair with
    equal deaths.
  - Only 3 of 20 are explanations that contradict what the fitted rule says the model consistently does. These are
    the strongest unfaithfulness candidates, and one of the three is itself a near-tie.

---

## Experiments

### 5. Experiment 1: does a description of the model's behaviour beat its own explanation?

- **RQ:** If a simulator reads a description of how the model actually decides, does it predict better than with the
  model's own explanation? If so, the explanation is missing information.
- **Setup:**
  - GPT-5.2, Gemini 3 Flash, and Gemma 3 27B as the control. 100 held-out pairs each, from the paper's pairs.
  - 2 simulators: gpt-oss-20b and Gemini 3 Flash.
  - 5 conditions: none, own explanation, written description, automatic description, written + own.
  - 3,000 calls.
- **Finding:**
  - **Gemma: the description exposes what its explanation hides.** Accuracy is 64% with its own explanation (worse
    than 72% with none) and 92% with the description: NSG −29% → +71%. The automatic description also beats its own
    explanation, so this doesn't depend on Claude's writing.
  - **GPT-5.2 and Gemini 3 Flash: no evidence a description adds anything.** Their own explanations do as well or
    better (89–91% vs. 83–84%).
  - **Simulators don't fully apply a written rule:** 83–92% with it, against 88–96% for the rule itself. So "the
    explanation recovers half of the rule's gain" measured reader vs. fitted model, not missing information.

| Simulator sees | GPT-5.2 | Gemini 3 Flash | Gemma 3 27B |
|---|---|---|---|
| no explanation | 79% | 82% | 72% |
| own explanation | **89%** | **91%** | 64% |
| description (written) | 83% | 84% | **92%** |
| description (automatic) | 78% | 81% | 79% |

### 6. Experiment 2: purpose-built counterfactuals

- **RQ:** On counterfactuals chosen on purpose, is explanation gain lower near the model's decision boundary?
- **Setup:**
  - New pairs built from scenarios each model already answered, so no new reference-model calls.
  - Per model: 50 near-tie + 50 clear pairs, each changing exactly 1 decision factor. Half are flips; traffic-light
    changes are capped at half.
  - Own explanation vs. none; 2 simulators × 2 repeats; 3,000 calls.
  - Gemma arm: 50 extra pairs from held-out scenarios only, adding the description as a third condition.
- **Finding:**
  - **The near-tie gap didn't replicate.** NSG is 21% on near-tie vs. 22% on clear pairs (GPT-5.2 + Gemini 3 Flash;
    difference −1 point [−24, +22]). Near-ties are harder with or without the explanation (65% vs. 76% accuracy), but
    the explanation helps about as much.
  - It isn't a flip-rate effect (31.5% vs. 30.8% on the paper's pairs). Why the two sets of pairs differ is still
    open.
  - **Cited vs. uncited is confounded:** cited pairs flip far more often (75% vs. 29%), so it can't be read.
  - **Gemma's description still helps on new pairs:** 71% with its own explanation, 74% with none, 82% with the
    description (NSG −11% → +30% [+7, +56]).

### 7. Gemma answer-format test (fresh runs)

- **RQ:** Does the answer format *cause* Gemma's choice, and does it hold in fresh runs? (The debrief to-do.)
- **Setup:**
  - 100 held-out scenarios, each asked answer-first and answer-last, 3 samples each. Fresh Gemma 3 27B calls with the
    paper's sampling settings.
  - The two prompts differ only in where `[ANSWER]` is requested.
  - Control: Gemini 3 Flash. Calibration: 20 of the paper's prompts, unchanged.
  - 860 calls.
- **Finding:**
  - **Causal and reproducible.** On the same scenarios, Gemma picks Case 1 51% of the time when asked first and 94%
    when asked last.
  - 44% of scenarios flip with the format alone, all towards Case 1 when answering last. Gemini 3 Flash: 0%.
  - Beyond noise: the same prompt twice disagrees 10% of the time; across formats, 43%.
  - Fresh answers match the paper's stored answers on 93% of the unchanged prompts.
  - **Explanations still argue from content.** In scenario 341, "prioritizing the safety of those adhering to the
    law" is used to justify the answer that kills the lawful crossers (asked last); asked first, Gemma gave the
    opposite answer for the same reason.
  - **Consistency** (the debrief's question): 83–88% of scenarios give the same answer on all 3 samples.
  - **Parser note:** asked first, Gemma often omits the `[ANSWER]` tag. The repo parser drops those (mostly Case 2),
    inflating the answer-first rate. Strict parsing gives 65% vs. 97%: still a large effect.

---

## 8. What changed since the debrief

| At the debrief | Now |
|---|---|
| Low NSG may mean counterfactuals are "too far away" | Not raw distance: counts-only pairs are **too close**, and the near-tie effect **didn't replicate** |
| Build counterfactuals from the factors explanations cite | Explanations already cite almost everything. **Citing ≠ covering**: conditional rules fail outside their condition |
| Frontier explanations recover about half of what's predictable | That compared a fitted model with a reader. A behaviour description **doesn't beat** their own explanations |
| Gemma looks unfaithful (position bias) | **Causal and reproducible**: the same scenario flips with the answer format alone. Simulators confirm it 3 ways |

## 9. Limitations

- **Small samples:** 30 hand labels (one labeller); 100 pairs per model in the experiments. Intervals are wide.
- **Simulator noise:** the same prompt twice agrees 84–98% of the time, and one frontier result moved 25 NSG points
  between two runs.
- **Self-simulation:** Gemini 3 Flash is both a reference model and a simulator.
- **2 simulators, not the paper's 5**, and stratified samples: compare within each experiment, not with the paper's
  NSG.
- **Main effects only:** the fitted rule has no interactions, so where it's wrong the cause could be noise or a
  missing interaction.
- **Cost:** $3.63 of the $50 key.

## 10. Next steps (to Nov 2)

- Agree placement in the shared report with Caleb (coverage) and Christina (non-determinism).
- Offer Gemma as a real-world "known rule hidden from the explanation" case for the toy-model story, and "citing ≠
  covering" as a real-data refinement of coverage.
- Dropped: resampling near-ties, since its motivation (the near-tie gap) didn't replicate.
- Open question: why does the near-tie gap appear on the generator's pairs but not on single-factor pairs?
