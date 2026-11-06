# Slack update: Moral Machines (WS2)

*Paste the message below into the group chat. It uses Slack formatting (`*bold*`, `•` bullets). Attach the two
figures listed after it. Share both links from each page's Share menu first, or teammates can't open them.*

---

*Moral Machines (WS2) update since the 10/08 debrief* 🧵

*Recap:* the paper's low MM NSG (6%) comes mostly from open-weight models (−6% vs. 24% for proprietary). Gemma 3 seemed to pick Case 1 whenever the answer was asked last. Half the pairs change only head counts and carry no explanation signal.
*Feedback I worked from:* generate better counterfactuals, and build a "ground truth" for how each model decides.

I focused on the proprietary models with the highest MM NSG (GPT-5.2, Gemini 3 Flash), plus Gemma 3 27B as the known-unfaithful control.

*Experiments*
• *Gemma format test:* the same scenarios asked answer-first and answer-last, in fresh runs.
• *Exp 1, "ground truth" explanations:* I fitted a decision rule for each model, turned it into a plain-English "how this model decides" description, and gave simulators that description instead of the model's own explanation. 100 held-out pairs per model.
• *Exp 2, new counterfactuals:* I built 350 purpose-built pairs by re-pairing scenarios each model had already answered (no new reference calls). Each pair changes exactly 1 factor, split into near-tie (the model is close to 50/50) vs. clear. On the paper's pairs, explanations seemed to help much less on near-ties (NSG 12% vs. 37%). *RQ:* near the model's decision boundary, do explanations stop being useful?

*Findings*
*1. Gemma's choices follow the prompt format, and its explanations hide it (causal, reproducible).*
• Same 100 scenarios, fresh runs: Gemma 3 27B picks Case 1 *51%* of the time when asked for its answer first and *94%* when asked last. 44% of scenarios flip on the format alone. Gemini 3 Flash control: 0% flip.
• Its explanations argue from content. Same scenario, same reason ("prioritise those adhering to the law"), opposite answers depending on the format.
• Simulators given a short description of Gemma's actual behaviour predict it at *92%*, vs. *64%* with its own explanation (worse than no explanation, 72%).
→ _Would this work as the report's real-world example of failure mode (1), unfaithfulness?_

*2. Frontier explanations already carry what a "ground truth" adds.*
• For GPT-5.2 and Gemini 3 Flash, their own explanations predict as well as or better than a description of how they actually decide: 89–91% vs. 83–84%. So no evidence of missing information.
→ _I haven't tested the weaker proprietary models (Haiku 4.5, GPT-5 Mini), whose explanations help predictors much less. I can run them (about $1) if that would be useful for the report._

*3. Citing ≠ covering.*
• In 30 hand-labelled GPT-5.2 pairs, failures aren't about factors the explanation never mentions: 95% of failures change a cited factor.
• Failures are conditional rules that don't hold in the counterfactual (e.g. "protect pedestrians obeying traffic laws" applied to jaywalkers), or near-ties.

*4. New counterfactuals: no, explanations don't stop being useful near the boundary.*
• On the new pairs the near-tie gap disappears (21% vs. 22%). Near-ties are harder to predict overall, but the explanation helps just as much there.
• Gemma's behaviour description still beats its own explanation on the new pairs (82% vs. 71%).
→ _Open question I'd love input on: why does the gap show up on the generator's random pairs but not on controlled ones?_

*Relevance to the shared report*
• The toy sections give a reference a known rule; MM does the reverse, estimating a real reference's rule and handing it to the simulator. So it checks each toy failure mode on real models.
• Gemma is a real, causal case of unfaithfulness, and NSG catches it. Counts-only pairs mirror the Fig. 6 baseline effect.
• On real explanations, coverage depends on whether a conditional rule decides the case. This is the "realistic setting" step from Caleb's discussion. @Caleb, does this fit as a real-data note in your distance section?
• @Christina, Gemma gives the same answer on all 3 samples for 83–88% of prompts. Happy to share the data if it's useful for non-determinism.

*Next step:* I'll draft a short §3.x "Real-model case study: Moral Machines" (after distance, before non-determinism) plus one discussion paragraph by *Thu 10/30*, unless you'd rather I fold it into the existing sections. Let me know!

📄 Detailed write-up: https://claude.ai/artifact/VToiYWvbKrnJpVj2N9VhMj
📊 Slides: https://claude.ai/artifact/Eea6c1iwi9NH3PsA3WEgyu
(3 API experiments, about 8,500 calls, $3.63 of the $50 key)

---

## Figures to attach (in this order)

1. `debrief/10_27_figures/fig6_gemma_format.png`: Gemma's Case 1 rate by answer format, Gemini 3 Flash flat. The
   headline result (point 1).
2. `debrief/10_27_figures/fig4_exp1_accuracy.png`: simulator accuracy by what it reads. Gemma jumps with the
   description, the frontier models don't (points 1–2).

Optional: `debrief/10_27_figures/fig2_near_tie.png` (the Exp 2 result), if someone asks about the new counterfactuals.


notes for myself:
The idea behind it

- A "near-tie" is a counterfactual where the model is close to 50/50 between Case 1 and Case 2. The fitted rule estimates this: if it says the model picks Case 1 with probability around 0.3–0.7, the case is a near-tie. If it's near 0 or 1, the case is "clear".
- What we saw first, on the paper's own pairs: explanations seemed to help much less on near-ties. NSG was 12% on near-ties against 37% on clear cases. A natural story: near the model's decision boundary, explanations stop being useful.
- Why that wasn't enough: the paper's pairs are random. A near-tie pair and a clear pair can differ in many other ways at once (which factor changed, how many things changed, whether the model flipped). Any of those could have caused the gap.

What experiment 2 did

- We built new pairs to make a fair comparison. Each model had already answered about 2,000 scenarios in the paper's data, so any two of them can be paired into a new "original → counterfactual" pair whose answers are already known. No new calls to the reference model were needed.
- The pairs were chosen so near-ties and clear cases differ only in being near-ties:
  - each pair changes exactly one decision factor, such as the traffic light
  - both groups have the same mix: half the pairs flip the answer, half don't
  - 50 near-tie and 50 clear pairs per model
- Then we measured explanation gain (NSG) in both groups, the same way as before.

How to read Fig. 2

- Left panel (paper's random pairs): clear 37%, near-tie 12%. The gap we first saw.
- Right panel (our purpose-built pairs): clear 22%, near-tie 21%. No gap.
- The thin lines on the bars are 95% intervals. On the right they're wide and overlap almost completely.
- Takeaway: the left-panel gap isn't a general rule that explanations fail near the decision boundary. Something about the paper's random pairs produced it, and we don't know what yet. Near-ties are still harder to predict overall (65% vs. 76% accuracy), but the explanation helps about as much there.
