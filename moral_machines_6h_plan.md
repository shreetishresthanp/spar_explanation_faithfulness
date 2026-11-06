# Moral Machines: 6-hour plan to a pitchable WS2 result

**Pitch in one line:** On real explanations, whether a counterfactual is *in the explanation's scope*
(changes a factor it cites) predicts simulation failure better than raw distance — Caleb's
coverage result, measured on Moral Machines with scope defined by cited factors instead of Gower distance.

**Inputs already built:** `analysis/mm18_report.html` → Tag explanations → Coverage sample
(60 rows: GPT-5.2 medium + Gemini 3 Flash medium; 4 failures + 2 successes per pair type per model).

**Change from the original plan:** hand-label 30 rows (GPT-5.2 only) instead of all 60. All 60 would take about 2.5 hours,
leaving too little time to turn the labels into findings. 30 hand labels are enough to settle the codebook and check an
automatic tagger; Claude tags the remaining rows and runs the label-free analyses while you label.

---

## Hour 0:00–0:15 — Align with the team
- [ ] Message Caleb and Christina: "I'll contribute a real-data coverage result for the distance section:
      scope defined by cited factors, on Moral Machines. Near-tie pairs could go in the non-determinism section."
- [ ] Ask Caleb: what figure format/axes does his coverage plot use? (So yours can match.)

## Hour 0:15–1:30 — Hand-label 30 rows (YOU)
- [ ] Open the Coverage sample, set Model = GPT-5.2, label all 30 blind (factors → scope → literal prediction → reveal → failure mode).
- [ ] Write a note whenever the codebook is ambiguous (e.g. "more lives" vs. head-count changes, side swap = passengers or intervention?).
- [ ] Download `mm18_coverage_labels.csv` → `experiments/moral_machines/`.

**In parallel (CLAUDE, no labels needed):**
- [ ] Fit a per-model logistic decision rule (theme, per-side counts, in-car, intervention, law/light, answer position);
      cross-validate by pair. Sanity check: position ≈ 0 for GPT-5.2/Flash, large for Gemma 27B.
- [ ] Rule ceiling: fitted rule's accuracy on counterfactual answers vs. predictors with explanation.
- [ ] Boundary figure: NSG and flip rate binned by the rule's predicted P(flip), next to pair type.

## Hour 1:30–2:00 — First numbers from your labels (CLAUDE → YOU review)
- [ ] Table: failure vs. success rate by scope (in scope / irrelevant-factor / partial / out of scope), within pair type.
- [ ] Literal-prediction accuracy: how often does the explanation, read literally, give the right answer on failures?
- [ ] Failure-mode counts with one example each.
- [ ] **REVIEW:** do the numbers match your impression from labelling? Any codebook fixes before scaling?

## Hour 2:00–3:30 — Scale the tagging (CLAUDE → YOU review)
- [ ] Claude tags cited factors + scope for all GPT-5.2 (840) and Gemini 3 Flash (998) rows with subagents, using your fixed codebook.
- [ ] Agreement with your 30 labels (κ per factor and for scope).
- [ ] **REVIEW:** read the ~10 disagreements. If κ is low for a factor, decide: fix the prompt, or drop the factor.

## Hour 3:30–4:30 — Key figure (CLAUDE → YOU review)
- [ ] NSG by scope × crosses-fitted-boundary (2×2), with bootstrap CIs, per model; raw pair type alongside.
- [ ] Optional check: do the same pairs fail for GPT-5.2 high? (Existing data, no API.)
- [ ] **REVIEW:** is the headline claim supported? If in-scope vs. out-of-scope doesn't separate, the fallback pitch is
      the boundary result plus the failure taxonomy.

## Hour 4:30–6:00 — Write the pitch (YOU, Claude drafts)
- [ ] ~1 page: question, setup, 2 figures (boundary + scope), failure-mode table with examples,
      link to the draft's Fig. 6 (counts-only pairs: ~92% accuracy without the explanation, so no headroom for NSG).
- [ ] Next step to propose: counterfactuals built from explanations (change a cited factor / an uncited one / at random)
      on GPT-5.2; needs `OPENROUTER_API_KEY` and ~500–1,000 calls; generator needs a "render this scenario" function first (~half day).
- [ ] **REVIEW:** send to Caleb/Christina for placement in the shared report.

---

## Needs your decision
- [ ] Keep GPT-5.2 at medium (current) or switch to high? (NSG 36.6% vs 37.2%; switching resets GPT-5.2 labels.)
- [ ] Your own section, or a subsection of Caleb's?
- [ ] API key and budget for the follow-up pilot?

## Risks
- Scope labels may be noisy for vague explanations ("save more lives"); report κ honestly.
- 30 hand labels are a codebook check, not the result; the result rests on the scaled tagging.

## Caveats
- The times assume Claude's work runs while you label, so start the rule fit in the main conversation before you begin tagging.
- If the in-scope vs. out-of-scope split doesn't come out, the boundary figure plus the failure taxonomy is still a solid fallback pitch.

---

## Update after labelling GPT-5.2 (2026-10-27)

**The headline hypothesis failed for GPT-5.2.** Whether the counterfactual changes a cited factor doesn't separate
failures from successes:
- Counting a change to head counts as a change to "numbers": 19/20 failures vs 9/10 successes are in scope. This is
  near-constant by construction, since every pair changes head counts.
- Using the net difference in head counts instead: 10/20 vs 6/10.

Frontier explanations mention nearly everything, so citing a factor isn't the same as covering the counterfactual.

**What does separate them:**
- **Near-tie** (the fitted rule's P(Case 1) on the counterfactual is between 0.3 and 0.7): 12/20 failures vs 1/10 successes.
- **The labeller's literal reading is wrong:** 9/20 vs 0/10. This is partly mechanical, since the labeller reads the
  same text as the predictors.
- **The fitted rule is right:** 10/20 failures vs 9/10 successes.

**The 20 failures, split by literal reading × fitted rule** (descriptive; the sample is stratified, so don't read these as rates):

| Literal reading | Fitted rule | n | Reading |
|---|---|---|---|
| wrong | right | 3 | The explanation misstates what the model consistently does. Strongest unfaithfulness candidates (rows 35, 53, 56). |
| wrong | wrong | 6 | Neither predicts the choice. Noise or idiosyncrasy; resampling question (Christina's section). |
| can't tell | right | 5 | The explanation leaves out the decisive weighting. Incompleteness. |
| can't tell | wrong | 3 | Hard for everything. |
| right | — | 3 | Predictor error. All three are near-ties. |

**Re-plan:**
- **Dropped:** hour 2:00–3:30, the scaled factor tagging. It scales a variable that doesn't discriminate.
- **Pitch = the fallback:**
  - near-tie NSG 0.12 vs 0.37 on clear pairs (*later: didn't replicate on purpose-built pairs, 21% vs 22%*)
  - rule ceiling: explanations recover about half of what the fitted rule predicts
  - the failure table above, with one example per row
  - the labeller's notes as qualitative evidence that conditional rules fail when their condition doesn't hold in the
    counterfactual, e.g. "tie-breaker when casualties are equal" on a pair where they aren't equal. This is the point
    for Caleb: citing a factor isn't the same as covering the counterfactual.
- **Gemini 3 Flash labels:** optional, only to show the table isn't specific to GPT-5.2.
- **Caution:** "irrelevant" on numbers probably means "counts are tied here". Law is tagged irrelevant 5 times while
  it's GPT-5.2's largest fitted coefficient. Spot-check those rows before claiming a stated-vs-revealed gap.

**Files:**
- Code: `analysis/mm18_coverage.py`
- Scored labels: `experiments/moral_machines/mm18_coverage_labels_scored.csv`
