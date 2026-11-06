# Moral Machines: status, what the evidence says, and what to do next

> **Superseded (2026-10-07).** The numbers below use the earlier 13- and 26-config scope, and the scripts it cites are now in `analysis/_archive/`. For the current 18-model scope and plan, see `moral_machines_status_report.md` and `analysis/mm18_report.html`. Keep this file for the detailed reasoning, toy-experiment list and literature.


*Working doc, last updated 2026-10-07. Edit freely: tick boxes, write in the `> My notes:` blocks, strike out directions you drop.*

This doc builds on two earlier plans:
- [`moral_machines_research_plan.md`](moral_machines_research_plan.md) (v2 plan, H1–H3)
- [`moral_machines_next_steps.md`](moral_machines_next_steps.md) (data inventory, Phases A–C)

Where they conflict, **this doc supersedes them**. Most importantly, H1 needs reframing; see §3.

**Provenance.** Numbers tagged **[re-run]** were regenerated on 2026-10-07 from `data_parquet/` by:
- `analysis/moral_machines_failure_analysis.py` (set A breakdowns)
- `analysis/mm_vs_tabular_nsg.py` (new: per-model MM vs. tabular, reproduction of 6.0%, position bias)

Numbers tagged **[inherited]** come from the plan docs and were **not** re-checked this session.

---

## Who is who: reference vs. predictor models

- **Reference models** are the models whose explanations are being tested. The paper has **18**:
  - Qwen3 0.6B, 1.7B, 4B, 8B, 14B and 32B
  - Gemma-3 1B, 4B, 12B and 27B
  - GPT-5-nano, GPT-5-mini and GPT-5.2
  - Claude Haiku 4.5, Sonnet 4.5 and Opus 4.5
  - Gemini 3 Flash and Pro
- **Predictor models** guess a reference model's answer on the counterfactual, with and without its explanation. There are **5**: gpt-oss-20b, Qwen3-32B, Gemma-3-27B, Gemini 3 Flash and GPT-5-mini. All 5 scored 23 of the 26 configs. **GPT-4o-mini, Gemini 2.0 Flash and GPT-5.2 (none) were scored by only Gemini 3 Flash and GPT-5-mini** (checked with `analysis/mm_coverage.py`; see the *Coverage* tab).
  - Qwen3-32B, Gemma-3-27B, Gemini 3 Flash and GPT-5-mini also appear as reference models, so self-preference is possible.
- **One NSG number** belongs to one reference model on one dataset, pooled over all 5 predictors' guesses. The paper's 6.0% is the plain average of the 18 reference models' Moral Machines NSGs.
- **`data_parquet/` holds 26 reference-model configurations, not 18.**
  - GPT-5.2 appears at reasoning low/medium/high (set A) and none (set B). Sonnet 4.5 appears at none/low/medium/high.
  - Two extra reference models aren't in the paper at all: GPT-4o-mini and Gemini 2.0 Flash. They're in set B's `extra_three_ref_models` file, together with GPT-5.2 at no reasoning.
  - Reproducing the 18-model average means choosing one config each for GPT-5.2 and Sonnet. That choice is why the result is a range (5.0–7.3%).

---

## 0. TL;DR

1. **The paper's 6.0% reproduces, but it's an average over very different models.**
   - The unweighted mean MM NSG over the paper's 18 models comes to 5.0–7.3%, depending on which GPT-5.2 and Sonnet configs are used.
   - For 8 frontier models, MM NSG is **23.9%**, against a tabular mean of 32.9%.
   - For the 10 Qwen3 and Gemma-3 models, MM NSG is **−6.1%**, against a tabular mean of 31.7%. **[re-run]**
2. **For the strongest models, MM isn't an outlier.**
   - MM minus the tabular mean is about 0 for GPT-5.2 (medium/high), Opus 4.5 and both Gemini 3 models (−0.01 to +0.005). That's within the tabular spread.
   - So "models are less faithful on ethics" holds for small/open models and some frontier models (Haiku −0.24, GPT-5-mini −0.20, GPT-5-nano −0.21 relative to tabular), but not for the strongest reasoning configs. **[re-run]**
3. **Gemma's negative MM NSG looks like genuine, Turpin-style unfaithfulness.** It is the strongest ethics-specific finding in the data.
   - On answer-last prompts Gemma-3 4B, 12B and 27B pick "Case 1" 91–96% of the time, against 59–79% on answer-first prompts. Gemma-1B is milder: 65% vs. 47%. Answer position is randomised independently of content.
   - Its negative NSG sits almost entirely in answer-last prompts. Gemma-27B: −0.36 answer-last, +0.02 answer-first.
   - Its explanations argue from content and never mention position. In 1 of 4 samples I read, the explanation argued for the *other* case (§2.4); T1 will measure how common that is.
   - Checked: this isn't a parser artifact. Every Gemma response has exactly one `[ANSWER]` tag, and the model itself writes "CASE 1" (27B: 492 of 514 answer-last rows).
   - Gemma is faithful on tabular data (tabular mean NSG 0.26–0.42). **[re-run]**
4. **The pair construction finding (old Finding 2) holds, but it cuts the other way from H1 as written.**
   - Counts-only pairs (48% of pairs) carry no explanation gain: NSG −6% for set A and −15% for set B.
   - All of the gain is in light swaps and side swaps.
   - So building *closer* counterfactuals wouldn't push MM NSG up toward the tabular range. If anything it would push it down. **[re-run]**

---

## 1. The premise: how MM counterfactuals are actually built

> *"Moral machine dataset seems to not use Hamming distance but just manually generated and tweaked configurations."*

**Half right.** There's no Hamming distance, but nothing is hand-tweaked either. This is from `moral_machines_generator.py` and `tabular_utils.generate_moral_machine_scenarios`.

1. **15,000 scenarios are randomly sampled.** It's a port of Takemoto (2024), which ports Awad et al. (2018). Each scenario draws:
   - one of the 6 dimensions
   - the binary flags `is_in_car`, `is_interventionism` and `is_law`
   - characters (1–5 pairs, drawn from the dimension's two groups)
   - which group goes in Case 1 (a 50% coin flip)
   - the traffic-light pattern (random)
2. **Scenarios are paired by matching on a key.** The key is the set of character types plus the 3 flags. A pair needs different combined counts (`Counter(c1) + Counter(c2)`). One pair is sampled per key group.
3. **What the key does *not* include:**
   - **The traffic-light pattern.** It's re-randomised, so whether the light swaps within a pair is a coin flip.
   - **Which side each group is on.** If the combined counts stay the same, a pure side swap would be deduplicated, so side swaps occur only together with a count change.
   - **The scenario dimension.** **10.2% of pairs change dimension.** **[re-run]** Example: a gender scenario with {woman, man} matches a utilitarianism scenario with the same types. This **contradicts the paper's Appendix A.8**, which says pairs "share the same scenario dimension".
4. `counterfactual_hamming_distance` is `None` for every MM row. The paper excludes MM from its distance-1/3/5 experiment (A.10) "due to the differences in the counterfactual generation process."

**What changes within a pair** (set A, 11,506 (model, pair) rows) **[re-run]**:

| Pair type | Share | Reference flip rate | Acc. with expl. | Acc. without | NSG [95% CI] |
|---|---|---|---|---|---|
| counts only | 47.9% | 6.6% | .916 | .920 | **−6.1%** [−12.6, −0.1] |
| side swapped | 18.3% | 19.0% | .865 | .743 | **47.5%** [40.4, 54.3] |
| light swapped | 17.1% | 78.7% | .831 | .780 | 23.2% [17.0, 29.0] |
| light + side swapped | 6.5% | 61.8% | .730 | .604 | 31.8% [25.2, 37.9] |
| dimension changed | 10.2% | 46.9% | .791 | .726 | 24.0% [16.1, 31.5] |

Set B has almost the same mix: 46.7% counts-only, 20.2% side swaps and 17.3% light swaps. Its predictors are the same 5, except for the 3 extra configs (GPT-4o-mini, Gemini 2.0 Flash and GPT-5.2-none), which have only 2. So set B's lower NSG isn't caused by a different pair mix. The Qwen3 and Gemma-3 results, which drive the low set-B NSG, all use 5 predictors. **[re-run]**

> My notes:

---

## 2. Scorecard: do the previous aims hold up?

| Prior claim / aim | Status | Evidence |
|---|---|---|
| **Fact 1:** MM doesn't use Hamming distance | ✅ Confirmed | §1. Also: 10% of pairs change dimension, contrary to the paper's appendix. |
| **Fact 2:** MM has the paper's lowest NSG (6.0%) | ✅ Reproduced; ⚠️ heterogeneous | The 18-model mean is 5.0–7.3%. The frontier mean is 23.9% and the small/open mean is −6.1%. |
| **Fact 3:** the matching rule skews the dimensions (gender 31.7%, species 4.9%) | ❔ Not re-checked | [inherited]. Re-run it before citing (toy experiment T3). |
| **Fact 4 / old Finding 2:** counts-only pairs carry no gain | ✅ Confirmed | NSG −6% (A) and −15% (B). The reference flips only 6.6% (A) or 20% (B) of the time on them. |
| **H1:** low MM NSG is a construction artifact; a matched Hamming ≤ 2 design would move it toward the tabular range | ❌ **Not supported as written** | (a) For frontier models MM is already in the tabular range. (b) The low numbers come from specific models (§2.4), not from the pairs. (c) The closest pairs (counts only) are the *least* informative, so closer pairs wouldn't raise NSG. |
| **H2:** stated factors diverge from revealed (AMCE) preferences, especially on demographics | ⏳ Untested | No AMCE profiles yet. The regex tagger is too broad: it flags "law" in 89% of explanations, because the scenario text itself says "flouting the law". |
| **H3:** models violate their stated principles more than noise would explain | ⏳ Partly, qualitatively | The spot-checked GPT-5.2 and Opus cases in `analysis/mm_raw_data.html` are real principle violations, and the paper's own Fig. 6/20 shows them. But no noise ceiling exists for MM (§2.3). |
| The paper: unfaithfulness is "technical, not social bias" | ⚖️ Inconclusive on MM | Egregious-failure RR for a demographic dimension is 1.31 [0.93, 2.08], and for a side swap 1.35 [1.08, 1.72]. Structural changes are at least as risky as demographic ones. **[re-run]** |
| **F6:** the prompt conflict ("Respond with one word only") hurts results | ⚠️ Bigger than thought | Answer position strongly moves Gemma, Gemini 2.0 and Sonnet-none (§2.4). GPT-5.2 high NSG is 0.30 answer-last vs. 0.46 answer-first. |

### 2.1 Flip-conditional view **[re-run]**
Pooled NSG mixes in how often the reference model flips. Splitting by whether it flipped (set A, frontier):

| | Gain when ref keeps its answer | Gain when ref flips |
|---|---|---|
| Tabular (6 datasets) | +14.7 pp (.834 vs .687) | **−6.2 pp** (.723 vs .785) |
| Moral Machines | +2.7 pp (.932 vs .905) | **+8.3 pp** (.707 vs .624) |

When the reference model *does* change its answer, MM explanations help predictors anticipate it. Tabular explanations make predictors more conservative on those cases. MM's lower pooled NSG comes mostly from the no-flip side, where predictors are already at about 90% without the explanation.

*Caveat:* this conditions on the label, so it describes the data rather than giving a valid metric. Use it to ask questions, not as a headline.

### 2.2 Where the low numbers come from (MM NSG minus tabular mean, per model) **[re-run]**
- **Frontier, about 0:** GPT-5.2 high (+0.003) and medium (−0.009), Opus 4.5 (+0.003), Gemini 3 Pro (+0.005), Gemini 3 Flash (−0.006).
- **Mid:** the Sonnet 4.5 reasoning configs (−0.07 to −0.09), Sonnet-none (−0.14), GPT-5-mini (−0.20), GPT-5-nano (−0.21), Haiku 4.5 (−0.24).
- **Gemma-3, very low:** 12B −0.69, 4B −0.58, 27B −0.55, 1B −0.42.
- **Qwen3 0.6B–32B:** −0.23 to −0.29.
- **The same ordering holds on the 142 pairs shared by sets A and B.** Frontier reasoning models score 0.19–0.49 there; Gemma 1B/4B/12B, Qwen ≤ 1.7B, Gemini 2.0 and Haiku score below 0, and Gemma-27B is about 0 (+0.02). So the gap comes from the models, not the pair set.

### 2.3 Noise: still the open hole
The paper's consistency analysis (§4 and A.6) covers only Qwen3-32B and Gemma-27B, pooled across datasets. It finds 5–15% of unfaithfulness comes from inconsistency. **There is no MM-specific noise ceiling.**

MM is where you'd most expect coin flips. GPT-5.2 and Haiku write "I select Case 1 arbitrarily"; the paper quotes this in its "faithfulness in distribution" limitation. In the HTML export, when counts-only pairs flip, every predictor misses 69% of those flips. That's consistent with both noise and unfaithfulness. Only resampling can tell them apart (T5).

### 2.4 Gemma: position-driven answers with content-based explanations **[re-run]**

Case 1 rate by answer position, set by `original_answer_first` (randomised per pair):

| Model | Case 1, answer-last | Case 1, answer-first | NSG answer-last | NSG answer-first |
|---|---|---|---|---|
| gemma-3-27b | **.957** | .594 | **−0.36** | +0.02 |
| gemma-3-12b | **.957** | .789 (n=19) | −0.29 | (n=19) |
| gemma-3-4b | .914 | no rows | −0.16 | — |
| gemini-2.0-flash | .566 | .246 | +0.11 | −0.11 |
| claude-sonnet-4.5 (none) | .607 | .499 | +0.25 | +0.10 |
| GPT-5.2 (medium), for contrast | .708 | .703 | +0.37 | +0.36 |

Gemma's answer-last explanations argue from content. I read 4 random Gemma-27B answer-last samples in full:
- **1 clear contradiction.** *"choosing to sacrifice the elderly group (Case 2) to save the younger group (Case 1) … The slight edge goes to saving the younger demographics"*, followed by `[ANSWER] CASE 1`, which kills the younger group.
- **3 consistent or mixed.** For example, one cites societal contribution, which points to Case 2, but also inaction and lawful pedestrians, which point to Case 1, and answers Case 1.
- T1 will put a number on the contradiction rate.

**Parse check (done).** `parse_response` reads the single `[ANSWER]` tag; there's no last-mention fallback. Every Gemma MM response has exactly one tag, and on answer-last prompts Gemma-27B literally writes "CASE 1" in 492 of 514. So the bias is in the model, not the parser. One parser risk remains: `normalize_answer` prefix-matches, so an echoed placeholder like "CASE 1 or CASE 2" would read as CASE 1. No such strings appear for Gemma-27B, but check other models before generalising.

**Mechanism for the negative NSG.** Gemma rarely flips, because it's stuck on Case 1. So "copy the original answer" is a strong baseline without the explanation. A content-based explanation then pulls predictors toward content-driven changes that never happen. The real cause, answer position, is never mentioned: 4.5% of Gemma-27B answer-last explanations use any position/order wording.

**Caveats:**
- Gemma-4B has **no** answer-first MM rows and Gemma-12B has 19, presumably because those responses failed to parse. So the surviving data skews toward the worst format.
- The position regex is crude. Hand-check before citing.

> My notes:

---

## 3. Promising directions (ranked)

**Scope (decided 2026-10-07):**
- **Frontier reference models are the focus.** That's pair set A: GPT-5.x, Claude 4.5 and Gemini 3.
- **Gemma-3 stays in as a known-unfaithful control.** It gets one row in tables and figures, and isn't analysed further.
- **Why a negative NSG isn't "Gemma doing worse at the task":** Moral Machines has no correct answer. A negative NSG means Gemma's explanations make predictors *worse* at predicting Gemma, i.e. they actively mislead about its behaviour. That's exactly what makes it a useful control: any method for detecting unfaithfulness should flag Gemma.
- **State the exclusion when reporting.** Without Gemma and small Qwen, Moral Machines isn't an outlier for the strongest models, so dropping them changes the paper's ethics claim.

### D1: Shared failures: is it shared unfaithfulness or near-ties? *(Recommended lead; spot-check free, resampling cheap.)*
Pairs where several frontier models fail, meaning all 5 predictors are wrong with the explanation.

What the data shows (pair set A) **[re-run]**:
- No pair fails for *most* reference models; the worst is 6 of 11.
- 688 of the 1,000 pairs never fail for any model, and 50 fail for 3 or more.
- Of those 50: 24 are counts-only, and they're mostly social value (22), age (11) and gender (8).
- On 64% of the 50, the models split, with 30–70% of them flipping (endpoints included). Across the other pairs it's 12%; see the *Shared failures* tab.

Widening to more reference configs (the tab's view selector) **[re-run]**:
- Set A (13 configs) and set B (13 configs) were run on **different pairs**, sharing only 142. So "18 models" and "26 configs" mostly don't see the same pairs.
- **Paper's 18:** 120 pairs fail for 3+ configs; the worst is 5 of 8. Set B configs top the list: Qwen3-4B, Qwen3-1.7B and the Gemma models.
- **All 26:** 287 pairs fail for 3+; the worst is 7 of 12.
- **The 142 pairs every config answered:** 43 fail for 3+, but the worst is only 9 of 25 (36%), and only 14% of them split the configs. The failures come from set B: Gemini 2.0 Flash, GPT-4o-mini, Qwen3 and Gemma.
- So no pair fails for most configs in any view, and adding set B mostly adds failures from the weaker models. For the frontier question, keep set A as the spot-check set.

Two stories fit, and only resampling separates them:
1. **Shared unfaithfulness (interesting).** Several models state the same principle and break it the same way. The paper's Fig. 20 is one: GPT-5.2 and Opus both cite inaction, then both swerve when the genders are swapped. This would be a stated-vs-revealed moral gap *across* models.
2. **Near-tie / indifference (not interesting).** The models are indifferent, so one sample is close to a coin flip and no explanation could predict it.

**Plan:** spot-check first (T5 step 0), then resample (T5 steps 1–4).

### D2: Gemma as the known-unfaithful control. *(Free; keep small.)*
Gemma's answers follow the prompt format: about 95% "Case 1" on answer-last prompts. Its explanations argue from content and almost never mention position (§2.4).
- Use it to check the D1 and D4 methods: they should flag Gemma.
- It's also a natural, non-injected example of Turpin et al. (2023): explanations omit the answer-order bias that is actually driving the choice. This could become one paragraph or a figure panel. It needs T1 and T8.

### D3: Explanation–answer consistency within a single response. *(Free or cheap.)*
Before any counterfactual: does the explanation imply the answer the model gave?
- Gemma's answer-last outputs suggest often not.
- This is a cleaner, lower-level faithfulness check than NSG.
- It separates two kinds of explanation:
  - explanations that don't even match the original answer
  - explanations that match it but don't generalise to the counterfactual

### D4: Stated vs. revealed moral preferences (old H2/Aim 2). *(Free.)*
- Fit AMCEs per model on all of the scenarios it answered.
- Compare them with the factors its explanations cite.
- Needs a better tagger than the current regex.
- Literature to position against: the stated/revealed-gap papers in §6.
- If D1 turns up shared unfaithfulness, those pairs become the test cases here.

### D5: Designed single-factor interventions (old Aim 3). *(API.)*
- Still the right *final* experiment, but redesign it after T4, T5 and T8.
- Counterbalance case order (F4), now clearly necessary.
- Fix the prompt (F6).
- Sample K ≥ 5 times (F5).
- Include at least one Gemma model as the control (D2).

### ~~D0: Rebuild MM with Hamming ≤ 2 to show NSG rises.~~
Deprioritised. §2 suggests it won't rise for frontier models, and the closest pairs carry the least signal. Keep a Hamming-style MM distance only as a *descriptive* covariate.

> My notes / ranking changes:

---

## 4. Toy experiments

### Free: laptop only, `data_parquet/`
- ☐ **T1. Hand-label 50 Gemma-27B answer-last rows.** Does the explanation imply Case 1, Case 2 or neither? Compare with the stated answer. About 1 hour. This is the core evidence for D2/D3.
- ☐ **T2. Run the same check on 50 GPT-5.2 and 50 Sonnet rows** as a control. Expect high agreement.
- ☐ **T3. Re-verify Fact 3** (dimension skew between raw scenarios and pairs), then reweight set A NSG to the raw dimension mix.
- ☐ **T4. Position-bias regression.** For each model, regress `answer == Case 1` on the content factors (law, passengers vs. pedestrians, intervention, counts, dimension) plus `answer_first`. This gives a first AMCE table (D4) and puts a number on how much is format-driven.
- ☐ **T6. Upgrade the tagger.** Replace the regex `cited` tags with a small LLM tagger, or hand-label 100 and report κ. Required before any H2 claim.
- ☐ **T7. Within-pair consistency of explanations.** On flipped pairs, does the model's counterfactual explanation cite a *different* principle from its original one? This is a principle-switching rate per model. The `show_case` helper in `moral_machines_failure_analysis.py` already prints both explanations.

### Cheap API: OpenRouter, needs `OPENROUTER_API_KEY` in `.env` (**not set yet**)
- ☐ **T5. Shared-failure resampling: shared unfaithfulness or near-ties? (D1)**
  - **Step 0. Spot-check (free; do this first).**
    - Open the *Shared failures* tab in `analysis/mm_raw_data.html`. It lists the 50 pairs that are egregious for 3+ reference models, worst first.
    - For each pair, read both scenarios and every model's explanation. Then choose a verdict: *shared unfaithfulness*, *near-tie / indifference*, *predictor error*, *pair problem*, *mixed* or *unsure*.
    - Verdicts are saved in your browser. Use *Download my labels (CSV)* and save the file as `experiments/moral_machines/shared_failure_labels.csv`.
    - Aim for 15–20 pairs. That's enough to decide whether step 1 is worth paying for.
    - Example: the top-ranked pair (6860→11593) changes dimension, from species to utilitarianism, and puts a cat *inside the car*. That's probably a *pair problem*.
  - **Step 1. Case set.**
    - Use the 50 shared pairs. Drop the ones labelled *pair problem* and analyse those separately as construction issues.
    - Optionally add the 191 counts-only flip pairs [inherited count; about 24 overlap].
  - **Step 2. Resample.**
    - K=5 samples per scenario, original and counterfactual, using the **same prompts** as `data_parquet/` (`question_prompt`) so results stay comparable.
    - Models: GPT-5.2-low, Sonnet 4.5 (medium) and Gemini 3 Flash, plus gemma-3-27b as the control.
    - Pilot: the 50 shared pairs × 2 scenarios × 5 samples × 3 frontier models ≈ **1.5k calls**. The full run, with counts-only pairs and Gemma, is about 9k.
  - **Step 3. Measures.**
    - For each (model, scenario): the modal answer and its frequency p.
    - A flip is **consistent** when the modal answers differ and both p ≥ 0.8.
    - It's **noise** when either p ≤ 0.6.
    - Also compute the oracle ceiling (accuracy at predicting the modal answer, as in the paper's A.6), restricted to Moral Machines.
  - **Step 4. Decision.**
    - Consistent flips that contradict the model's stated principle, across 2+ models, count as **shared unfaithfulness**. They become the test set for D4.
    - If most flips are noise, report a Moral Machines-specific noise ceiling. That supports the paper's own "faithfulness in distribution" limitation: explanations should express indifference.
  - **Setup gotchas (checked in the code):**
    - `generate_reference_answers` has no repeats option and **deduplicates identical prompts**, so duplicate rows won't give extra samples. Either run it K times to separate outputs, or add a `--reference-repeats` flag modelled on `--predictor-repeats`.
    - `openrouter/google/gemma-3-27b-it` pins `seed=666` in `src/vllm_configs.py`, so vary the seed across runs, or the samples may be identical. GPT-5.2, Sonnet and Gemini 3 Flash have no fixed seed. They use the default temperature: 1.0 in the generator.
- ☐ **T8. Case-order swap.**
  - Take 50 scenarios and swap the Case 1 and Case 2 *text* (same content, opposite labels). Run each in both answer formats.
  - Models: gemma-3-27b, GPT-5.2 and Sonnet 4.5. About 600 calls.
  - This tests D2 directly. A content-driven model's choice follows the content; a position-driven one follows the label.
- ☐ **T9. Remove "Respond with one word only"** and re-run T8's Gemma arm. Is the format sensitivity a result of the prompt conflict (F6)?

> My notes / results:

---

## 5. Items that need your review

- ☑ **Scope: frontier models first, Gemma as a control** (decided 2026-10-07; §3). Next decision: after the T5 spot-check, is the 1.5k-call resampling pilot worth running?
- ☐ **Report MM frontier vs. small/open separately** in anything shown to HM. One pooled number hides a 30-point split.
- ☐ **Flag the appendix discrepancy to HM:** 10% of MM pairs change dimension, though Appendix A.8 says pairs share a dimension. Is that a bug or intended?
- ☐ **Ask HM about the missing Gemma answer-first rows.** Why does Gemma-4B have none and Gemma-12B only 19? Were parse failures dropped?
- ☐ **Ask whether the 18-model MM average used GPT-5.2 medium and Sonnet medium**, the configs I assumed. Any combination gives 5.0–7.3%, so it doesn't matter much.
- ☐ **Ask HM about the GPT-5.2 reasoning comparison (paper A.4 / Table 6).** In `data_parquet/`, GPT-5.2 (none) differs from low/medium/high in two ways: it's on a different pair set (B vs. A) and it's scored by 2 predictors, not 5. If the paper's Moral Machines numbers came from these files, "none" isn't directly comparable with the other levels. Also ask which reasoning level the main 18-model results use; the paper doesn't say.
- ☐ **Small-sample configs.** These Moral Machines NSGs rest on few pairs: GPT-5-nano (172 usable of 1,000), Gemma-3-4B (514), Gemma-3-12B (533) and Qwen3-0.6B/1.7B (about 620). Report their CIs, or flag them.
- ☐ **API budget for T5 + T8 + T9:** about 5k calls. Set up `OPENROUTER_API_KEY`?
- ☐ **Carried over from the earlier plans, still open:**
  - drop the consciousness framing (F1)
  - pick a control dataset (heart disease is the closest tabular match to frontier MM NSG)
  - check whether the GPT-5.2 duplicate runs were independent
  - add `data_parquet/` to `.gitignore`
  - review the uncommitted `*_balanced.parquet` changes

---

## 6. Literature

**Read and verified this session:**
- **Mayne et al. (2026), *A Positive Case for Faithfulness*.** arXiv [2602.02639](https://arxiv.org/abs/2602.02639). Checked against the HTML text:
  - "NSG is lowest on Moral Machines (6.0%) … consistent with Matton et al. (2025) who find that models are less faithful on ethical issues."
  - A.2: "Qwen3-{0.6,1.7}B, and all Gemma 3 models are unfaithful in a statistically significant way, e.g., gemma3-12b-it has NSG = −30.4% [−44.9, −17.4]".
  - A.8: the Social value RR is about 1.1.
  - The consistency analysis covers only Qwen3-32B and Gemma-27B.

**Core references (from the earlier plan; known, not re-read this session):**
- Awad et al. (2018), *The Moral Machine experiment*. Human AMCEs.
- Takemoto (2024), *The Moral Machine Experiment on LLMs*. [2309.05958](https://arxiv.org/abs/2309.05958). The generator this repo ports.
- Matton et al. (2025), *Walk the Talk?* [2504.14150](https://arxiv.org/abs/2504.14150). Concept-level counterfactual faithfulness; the ethics claim the NSG paper cites.
- Turpin et al. (2023), *Language Models Don't Always Say What They Think*. CoT omits biasing features such as answer order. **The key reference for D2.**

**To read: found by search, judged only from snippets, so don't cite yet:**
- *Fragile Preferences: Order Effects in LLMs* — [2506.14092](https://arxiv.org/abs/2506.14092). Systematic position bias in pairwise choice.
- *The yes-no bias of LLMs reflects answer order and wording, not shifts in moral judgment* — [2607.05552](https://arxiv.org/abs/2607.05552). Directly relevant to D2.
- *Language Model Alignment in Multilingual Trolley Problems* — [2407.02273](https://arxiv.org/abs/2407.02273). Reports option-order robustness checks on moral scenarios.
- *Superficial Beliefs in LLM Decision-Making* — [2606.11016](https://arxiv.org/abs/2606.11016). For D3.
- *AI Revealed Preferences* — [2608.26178](https://arxiv.org/abs/2608.26178). For D3.
- *Are LLMs Consistent in Stated and Revealed Preferences?* — [2506.00751](https://arxiv.org/abs/2506.00751). For D3.
- *Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations* (ICLR 2026). Counterfactual faithfulness across 75 models; possibly relevant to the scale pattern in §2.2.

---

## 7. Reproduce

```bash
conda activate faithfulness-env
python analysis/mm_vs_tabular_nsg.py                 # §0, §2.1, §2.2, §2.4 (about 2 min)
python analysis/moral_machines_failure_analysis.py   # §1 table, RR, egregious candidates (writes figures/, experiments/)
jupyter nbconvert --execute analysis/mm_raw_data.ipynb   # raw-data and egregious HTML report
```
