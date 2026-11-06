# Gemma answer-format test: fresh runs, same scenario both ways (draft plan)

*Draft, 2026-10-27. Nothing has run yet. Items marked **[REVIEW]** need your decision, and each has a proposed
default. Reply "go", or tell me what to change.*

## Why

Gemma's hidden cause is now the headline result, but all of the evidence comes from the paper's single stored run.
In that run each scenario was asked in only **one** format, so "96% Case 1 when answering last vs. 59% when answering
first" compares *different* scenarios.

This test asks Gemma the **same scenario in both formats**, in fresh calls:
- **If its choice flips with the format alone,** that's direct evidence that the format drives its answer, not just a
  correlation.
- **Repeats** also answer the debrief's consistency question (same prompt several times) for Gemma.

## Questions

1. **Does the format effect hold in fresh runs?** Compare the Case 1 rate with answer-first vs. answer-last prompts.
2. **Is it causal within a scenario?** How often does Gemma's choice change when only the format changes?
3. **How consistent is Gemma?** Agreement across 3 samples of the identical prompt.
4. **Does the fresh run match the paper's?** Agreement with the stored answers on unchanged paper prompts.

## Setup

**Model:** `openrouter/google/gemma-3-27b-it`, which is already in `src/vllm_configs.py`.
- Temperature 1, no top-p/top-k.
- The paper ran Gemma locally with temperature 1.0, top-p 0.95 and top-k 64, so the fresh runs aren't identical to
  the paper's. Question 4 checks how much that matters.
- **[REVIEW]** Pass top-p 0.95 / top-k 64 through OpenRouter to match the paper more closely, if the provider accepts
  them? *(Proposed: yes, if the smoke test shows they're accepted; otherwise use defaults and say so.)*

**Prompts:** built with the repo's own `MoralMachines.create_reference_prompt(question, answer_last=…)`. The only
difference between the two formats is whether `[ANSWER]` is requested first or last. The smoke test checks that the
rebuilt prompts match the paper's stored prompts.

**Scenarios: 100 from Gemma's answered pool.**
- Sample from experiment 1's **held-out** scenarios, stratified by theme and car vs. pedestrians.
- Each scenario is asked in **both** formats, 3 samples each.
- **[REVIEW]** 100 scenarios × 2 formats × 3 samples = **600 calls**? *(Proposed: yes.)*

**Calibration set:**
- **20 of the paper's original prompts, unchanged** (in whatever format each was asked), 3 samples each, for 60
  calls.
- Compare with the stored answers.
- **Reading the result:** if fresh answers agree with the stored ones about as often as fresh samples agree with each
  other, the serving setup didn't change behaviour.

**Optional control: a model that shouldn't care about format.**
- Gemini 3 Flash on the same 100 scenarios × 2 formats × 1 sample, for 200 calls.
- Its fitted answer-position effect was small (−0.2), so its flip rate under format changes should be low. That shows
  the test itself doesn't create flips.
- **[REVIEW]** Include it? *(Proposed: yes; it makes the Gemma result much easier to read. GPT-5.2 would cost about
  10× more.)*

## Metrics

- **Case 1 rate by format**, pooled over scenarios and samples, with a 95% bootstrap over scenarios.
- **Format flip rate:** the share of scenarios whose majority answer (over 3 samples) differs between the two formats.
  - Also: the share of format flips that go "towards Case 1 when answering last".
- **Repeat consistency:** the share of scenarios where all 3 samples agree, per format.
- **Calibration:** fresh vs. stored agreement on the 20 unchanged prompts.
- **Content check:** for scenarios where the format changes the answer, do Gemma's explanations still argue from
  content (spot-check 10)? This connects to the "argues for the passengers, then sacrifices them" example.

## What would change the write-up

| Result | Reading |
|---|---|
| Case 1 rate much higher when answering last, and many within-scenario flips | The hidden cause is causal and reproducible: the headline stands, and gets stronger |
| Similar rates in both formats, but stored answers reproduce on the calibration set | The format effect was specific to the paper's run (scenarios or sampling): weaken the Gemma claim |
| Stored answers don't reproduce on the calibration set | The serving setup changed Gemma's behaviour. The result can't settle the question; report it as inconclusive |

## Cost

Gemma 3 27B on OpenRouter costs $0.08 / $0.35 per million input / output tokens. A reference call is about 700 tokens
in and 300–500 out, so roughly $0.0002 per call.
- Gemma: 600 + 60 calls ≈ $0.15
- Gemini 3 Flash control: 200 calls, about $0.0015 each ≈ $0.30
- **Total: under $1.** The $50 key has $2.53 used so far.

A $6 cap is enforced after each batch, as before. The 10-call smoke test measures the real cost first.

## Steps

1. **[CLAUDE]** Write `analysis/mm18_format_run.py`, run a 10-call smoke test and report:
   - prompt match with the stored prompts
   - parse rate
   - cost per call
   - whether top-p/top-k are accepted
2. **[YOU] Gate:** check the example prompt pair. They should be identical except for where `[ANSWER]` sits.
3. **[CLAUDE]** Run the main set, the calibration set and the control (about 860 calls), then analyse in
   `analysis/mm18_format.py`.
4. **[CLAUDE]** Add the results to the report's Experiments tab and one slide to the deck.

## Review checklist
- [ ] Match the paper's sampling (top-p 0.95, top-k 64) if the provider accepts it
- [ ] 100 held-out scenarios × 2 formats × 3 samples (600 calls)
- [ ] 20-prompt calibration set against the stored answers (60 calls)
- [ ] Gemini 3 Flash control (200 calls)
