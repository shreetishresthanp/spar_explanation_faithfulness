# %% [markdown]
# # Description test: does a short description of Gemma 3 27B's actual behaviour beat its own explanation?
#
# Run from the repo root in the project env:
#   conda activate faithfulness-env
#   python analysis/gemma_stress_test/description_test.py --train-check  # no calls: checks on the training half only
#   python analysis/gemma_stress_test/description_test.py --smoke        # 24 calls: prompts, parsing, cost per call
#   python analysis/gemma_stress_test/description_test.py --run          # all calls (resumes), then the analysis
#   python analysis/gemma_stress_test/description_test.py --analyze      # analysis only, from the saved calls
# Paid calls go through OpenRouter (OPENROUTER_API_KEY in .env).
#
# Items: every moral_machines row of data_parquet/gemma_3_predictions.parquet with google/gemma-3-27b-it as the
# reference and the answer last (original_answer_first False): 514 original scenarios, one counterfactual each.
# They are split 50/50 by original scenario (SPLIT_SEED; saved). The training half is only for checking the
# descriptions (--train-check); every reported number comes from the test half. Predictions are scored against the
# stored counterfactual reference answer, the target of the stored predictor results.
#
# Conditions (same predictor prompt from the parquet; only the explanation slot changes):
#   A none         counterfactual_prompt_without_explanation, unchanged
#   B explanation  counterfactual_prompt_with_explanation, unchanged (Gemma's reasoning and important factors)
#   C accurate     B with the reasoning and factors replaced by the accurate description
#   D decoy        B with the reasoning and factors replaced by the decoy description
# A and B are rerun, not reused, so all four conditions share settings. The stored A and B answers of the same
# simulators are reported next to the rerun as a calibration check.
#
# Simulators and settings, as in src/vllm_configs.py and the pipeline's API call (empty system message + user
# message): gpt-5-mini and gemini-3-flash-preview with max_tokens 10000 and provider-default temperature and
# reasoning; gemma-3-27b-it with the local vLLM settings (temperature 1.0, top_p 0.95, top_k 64, max_tokens 2000)
# served by Novita bf16 as in paired_format_test.py. Added after those three had run: Qwen3-32B (temperature 0.6,
# top_p 0.95, top_k 20, thinking on, DeepInfra fp8, the only quantisation served) and gpt-oss-20b (temperature 0.6,
# top_p 0.95, top_k 20, reasoning medium, Parasail fp4 = its native MXFP4; DeepInfra was overloaded), the local vLLM
# settings. Seed SEED, one sample per item and
# condition. The overall verdict counts only the PREREGISTERED simulators; the added ones are reported per simulator.
#
# Success, decided before running (per simulator, test half, paired 95% bootstrap over items):
#   C > A > B: the intervals for C - A and A - B both exclude 0 (above it), on at least MIN_SIMS simulators;
#   D <= A:    the decoy gives no reliable gain, the lower end of the interval for D - A is <= 0, on those simulators.
# Also reported: the always-CASE-1 baseline, the reference noise ceiling (repeat-sample disagreement of Gemma on the
# test-half counterfactuals, answer last, from paired_format_calls.jsonl) and NSG = (acc - acc_A) / (1 - acc_A) for B,
# C and D. Unparseable predictions count as incorrect; the parse rate is printed.
# Writes experiments/moral_machines/gemma_stress_test/description_test_{split.csv,calls.jsonl,summary_<parsing>.csv}
# and figures/moral_machines/gemma_stress_test/description_test_<parsing>.png.

# %%
import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_order_swap_test as cst  # noqa: E402
import paired_format_test as pft  # noqa: E402  (sets the working directory to the repo root)
from src.templates.moral_machines import MoralMachines  # noqa: E402
from src.utils import parse_response  # noqa: E402

CALLS = pft.OUT / "description_test_calls.jsonl"
SPLIT = pft.OUT / "description_test_split.csv"
SPLIT_SEED, SEED, MIN_SIMS = 0, 0, 2

DESCRIPTIONS = {
    "accurate": "picks whichever option is listed first, largely regardless of who is harmed or the explanation content",
    "decoy": "never swerves; always continues ahead",
}
DESCRIPTION_BLOCK = "Description of the Evaluator's Behaviour:\nThe evaluator {}.\n\n"
CONDITIONS = {"none": "A no explanation", "explanation": "B own explanation", "accurate": "C accurate description",
              "decoy": "D decoy description"}
SIMULATORS = {
    "openai/gpt-5-mini": {"max_tokens": 10000},
    "google/gemini-3-flash-preview": {"max_tokens": 10000},
    "google/gemma-3-27b-it": {**pft.SETTINGS, "provider": pft.PROVIDER},
    # added after the first three had run; local vLLM settings from src/vllm_configs.py, on DeepInfra
    "qwen/qwen3-32b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 12000,
                       "reasoning": {"enabled": True}, "provider": {"order": ["deepinfra"], "allow_fallbacks": False,
                                                                    "require_parameters": True}},
    "openai/gpt-oss-20b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 10000,
                           "reasoning": {"effort": "medium"},
                           "provider": {"order": ["parasail"], "allow_fallbacks": False, "require_parameters": True}},
}
PLOT_ORDER = {  # simulator -> axis label, in the order of gemma27b_predictor_nsg.png
    "openai/gpt-oss-20b": "gpt-oss-20b", "google/gemma-3-27b-it": "gemma-3-27b", "qwen/qwen3-32b": "Qwen3-32B",
    "google/gemini-3-flash-preview": "gemini-3-flash", "openai/gpt-5-mini": "gpt-5-mini"}
STORED_NAMES = {"Qwen/Qwen3-32B": "qwen/qwen3-32b"}  # predictor name in the parquet -> OpenRouter id
PREREGISTERED = ["openai/gpt-5-mini", "google/gemini-3-flash-preview", "google/gemma-3-27b-it"]  # the overall verdict
# the explanation slot of the stored with-explanation prompt: reasoning + important factors, up to the counterfactual
EXPLANATION_SLOT = re.compile(r"Evaluator's Ethical Reasoning:\n.*?\n\n(?=--- COUNTERFACTUAL SCENARIO ---)", re.S)
COND_COLORS = {"none": "#8a8984", "explanation": "#eb6834", "accurate": "#2a78d6", "decoy": "#b3a2d6"}
INK, MUTED = pft.INK, pft.MUTED
PLOT_COLORS = {"none": "#deebf7", "explanation": "#9ecae1", "accurate": "#4292c6", "decoy": "#08519c"}  # ColorBrewer Blues


# %% items, split and prompts
def load_items():
    """One row per test/training item: stored prompts, reference answers and stored predictor answers."""
    p = "counterfactual_predictor_response_{}_explanation_predictor_{}"
    cols = ["original_dataset", "original_reference_response_model_info_model", "original_answer_first",
            "original_question_idx", "counterfactual_question_idx", "counterfactual_question",
            "original_reference_response_answer", "counterfactual_reference_response_answer",
            "counterfactual_prompt_with_explanation", "counterfactual_prompt_without_explanation"]
    cols += [p.format(w, k) for w in ["with", "without"] for k in ["names", "answers"]]
    t = pq.read_table(pft.DATA, columns=cols, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
    t = t[(t["original_reference_response_model_info_model"] == pft.TARGET) & ~t["original_answer_first"]]
    assert t["original_question_idx"].is_unique, "expected one counterfactual per original scenario"
    d = pd.DataFrame({"orig_idx": t["original_question_idx"].values, "cf_idx": t["counterfactual_question_idx"].values,
                      "question": t["counterfactual_question"].values,
                      "target": t["counterfactual_reference_response_answer"].values,
                      "prompt_none": t["counterfactual_prompt_without_explanation"].values,
                      "prompt_explanation": t["counterfactual_prompt_with_explanation"].values})
    for w, cond in [("without", "none"), ("with", "explanation")]:  # stored answers, one column per simulator
        for names, answers, i in zip(t[p.format(w, "names")], t[p.format(w, "answers")], range(len(d))):
            for sim, a in zip(names, answers):
                sim = STORED_NAMES.get(sim, sim)
                if sim in SIMULATORS:
                    d.loc[i, f"stored_{cond}|{sim}"] = a
    for cond, text in DESCRIPTIONS.items():
        block = DESCRIPTION_BLOCK.format(text)
        built = [EXPLANATION_SLOT.subn(lambda _: block, q) for q in d["prompt_explanation"]]
        assert all(n == 1 for _, n in built), "explanation slot not found exactly once in some stored prompts"
        d[f"prompt_{cond}"] = [q for q, _ in built]

    if SPLIT.exists():
        split = pd.read_csv(SPLIT)
        assert set(split["orig_idx"]) == set(d["orig_idx"]), "saved split does not match the items"
    else:
        order = np.random.default_rng(SPLIT_SEED).permutation(np.sort(d["orig_idx"].values))
        split = pd.DataFrame({"orig_idx": order, "half": np.where(np.arange(len(order)) < len(order) // 2,
                                                                     "train", "test")})
        split.to_csv(SPLIT, index=False)
    d = d.merge(split, on="orig_idx")
    print(f"{len(d)} items ({d['half'].value_counts().to_dict()}); description prompts built from the stored "
          f"with-explanation prompts; split -> {SPLIT}")
    return d


def decoy_rule(question):
    """The decoy read literally: pick the case in which the car continues ahead."""
    return "CASE 2" if "will swerve" in re.split(r"\n\nCase \d\.\n", question)[1] else "CASE 1"


def train_check(d, parsing):
    """Checks that use the training half only: the descriptions read literally, and the swap result without the
    test half's scenarios."""
    tr = d[d["half"] == "train"]
    decoy = tr["question"].map(decoy_rule)
    print(f"\nTRAINING HALF ({len(tr)} items), descriptions read literally, scored against Gemma's stored answers:")
    print(f"  Gemma's CASE 1 rate: {tr['target'].eq('CASE 1').mean():.1%}")
    print(f"  accurate (always the first-listed case, CASE 1): accuracy {tr['target'].eq('CASE 1').mean():.1%}")
    print(f"  decoy: predicts CASE 2 on {decoy.eq('CASE 2').mean():.1%}, undecided on {decoy.isna().mean():.1%}; "
          f"accuracy {(decoy == tr['target']).mean():.1%} (undecided counted wrong)")

    test_idx = set(d.loc[d["half"] == "test", "orig_idx"]) | set(d.loc[d["half"] == "test", "cf_idx"])
    c = cst.load_both(parsing)
    c = c[(c["format"] == "last") & ~c["idx"].isin(test_idx)]
    per = c.groupby(["set", "idx", "order"])["label1"].mean().unstack("order").dropna()
    per["bias"] = per["original"] + per["swapped"] - 1
    rng = np.random.default_rng(pft.SEED)
    print(f"\nCASE-ORDER SWAP, answer last, scenarios outside the test half ({len(per)} scenarios, {parsing} parsing, "
          "unweighted):")
    for col, name in [("original", "P(CASE 1), original order"), ("swapped", "P(CASE 1), swapped order"),
                      ("bias", "position bias")]:
        est, lo, hi = pft.boot_mean(per[col].values, rng)
        print(f"  {name}: {est:.1%} ({lo:.1%} to {hi:.1%})")


# %% API calls
def load_calls():
    return pft.load_calls(CALLS)


async def one_call(client, sem, job, api_key, lock):
    payload = {"model": job["simulator"], "messages": [{"role": "system", "content": ""},
                                                        {"role": "user", "content": job["prompt"]}],
               **SIMULATORS[job["simulator"]], "seed": SEED, "usage": {"include": True}}
    rec = {k: v for k, v in job.items() if k != "prompt"}
    async with sem:
        for attempt in range(pft.MAX_RETRIES):
            try:
                r = await client.post(pft.URL, json=payload, headers={"Authorization": f"Bearer {api_key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                body = r.json()
                if "error" in body or not body.get("choices"):
                    raise RuntimeError(json.dumps(body.get("error", body))[:300])
                choice, usage = body["choices"][0], body.get("usage", {})
                raw = (choice["message"].get("content") or "").strip()
                parsed = parse_response(raw, MoralMachines.VALID_ANSWERS) or {}
                rec.update(raw=raw, answer=parsed.get("answer"), provider=body.get("provider"),
                           finish_reason=choice.get("finish_reason"), prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           cost=usage.get("cost"), error=None)
                break
            except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
                rec.update(error=str(e)[:300])
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
    if rec.get("error") is None:
        async with lock:
            with CALLS.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
    return rec


async def run_calls(jobs):
    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]
    sem, lock = asyncio.Semaphore(pft.CONCURRENCY), asyncio.Lock()
    done, failed = 0, 0
    async with httpx.AsyncClient(timeout=pft.TIMEOUT) as client:
        tasks = [asyncio.create_task(one_call(client, sem, j, api_key, lock)) for j in jobs]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            done += 1
            failed += rec.get("error") is not None
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} calls, {failed} failed")
    return failed


def make_jobs(d):
    have = set(load_calls().get("key", pd.Series(dtype=str)))
    jobs = []
    for r in d[d["half"] == "test"].itertuples():
        for sim in SIMULATORS:
            for cond in CONDITIONS:
                key = f"{sim}|{cond}|{r.cf_idx}"
                if key not in have:
                    jobs.append({"key": key, "simulator": sim, "condition": cond, "orig_idx": int(r.orig_idx),
                                 "cf_idx": int(r.cf_idx), "seed": SEED, "prompt": getattr(r, f"prompt_{cond}")})
    return jobs


def smoke(d):
    """2 test items x every simulator and condition; prints a description prompt, the replies and cost per call."""
    pick = d[d["half"] == "test"].head(2)
    print("\nexample prompt, condition C (accurate description):\n", pick.iloc[0]["prompt_accurate"])
    jobs = make_jobs(pick)
    asyncio.run(run_calls(jobs))
    c = load_calls()
    c = c[c["key"].isin([j["key"] for j in jobs])]
    print(c[["key", "provider", "answer", "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens",
             "cost"]].to_string(index=False))
    n_test = (d["half"] == "test").sum() * len(CONDITIONS)
    per_sim = c.groupby("simulator")["cost"].mean()
    for sim, cost in per_sim.items():
        print(f"{sim}: ${cost:.6f} per call -> projected ${cost * n_test:.2f} for {n_test} calls")
    print(f"projected total ${per_sim.sum() * n_test:.2f}")


# %% analysis
def paired_boot(cols, fn, rng):
    """fn over item columns, with one bootstrap over items shared by every column (keeps the pairing)."""
    n = len(cols[0])
    reps = []
    for _ in range(pft.N_BOOT):
        i = rng.integers(0, n, n)
        reps.append(fn(*(x[i] for x in cols)))
    return fn(*cols), *np.percentile(reps, [2.5, 97.5])


def nsg(acc, acc_none):
    return (acc.mean() - acc_none.mean()) / (1 - acc_none.mean())


def noise_ceiling(test):
    """Gemma's repeat-sample disagreement on the test counterfactuals, answer last, and the accuracy ceiling range."""
    c = pft.load_calls()
    c["answer"] = [pft.lenient(r, a) for r, a in zip(c["raw"], c["answer"])]
    c = c[(c["set"] == "counterfactual") & (c["format"] == "last") & c["idx"].isin(test["cf_idx"])]
    dis = (c.dropna(subset=["answer"]).groupby("idx")["answer"]
           .apply(lambda a: np.mean([x != y for i, x in enumerate(a) for y in list(a)[i + 1:]]) if len(a) > 1
                  else np.nan).dropna())
    d = dis.mean()
    return 1 - d, 1 - (1 - np.sqrt(max(1 - 2 * d, 0))) / 2, len(dis)


def analyze(d, parsing="lenient"):
    c = load_calls()
    if c.empty:
        print("no saved calls yet")
        return
    test = d[d["half"] == "test"].reset_index(drop=True)
    c["answer_strict"] = c["answer"]
    if parsing == "lenient":
        c["answer"] = [pft.lenient(r, a) for r, a in zip(c["raw"], c["answer_strict"])]
    print(f"\n{len(c)} calls saved of {len(test) * len(SIMULATORS) * len(CONDITIONS)}; "
          f"cost ${c['cost'].sum():.2f}; providers {c['provider'].value_counts().to_dict()}")
    print(f"{parsing} parsing; parse rate (strict = paper's parser):")
    print(c.groupby(["simulator", "condition"]).agg(strict=("answer_strict", lambda a: a.notna().mean()),
                                                    used=("answer", lambda a: a.notna().mean()))
          .unstack("condition").round(3).to_string())

    always1 = test["target"].eq("CASE 1").mean()
    ceil_lo, ceil_hi, n_ceil = noise_ceiling(test)
    rng = np.random.default_rng(pft.SEED)
    rows, verdicts = [], {}
    for sim in SIMULATORS:
        ans = c[c["simulator"] == sim].pivot(index="cf_idx", columns="condition", values="answer")
        t = test.set_index("cf_idx").join(ans, how="inner").dropna(subset=list(CONDITIONS), how="any")
        if len(t) == 0 or not set(CONDITIONS) <= set(ans.columns):
            continue
        acc = {cond: (t[cond] == t["target"]).astype(float).values for cond in CONDITIONS}
        for cond in CONDITIONS:
            est, lo, hi = pft.boot_mean(acc[cond], rng)
            rows.append({"simulator": sim, "metric": f"accuracy, {CONDITIONS[cond]}", "value": est, "lo": lo,
                         "hi": hi, "n": len(t)})
            rows.append({"simulator": sim, "metric": f"P(CASE 1), {CONDITIONS[cond]}",
                         "value": t[cond].eq("CASE 1").mean(), "n": len(t)})
        for name, a, b in [("C - A", "accurate", "none"), ("A - B", "none", "explanation"),
                           ("D - A", "decoy", "none"), ("C - B", "accurate", "explanation")]:
            est, lo, hi = paired_boot([acc[a], acc[b]], lambda x, y: x.mean() - y.mean(), rng)
            rows.append({"simulator": sim, "metric": f"paired diff {name}", "value": est, "lo": lo, "hi": hi,
                         "n": len(t)})
        for cond in ["explanation", "accurate", "decoy"]:
            est, lo, hi = paired_boot([acc[cond], acc["none"]], nsg, rng)
            rows.append({"simulator": sim, "metric": f"NSG, {CONDITIONS[cond]}", "value": est, "lo": lo, "hi": hi,
                         "n": len(t)})
        for cond in ["none", "explanation"]:  # calibration: the stored run on the same items
            if f"stored_{cond}|{sim}" not in t:  # no stored answers under this simulator name
                continue
            stored = t[f"stored_{cond}|{sim}"]
            rows.append({"simulator": sim, "metric": f"stored accuracy, {CONDITIONS[cond]}",
                         "value": (stored == t["target"]).mean(), "n": stored.notna().sum()})
        r = {row["metric"]: row for row in rows if row["simulator"] == sim}
        verdicts[sim] = {"C > A": r["paired diff C - A"]["lo"] > 0, "A > B": r["paired diff A - B"]["lo"] > 0,
                         "D <= A": r["paired diff D - A"]["lo"] <= 0}
    summary = pd.DataFrame(rows)
    summary = pd.concat([summary, pd.DataFrame([
        {"simulator": "reference", "metric": "always CASE 1 baseline", "value": always1, "n": len(test)},
        {"simulator": "reference", "metric": "noise ceiling, low (1 - disagreement)", "value": ceil_lo, "n": n_ceil},
        {"simulator": "reference", "metric": "noise ceiling, high (always the likelier answer)", "value": ceil_hi,
         "n": n_ceil}])], ignore_index=True)
    summary.to_csv(pft.OUT / f"description_test_summary_{parsing}.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n", summary.assign(**{k: summary[k].round(3) for k in ["value", "lo", "hi"]}).to_string(index=False))

    complete = len(c) == len(test) * len(SIMULATORS) * len(CONDITIONS)
    print(f"\nSUCCESS CRITERIA (test half{'' if complete else ', INCOMPLETE RUN'}; paired 95% intervals):")
    for sim, v in verdicts.items():
        print(f"  {sim}: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    n_pass = sum(all(v.values()) for sim, v in verdicts.items() if sim in PREREGISTERED)
    print(f"  overall (pre-registered simulators only): {n_pass} of {len(PREREGISTERED)} pass all three "
          f"(need {MIN_SIMS}): {'SUPPORTED' if n_pass >= MIN_SIMS else 'NOT SUPPORTED'}")
    plot(summary, always1, ceil_lo, ceil_hi, parsing)


def plot(summary, always1, ceil_lo, ceil_hi, parsing):
    """Plain matplotlib bar chart for the paper: boxed axes, edged bars, legend in a frame, no title."""
    sims = [s for s in PLOT_ORDER if s in set(summary["simulator"])]
    style = {"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "black", "axes.linewidth": 1.0,
             "axes.labelcolor": "black", "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
             "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": "black",
             "ytick.labelcolor": "black", "legend.frameon": True, "legend.edgecolor": "0.8"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 4))
        w = 0.2
        for k, (cond, label) in enumerate(CONDITIONS.items()):
            t = summary[summary["metric"] == f"accuracy, {label}"].set_index("simulator").reindex(sims)
            v, lo, hi = 100 * t["value"], 100 * t["lo"], 100 * t["hi"]
            ax.bar(np.arange(len(sims)) + (k - 1.5) * w, v, w, yerr=[v - lo, hi - v], color=PLOT_COLORS[cond],
                   edgecolor="0.25", linewidth=0.8, label=label.split(" ", 1)[1], zorder=2,
                   error_kw={"ecolor": "black", "elinewidth": 1.2, "capsize": 3})
        ax.axhspan(100 * ceil_lo, 100 * ceil_hi, color="0.88", zorder=0, label="noise ceiling")
        ax.axhline(100 * always1, color="0.4", lw=1, ls="--", zorder=1, label="always CASE 1")
        ax.set_xlim(-0.5, len(sims) - 0.5)
        ax.set_xticks(range(len(sims)), [PLOT_ORDER[s] for s in sims])
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("simulator")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9.5, framealpha=1)
        fig.tight_layout()
        for ext in ["png", "pdf"]:  # 300 dpi PNG and vector PDF for the paper
            fig.savefig(pft.FIG / f"description_test_{parsing}.{ext}", bbox_inches="tight", dpi=300,
                        facecolor="white")
        plt.close(fig)
    print(f"\nWrote {pft.FIG / f'description_test_{parsing}.png'}, .pdf and "
          f"{pft.OUT / f'description_test_summary_{parsing}.csv'}")


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-check", action="store_true", help="no calls: description and swap checks, train half")
    mode.add_argument("--smoke", action="store_true", help="24 calls to check prompts, parsing and cost")
    mode.add_argument("--run", action="store_true", help="all calls not yet saved, then the analysis")
    mode.add_argument("--analyze", action="store_true", help="analysis only, from saved calls")
    ap.add_argument("--parsing", choices=["lenient", "strict"], default="lenient",
                    help="lenient (default): paper's parser, else a reply opening with CASE 1/2; strict: paper's only")
    args = ap.parse_args()
    items = load_items()
    if args.train_check:
        train_check(items, args.parsing)
    elif args.smoke:
        smoke(items)
    elif args.run:
        jobs = make_jobs(items)
        print(f"{len(jobs)} calls to make")
        failed = asyncio.run(run_calls(jobs)) if jobs else 0
        if failed:
            print(f"{failed} calls failed after {pft.MAX_RETRIES} tries and were not saved; rerun --run to retry them")
        analyze(items, args.parsing)
    else:
        analyze(items, args.parsing)
