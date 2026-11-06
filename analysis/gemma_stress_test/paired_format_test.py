# %% [markdown]
# # Paired answer-position test: Gemma 3 27B on the same Moral Machines scenarios, answer first and answer last
#
# Run from the repo root in the project env (the prompt template needs it):
#   conda activate faithfulness-env
#   python analysis/gemma_stress_test/paired_format_test.py --smoke     # 4 calls: provider, parsing, cost per call
#   python analysis/gemma_stress_test/paired_format_test.py --run       # all calls (resumes; skips calls already saved)
#   python analysis/gemma_stress_test/paired_format_test.py --analyze   # analysis only, from the saved calls
# Paid calls go through OpenRouter (OPENROUTER_API_KEY in .env).
#
# Scenarios: every moral_machines row of data_parquet/gemma_3_predictions.parquet answered by google/gemma-3-27b-it,
# both its original scenario and its counterfactual scenario (1,000 + 1,000), each asked in both formats, SAMPLES
# samples per format. In the stored data each scenario was asked in one format only. The prompt for that format is
# the stored one (original_question_prompt / counterfactual_question_prompt) sent unchanged; the other format is built
# with MoralMachines.create_reference_prompt, the function that built the stored prompts (checked byte-for-byte on
# every row at startup). Predictor prompts are not rerun: their format is fixed, so they do not test answer position.
#
# Settings match the stored run (local vLLM, bf16): temperature 1.0, top_p 0.95, top_k 64, max_tokens 2000, a single
# user message with no system prompt. Served by Novita (the only bf16 provider), no fallback, require_parameters so
# top_k and seed are honoured. Seed = sample index (0, 1, 2), so a rerun with the same provider should reproduce;
# OpenRouter does not guarantee determinism, so the analysis also reports how often repeat samples disagree.
# Answers are parsed with src.utils.parse_response, as in the paper's pipeline. Gemma sometimes drops the [ANSWER]
# tag and opens with "CASE 1"; the default --parsing lenient also accepts a reply that opens with CASE 1 / CASE 2
# (raw replies are saved, so --parsing strict reruns the analysis with the paper's parser only).
#
# Success, decided before running, on the original scenarios:
#   calibration: answer-last CASE 1 rate > 85% (stored data: 96%);
#   replication: paired gap, mean over scenarios of P(CASE 1 | answer first) - P(CASE 1 | answer last), <= -10 points
#                with the 95% bootstrap interval excluding 0 (stored data, unpaired: -36).
# The counterfactual scenarios are reported the same way as an independent replication set.
# Writes experiments/moral_machines/gemma_stress_test/paired_format_{calls.jsonl,summary_<parsing>.csv} and
# figures/moral_machines/gemma_stress_test/paired_format_test_<parsing>.png.

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
from scipy import stats  # noqa: E402

if Path.cwd().name in {"analysis", "gemma_stress_test"}:
    os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, ".")
from src.templates.moral_machines import MoralMachines  # noqa: E402
from src.utils import parse_response  # noqa: E402

DATA = Path("data_parquet/gemma_3_predictions.parquet")
TARGET = "google/gemma-3-27b-it"
OUT = Path("experiments/moral_machines/gemma_stress_test")
FIG = Path("figures/moral_machines/gemma_stress_test")
CALLS = OUT / "paired_format_calls.jsonl"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SAMPLES, FORMATS, SETS = 3, ["first", "last"], ["original", "counterfactual"]
SETTINGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 2000}
PROVIDER = {"order": ["novita"], "quantizations": ["bf16"], "allow_fallbacks": False, "require_parameters": True}
URL = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY, MAX_RETRIES, TIMEOUT = 16, 6, 120
CALIBRATION_MIN, GAP_MAX, N_BOOT, SEED = 0.85, -0.10, 2000, 0

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FORMAT_COLORS = {"first": "#2a9d8f", "last": "#8a8984"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False})


# %% scenarios and prompts
def load_scenarios():
    """One row per (set, scenario): stored format, stored prompt and answer, and the prompt for each format."""
    cols = ["original_dataset", "original_reference_response_model_info_model", "original_answer_first"]
    for s in SETS:
        cols += [f"{s}_question", f"{s}_question_idx", f"{s}_question_prompt", f"{s}_reference_response_answer"]
    t = pq.read_table(DATA, columns=cols, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
    t = t[t["original_reference_response_model_info_model"] == TARGET]
    rows = []
    for s in SETS:
        part = pd.DataFrame({"set": s, "idx": t[f"{s}_question_idx"].values,
                             "stored_format": np.where(t["original_answer_first"], "first", "last"),
                             "question": t[f"{s}_question"].values, "stored_prompt": t[f"{s}_question_prompt"].values,
                             "stored_answer": t[f"{s}_reference_response_answer"].values})
        rows.append(part.drop_duplicates(["set", "idx"]))
    d = pd.concat(rows, ignore_index=True)
    for f in FORMATS:
        d[f"prompt_{f}"] = [MoralMachines.create_reference_prompt(q, answer_last=(f == "last")) for q in d["question"]]
    rebuilt = d.apply(lambda r: r[f"prompt_{r['stored_format']}"], axis=1)
    mismatch = (rebuilt != d["stored_prompt"]).sum()
    assert mismatch == 0, f"template does not reproduce {mismatch} stored prompts"
    for f in FORMATS:  # send the stored prompt itself for the stored format
        d.loc[d["stored_format"] == f, f"prompt_{f}"] = d.loc[d["stored_format"] == f, "stored_prompt"]
    print(f"{len(d)} scenarios ({d.groupby('set').size().to_dict()}); template reproduces all stored prompts")
    return d


def call_key(s, idx, fmt, sample):
    return f"{s}|{idx}|{fmt}|{sample}"


def load_calls(path=CALLS):
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


# %% API calls
async def one_call(client, sem, job, api_key, lock, path=CALLS):
    """One API call; the record keeps every job field except the prompt, and is appended to `path` on success."""
    payload = {"model": TARGET, "messages": [{"role": "user", "content": job["prompt"]}], **SETTINGS,
               "seed": job["sample"], "provider": PROVIDER, "usage": {"include": True}}
    rec = {k: v for k, v in job.items() if k != "prompt"}
    rec.update(seed=job["sample"])
    rec.setdefault("prompt_source", "stored" if job["format"] == job["stored_format"] else "template")
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                r = await client.post(URL, json=payload, headers={"Authorization": f"Bearer {api_key}"})
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
                           completion_tokens=usage.get("completion_tokens"), cost=usage.get("cost"), error=None)
                break
            except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
                rec.update(error=str(e)[:300])
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
    if rec.get("error") is None:
        async with lock:
            with path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
    return rec


async def run_calls(jobs, path=CALLS):
    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    done, failed = 0, 0
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [asyncio.create_task(one_call(client, sem, j, api_key, lock, path)) for j in jobs]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            done += 1
            failed += rec.get("error") is not None
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} calls, {failed} failed")
    return failed


def make_jobs(d, samples):
    have = set(load_calls().get("key", pd.Series(dtype=str)))
    jobs = []
    for r in d.itertuples():
        for f in FORMATS:
            for sample in samples:
                key = call_key(r.set, r.idx, f, sample)
                if key not in have:
                    jobs.append({"key": key, "set": r.set, "idx": int(r.idx), "stored_format": r.stored_format,
                                 "format": f, "sample": sample, "prompt": getattr(r, f"prompt_{f}")})
    return jobs


def smoke(d):
    """2 original scenarios (one stored in each format) x 2 formats x sample 0; prints cost per call and projection."""
    pick = pd.concat([d[(d["set"] == "original") & (d["stored_format"] == f)].head(1) for f in FORMATS])
    jobs = make_jobs(pick, [0])
    asyncio.run(run_calls(jobs))
    c = load_calls()
    c = c[c["key"].isin([j["key"] for j in jobs])]
    print(c[["key", "prompt_source", "provider", "answer", "finish_reason", "prompt_tokens", "completion_tokens",
             "cost"]].to_string(index=False))
    total = len(d) * len(FORMATS) * SAMPLES
    if c["cost"].notna().any():
        print(f"\nmean cost per call ${c['cost'].mean():.6f} -> projected ${c['cost'].mean() * total:.2f} "
              f"for {total} calls")
    print("\nfirst raw reply:\n", c["raw"].iloc[0][:600] if len(c) else "(none)")


# %% analysis
def boot_mean(x, rng):
    reps = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)])
    return x.mean(), *np.percentile(reps, [2.5, 97.5])


def lenient(raw, strict):
    """The paper's parse, else a reply that opens with CASE 1 / CASE 2 before any tag (Gemma sometimes drops [ANSWER])."""
    if strict is not None or not isinstance(raw, str):
        return strict
    m = re.match(r"\W*(CASE [12])\b", raw.strip().upper())
    return m.group(1) if m else None


def analyze(d, parsing="lenient"):
    c = load_calls()
    if c.empty:
        print("no saved calls yet")
        return
    c["answer_strict"] = c["answer"]
    c["answer"] = [lenient(r, a) for r, a in zip(c["raw"], c["answer_strict"])] if parsing == "lenient" else c["answer"]
    c["case1"] = np.where(c["answer"].isna(), np.nan, (c["answer"] == "CASE 1").astype(float))
    print(f"\n{len(c)} calls saved of {len(d) * len(FORMATS) * SAMPLES}; providers {c['provider'].value_counts().to_dict()}")
    print(f"analysis uses {parsing} parsing; parse rate by set and format (strict = paper's parser):")
    print(c.groupby(["set", "format"]).agg(strict=("answer_strict", lambda a: a.notna().mean()),
                                           lenient=("answer", lambda a: a.notna().mean())).round(3).to_string())
    print("finish_reason counts:", c["finish_reason"].value_counts().to_dict())

    rng = np.random.default_rng(SEED)
    rows = []
    per = (c.groupby(["set", "idx", "stored_format", "format"])["case1"].mean().unstack("format")
           .dropna(subset=FORMATS).reset_index())
    per["gap"] = per["first"] - per["last"]
    for s in SETS + ["both"]:
        cs, ps = (c, per) if s == "both" else (c[c["set"] == s], per[per["set"] == s])
        if ps.empty:
            continue
        for f in FORMATS:
            est, lo, hi = boot_mean(ps[f].values, rng)
            rows.append({"set": s, "metric": f"CASE 1 rate, answer {f}", "value": est, "lo": lo, "hi": hi, "n": len(ps)})
        est, lo, hi = boot_mean(ps["gap"].values, rng)
        rows.append({"set": s, "metric": "paired gap (first - last)", "value": est, "lo": lo, "hi": hi, "n": len(ps)})
        for sf in FORMATS:  # does the effect depend on which format the stored run used?
            g = ps[ps["stored_format"] == sf]["gap"].values
            if len(g):
                est, lo, hi = boot_mean(g, rng)
                rows.append({"set": s, "metric": f"paired gap, stored answer {sf}", "value": est, "lo": lo, "hi": hi,
                             "n": len(g)})
        # majority vote per scenario and format; exact McNemar on scenarios whose vote differs between formats
        vote = (cs.groupby(["idx", "set", "format"])["case1"].mean().unstack("format").dropna() > 0.5)
        b, cc = int((vote["last"] & ~vote["first"]).sum()), int((~vote["last"] & vote["first"]).sum())
        p = stats.binomtest(b, b + cc).pvalue if b + cc else np.nan
        rows.append({"set": s, "metric": "McNemar p (majority vote)", "value": p, "n": b + cc,
                     "note": f"CASE 1 last only: {b}, first only: {cc}"})
        # repeat-sample consistency within a format
        n_ans = cs.dropna(subset=["case1"]).groupby(["set", "idx", "format"])["case1"]
        split = (n_ans.nunique() > 1)[n_ans.size() > 1]
        rows.append({"set": s, "metric": "scenario-formats whose repeat samples disagree", "value": split.mean(),
                     "n": len(split)})
        # calibration: rerun in the stored format vs the stored answer
        st = cs[cs["format"] == cs["stored_format"]].merge(d[["set", "idx", "stored_answer"]], on=["set", "idx"])
        st = st.dropna(subset=["answer", "stored_answer"])
        rows.append({"set": s, "metric": "agreement with stored answer (stored format)",
                     "value": (st["answer"] == st["stored_answer"]).mean(), "n": len(st)})
        sd = d[d["set"] == s] if s != "both" else d
        for f in FORMATS:
            stored = sd[sd["stored_format"] == f]["stored_answer"].dropna()
            rows.append({"set": s, "metric": f"stored CASE 1 rate, answer {f} (unpaired)",
                         "value": (stored == "CASE 1").mean(), "n": len(stored)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / f"paired_format_summary_{parsing}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", summary.assign(**{k: summary[k].round(3) for k in ["value", "lo", "hi"] if k in summary}).to_string(
        index=False))

    o = summary[summary["set"] == "original"].set_index("metric")
    if "CASE 1 rate, answer last" in o.index:
        cal = o.at["CASE 1 rate, answer last", "value"]
        gap, hi = o.at["paired gap (first - last)", "value"], o.at["paired gap (first - last)", "hi"]
        complete = len(per[per["set"] == "original"]) == (d["set"] == "original").sum()
        print(f"\nSUCCESS CRITERIA (original scenarios{'' if complete else ', INCOMPLETE RUN'}):")
        print(f"  calibration  answer-last CASE 1 rate {cal:.1%} > {CALIBRATION_MIN:.0%}: "
              f"{'PASS' if cal > CALIBRATION_MIN else 'FAIL'}")
        print(f"  replication  paired gap {gap * 100:+.1f} pts (95% CI upper {hi * 100:+.1f}) <= {GAP_MAX * 100:+.0f} "
              f"and CI excludes 0: {'PASS' if gap <= GAP_MAX and hi < 0 else 'FAIL'}")
    plot(summary, per, parsing)


def plot(summary, per, parsing):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    sets = [s for s in SETS if s in set(summary["set"])]
    w = 0.38
    for k, f in enumerate(FORMATS):
        t = summary[summary["metric"] == f"CASE 1 rate, answer {f}"].set_index("set").reindex(sets)
        xs = np.arange(len(sets)) + (k - 0.5) * w
        bars = ax.bar(xs, t["value"], w, yerr=[t["value"] - t["lo"], t["hi"] - t["value"]], color=FORMAT_COLORS[f],
                      label=f"answer {f}", error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in t["value"]], padding=8, fontsize=9, color=MUTED)
    ax.axhline(CALIBRATION_MIN, color=INK, lw=0.8, ls="--")
    ax.text(len(sets) - 0.5, CALIBRATION_MIN + 0.01, f"calibration bar {CALIBRATION_MIN:.0%}", ha="right", fontsize=8,
            color=MUTED)
    ax.set_xticks(range(len(sets)), [f"{s} scenarios" for s in sets])
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("P(CASE 1), same scenarios")
    ax.set_title("CASE 1 rate by answer position")
    ax.legend(loc="upper left", ncol=2)

    step = 1 / SAMPLES  # with SAMPLES answers per format, per-scenario gaps are multiples of 1/SAMPLES
    bins = np.arange(-1 - step / 2, 1 + step, step)
    for s, color in zip(sets, ["#2a78d6", "#eb6834"]):
        bx.hist(per[per["set"] == s]["gap"], bins=bins, alpha=0.6, color=color, label=f"{s} scenarios")
    bx.axvline(0, color=INK, lw=0.8)
    bx.set_xlabel("Per-scenario gap: P(CASE 1 | first) - P(CASE 1 | last)")
    bx.set_ylabel("Scenarios")
    bx.set_title("Per-scenario paired gap")
    bx.legend()
    fig.suptitle(f"Paired answer-position test: Gemma 3 27B, Moral Machines ({parsing} parsing)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"paired_format_test_{parsing}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG / f'paired_format_test_{parsing}.png'} and {OUT / f'paired_format_summary_{parsing}.csv'}")


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="4 calls to check provider, parsing and cost")
    mode.add_argument("--run", action="store_true", help="all calls not yet saved, then the analysis")
    mode.add_argument("--analyze", action="store_true", help="analysis only, from saved calls")
    ap.add_argument("--parsing", choices=["lenient", "strict"], default="lenient",
                    help="lenient (default): paper's parser, else a reply opening with CASE 1/2; strict: paper's only")
    args = ap.parse_args()
    scenarios = load_scenarios()
    if args.smoke:
        smoke(scenarios)
    elif args.run:
        jobs = make_jobs(scenarios, range(SAMPLES))
        print(f"{len(jobs)} calls to make")
        failed = asyncio.run(run_calls(jobs)) if jobs else 0
        if failed:
            print(f"{failed} calls failed after {MAX_RETRIES} tries and were not saved; rerun --run to retry them")
        analyze(scenarios, args.parsing)
    else:
        analyze(scenarios, args.parsing)
