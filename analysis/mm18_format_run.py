# %% [markdown]
# # Gemma answer-format test: the same scenario asked answer-first and answer-last (moral_machines_gemma_format_plan.md)
#
# Run from the repo root. Paid calls go through OpenRouter (key in .env).
#   python analysis/mm18_format_run.py --smoke     # 10 calls: prompt check, parse rate, cost, top-p/top-k check
#   python analysis/mm18_format_run.py --full      # main set + calibration + Gemini 3 Flash control (~860 calls)
#   python analysis/mm18_format_run.py --qwen      # Qwen3 32B and 14B on the main set, 3 samples per format (1,200 calls)
#   python analysis/mm18_format_run.py --qwen --smoke-qwen   # 2 calls per Qwen model first
# Qwen 3 8B is not served on OpenRouter any more, so the paper-data figure's third Qwen model has no fresh run.
# Prompts come from MoralMachines.create_reference_prompt, the paper's reference prompt; --smoke checks the rebuilt
# prompts against the paper's stored ones first. Output: experiments/moral_machines/mm18_format_predictions.csv.

# %%
import argparse
import asyncio
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from src.templates.moral_machines import MoralMachines as MM  # noqa: E402
from src.utils import api_key, parse_response  # noqa: E402
from src.vllm_configs import VLLM_CONFIGS  # noqa: E402
from mm18_gt_run import BudgetExceeded, key_usage  # noqa: E402

OUT = Path("experiments/moral_machines")
PRED = OUT / "mm18_format_predictions.csv"
GEMMA, CONTROL = "openrouter/google/gemma-3-27b-it", "openrouter/google/gemini-3-flash-preview-medium"
N_MAIN, N_CALIB, N_SAMPLES, SEED, CONC = 100, 20, 3, 0, int(os.environ.get("MM_CONC", 16))
PAPER_SAMPLING = {"top_p": 0.95, "top_k": 64}  # the paper's local vLLM settings for Gemma (temperature 1.0)
# Qwen controls with the paper's reference settings (thinking on, temperature 0.6, top-p 0.95, top-k 20); max_tokens is
# raised so the thinking trace isn't cut off. Sent as-is, not through VLLM_CONFIGS (whose OpenRouter Qwen entries use 0).
QWEN = {"qwen32": "qwen/qwen3-32b", "qwen14": "qwen/qwen3-14b"}
QWEN_PAYLOAD = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8000, "reasoning": {"enabled": True}}
URL = "https://openrouter.ai/api/v1/chat/completions"
HEAD = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
norm = lambda s: re.sub(r"\n\s*\n+", "\n\n", s).strip()


async def one(client, sem, cfg_key, prompt, extra):
    if cfg_key in QWEN.values():
        payload = {"model": cfg_key, "messages": [{"role": "user", "content": prompt}], **QWEN_PAYLOAD}
    else:
        cfg = VLLM_CONFIGS[cfg_key]
        payload = {"model": cfg.model_name, "messages": [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                   "max_tokens": cfg.max_tokens, "temperature": cfg.temperature if cfg.temperature is not None else 1.0, **extra}
        if cfg.enable_reasoning in ("low", "medium", "high"):
            payload["reasoning"] = {"effort": cfg.enable_reasoning}
    async with sem:
        for attempt in range(6):
            try:
                r = await client.post(URL, headers=HEAD, json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt + random.random()); continue
                r.raise_for_status()
                return r.json()
            except (httpx.RequestError, httpx.TimeoutException, ValueError):
                await asyncio.sleep(2 ** attempt + random.random())
    return {}


async def call_all(jobs, extra_by_model):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(timeout=120) as client:
        return await asyncio.gather(*[one(client, sem, j["model_key"], j["prompt"], extra_by_model.get(j["model_key"], {})) for j in jobs])


def records(jobs, outs):
    rows = []
    for j, o in zip(jobs, outs):
        msg = (o.get("choices") or [{}])[0].get("message", {}) if o else {}
        content, usage = msg.get("content") or "", o.get("usage") or {} if o else {}
        p = parse_response(content, MM.VALID_ANSWERS) or {}
        rows.append({"stored_answer": None, **{k: v for k, v in j.items()}, "answer": p.get("answer"), "explanation": p.get("explanation"),
                     "content": content, "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"), "error": "" if o else "no response"})
    return rows


def run_jobs(jobs, extra_by_model, budget, batch=100):
    done = set()
    if PRED.exists():
        prev = pd.read_csv(PRED)
        done = set(prev["job_id"])
    todo = [j for j in jobs if j["job_id"] not in done]
    used0, _ = key_usage()
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        t0 = time.time()
        outs = asyncio.run(call_all(chunk, extra_by_model))
        out = pd.DataFrame(records(chunk, outs))
        if PRED.exists():  # append in the file's own column order
            out = out.reindex(columns=pd.read_csv(PRED, nrows=0).columns)
        out.to_csv(PRED, mode="a", header=not PRED.exists(), index=False)
        spent = key_usage()[0] - used0
        print(f"calls {k + 1}-{k + len(chunk)} of {len(todo)} in {time.time() - t0:.0f}s; key spend this run ${spent:.3f} (cap ${budget})")
        if spent > budget:
            raise BudgetExceeded(f"spend ${spent:.2f} over the ${budget} cap; stopped (re-run to resume)")
    return pd.read_csv(PRED)


# %% Scenario pools
def load():
    cols = ["original_dataset", "original_question", "original_question_idx", "original_question_prompt", "original_answer_first",
            "original_reference_response_answer", "counterfactual_question", "counterfactual_question_idx",
            "original_reference_response_model_info_model"]
    t = pq.read_table("data_parquet/gemma_3_predictions.parquet", columns=cols,
                      filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
    t = t[t["original_reference_response_model_info_model"].str.endswith("gemma-3-27b-it")].drop_duplicates("original_question_idx")
    raw = pd.read_csv("data/raw/moral_machines_raw.csv", index_col=0)
    split = pd.read_csv(OUT / "mm18_gt_split.csv")
    held = split[(split["model"] == "Gemma 3 27B") & (split["split"] == "heldout")]["pair_id"]
    held_idx = sorted({int(x) for p in held for x in p.split("-")})
    pool = raw.loc[held_idx, ["prompt", "scenario_dimension", "is_in_car"]].reset_index().rename(columns={"index": "idx"})
    rng = np.random.default_rng(SEED)
    main = (pool.groupby(["scenario_dimension", "is_in_car"], group_keys=False)
                .apply(lambda g: g.sample(frac=1, random_state=SEED)).reset_index(drop=True))
    # round-robin across strata so the 100 are spread over themes and car/pedestrian layouts
    main["_r"] = main.groupby(["scenario_dimension", "is_in_car"]).cumcount()
    main = main.sort_values(["_r", "scenario_dimension", "is_in_car"]).head(N_MAIN).drop(columns="_r")
    calib = t[~t["original_question_idx"].isin(held_idx)].sample(N_CALIB, random_state=SEED)
    return t, main, calib


def build_jobs(main, calib, n_samples=N_SAMPLES, control=True):
    jobs = []
    for r in main.itertuples():
        for fmt in ("answer_first", "answer_last"):
            prompt = MM.create_reference_prompt(r.prompt, answer_last=(fmt == "answer_last"))
            for s in range(n_samples):
                jobs.append(dict(job_id=f"main|gemma|{r.idx}|{fmt}|{s}", set="main", model_key=GEMMA, idx=int(r.idx),
                                 theme=r.scenario_dimension, in_car=bool(r.is_in_car), fmt=fmt, sample=s, prompt=prompt))
            if control:
                jobs.append(dict(job_id=f"control|flash|{r.idx}|{fmt}|0", set="control", model_key=CONTROL, idx=int(r.idx),
                                 theme=r.scenario_dimension, in_car=bool(r.is_in_car), fmt=fmt, sample=0, prompt=prompt))
    for r in calib.itertuples():
        fmt = "answer_first" if r.original_answer_first else "answer_last"
        for s in range(n_samples):
            jobs.append(dict(job_id=f"calib|gemma|{r.original_question_idx}|{fmt}|{s}", set="calibration", model_key=GEMMA,
                             idx=int(r.original_question_idx), theme="", in_car=None, fmt=fmt, sample=s,
                             prompt=r.original_question_prompt, stored_answer=r.original_reference_response_answer))
    return jobs


def qwen_jobs(main, n_samples=N_SAMPLES):
    return [dict(job_id=f"{tag}|{r.idx}|{fmt}|{s}", set=tag, model_key=model, idx=int(r.idx), theme=r.scenario_dimension,
                 in_car=bool(r.is_in_car), fmt=fmt, sample=s, prompt=MM.create_reference_prompt(r.prompt, answer_last=(fmt == "answer_last")))
            for tag, model in QWEN.items() for r in main.itertuples() for fmt in ("answer_first", "answer_last")
            for s in range(n_samples)]


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--qwen", action="store_true")
    ap.add_argument("--smoke-qwen", action="store_true")
    ap.add_argument("--budget", type=float, default=6.0)
    args = ap.parse_args()
    assert api_key, "OPENROUTER_API_KEY missing from .env"
    stored, main, calib = load()

    # The rebuilt reference prompts must match the paper's stored ones
    same = np.mean([norm(MM.create_reference_prompt(r.original_question, answer_last=not r.original_answer_first))
                    == norm(r.original_question_prompt) for r in stored.itertuples()])
    print(f"rebuilt reference prompts identical to the paper's stored Gemma prompts (up to blank lines): {same:.1%} (n={len(stored)})")
    assert same == 1, "prompt rebuild doesn't match; stopping before any calls"
    print(f"main set: {len(main)} held-out scenarios; themes {main['scenario_dimension'].value_counts().to_dict()}; "
          f"in car {main['is_in_car'].mean():.0%}")

    if args.smoke:
        # top-p/top-k accepted? one probe call each way
        probe = MM.create_reference_prompt(main.iloc[0]["prompt"], answer_last=True)
        r = httpx.post(URL, headers=HEAD, timeout=120, json={"model": VLLM_CONFIGS[GEMMA].model_name, "max_tokens": 50,
                       "messages": [{"role": "user", "content": probe}], "temperature": 1.0, **PAPER_SAMPLING})
        accepted = r.status_code == 200
        print(f"top_p/top_k request: HTTP {r.status_code} -> {'accepted' if accepted else 'rejected, using defaults'}")
        extra = {GEMMA: PAPER_SAMPLING if accepted else {}}
        (OUT / "mm18_format_sampling.txt").write_text("accepted" if accepted else "rejected")
        jobs = [j for j in build_jobs(main.head(2), calib.head(0), n_samples=2, control=False)]  # 8 Gemma calls
        jobs += [j for j in build_jobs(main.head(1), calib.head(0), n_samples=0, control=True)]  # 2 control calls
        jobs = [{**j, "job_id": "smoke|" + j["job_id"], "set": "smoke"} for j in jobs]
        used0, limit = key_usage()
        res = run_jobs(jobs, extra, args.budget)
        time.sleep(5)
        used1, _ = key_usage()
        s = res[res["set"] == "smoke"]
        print(f"\nparsed {s['answer'].notna().sum()} of {len(s)}; answers: {s[['model_key', 'idx', 'fmt', 'sample', 'answer']].to_string(index=False)}")
        print(s.groupby("model_key")[["prompt_tokens", "completion_tokens"]].mean().round(0).to_string())
        print(f"\nkey usage before ${used0:.4f}, after ${used1:.4f} (limit {limit}): this smoke test ${used1 - used0:.4f}")
        a, b = (MM.create_reference_prompt(main.iloc[0]["prompt"], answer_last=x) for x in (False, True))
        print("\n=== EXAMPLE PAIR: answer first ===\n" + a + "\n\n=== answer last ===\n" + b)

    if args.full:
        probe_ok = Path(OUT / "mm18_format_sampling.txt")
        extra = {GEMMA: PAPER_SAMPLING if probe_ok.exists() and probe_ok.read_text().strip() == "accepted" else {}}
        try:
            res = run_jobs(build_jobs(main, calib), extra, args.budget)
        except BudgetExceeded as e:
            print(f"\n*** BUDGET FLAG: {e}"); sys.exit(2)
        f = res[res["set"] != "smoke"]
        print(f"done: {len(f)} rows, parsed {f['answer'].notna().mean():.1%}")

    if args.qwen:
        jobs = qwen_jobs(main)
        if args.smoke_qwen:
            jobs = [{**j, "job_id": "smoke|" + j["job_id"], "set": "smoke"} for j in qwen_jobs(main.head(1), n_samples=1)]
        try:
            res = run_jobs(jobs, {}, args.budget)
        except BudgetExceeded as e:
            print(f"\n*** BUDGET FLAG: {e}"); sys.exit(2)
        q = res[res["job_id"].isin({j["job_id"] for j in jobs})]
        print(f"qwen rows {len(q)}, parsed {q['answer'].notna().mean():.1%}")
        print(q.groupby("model_key")[["prompt_tokens", "completion_tokens"]].agg(["mean", "max"]).round(0).to_string())
        if args.smoke_qwen:
            print(q[["model_key", "fmt", "answer", "completion_tokens"]].to_string(index=False))
            print("\n--- sample content ---\n" + str(q.iloc[0]["content"])[:800])
