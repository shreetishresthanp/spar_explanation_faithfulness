# %% [markdown]
# # Induced ground truth: simulator calls (exploratory, extends Exp 1)
#
# Run from the repo root after mm18_induce_prep.py, mm18_induce_write.py and mm18_induce_check_texts.py:
#   python analysis/mm18_induce_run.py --dry-run          # count calls, print one prompt
#   python analysis/mm18_induce_run.py                    # run (resumable, saves every batch)
#   python analysis/mm18_induce_run.py --retry-truncated 8000   # as in Exp 1: re-ask gpt-oss calls cut off at 2,000 tokens
# Same simulators, prompt builder and answer-position note as Exp 1 (mm18_gt_run.build_prompt; the description replaces
# the explanation, as in Exp 1's "gt" condition). Conditions, on each model's held-out pairs:
#   induced_1 … induced_5   free-form drafts (mm18_induce_drafts.json)
#   coef                    coefficient-informed text (mm18_induce_coef_texts.json)
#   own, none               only for the 6 models not in Exp 1; Exp 1's own/none answers are reused for the other 3
# Writes experiments/moral_machines/mm18_induce_predictions.csv (run = "induce"). Exp 1 files are not touched.

# %%
import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
import mm18_gt_run as G  # noqa: E402
from mm18 import load_mm  # noqa: E402
from src.utils import api_key, get_messages, parse_response  # noqa: E402
from src.templates.moral_machines import MoralMachines as MM  # noqa: E402
from src.vllm_configs import VLLM_CONFIGS  # noqa: E402

OUT = Path("experiments/moral_machines")
PATH, TAG = OUT / "mm18_induce_predictions.csv", "induce"
EXP1_MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]


def load_texts(name):
    return {m: v for m, v in json.loads((OUT / name).read_text()).items() if not m.startswith("_")}


def jobs_for(rows, drafts, coef, done):
    jobs = []
    for r in rows.itertuples():
        conds = {f"induced_{x['draft']}": (x["text"], x["factors"]) for x in drafts[r.model]}
        conds["coef"] = (coef[r.model]["text"], coef[r.model]["factors"])
        if r.model not in EXP1_MODELS:
            conds.update(own=None, none=None)
        for cond, tf in conds.items():
            if (r.pair_id, cond) in done:
                continue
            prompt = G.build_prompt(r, cond) if tf is None else G.build_prompt(r, "gt", tf[0], tf[1])
            jobs.append((r, r.pair_id, cond, prompt))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget", type=float, default=8.0, help="stop once estimated simulator spend exceeds this ($)")
    ap.add_argument("--retry-truncated", type=int, metavar="MAX_TOKENS")
    args = ap.parse_args()
    assert api_key, "OPENROUTER_API_KEY missing from .env"
    price = G.prices()

    if args.retry_truncated:  # same procedure as mm18_gt_run.py --retry-truncated
        allp = pd.read_csv(PATH)
        bad = allp[allp["answer"].isna() & (allp["completion_tokens"] >= allp["simulator"].map(lambda s: VLLM_CONFIGS[s].max_tokens))]
        print(f"re-running {len(bad)} truncated calls with max_tokens={args.retry_truncated}")
        for sim, gb in bad.groupby("simulator"):
            cfg = dataclasses.replace(VLLM_CONFIGS[sim], max_tokens=args.retry_truncated)
            outs = asyncio.run(get_messages(prompts=gb["prompt"].tolist(), system_prompt="", config=cfg))
            for i, o in zip(gb.index, outs):
                msg = (o.get("choices") or [{}])[0].get("message", {}) if isinstance(o, dict) else {}
                usage = (o.get("usage") or {}) if isinstance(o, dict) else {}
                content = msg.get("content") or ""
                allp.loc[i, ["answer", "prompt_tokens", "completion_tokens", "content", "reasoning"]] = [
                    (parse_response(content, MM.VALID_ANSWERS) or {}).get("answer"), usage.get("prompt_tokens"),
                    usage.get("completion_tokens"), content, msg.get("reasoning")]
        allp.to_csv(PATH, index=False)
        print(f"recovered {allp.loc[bad.index, 'answer'].notna().sum()} of {len(bad)}; "
              f"total simulator spend ${G.cost(allp, price):.3f}")
        sys.exit(0)

    drafts, coef = load_texts("mm18_induce_drafts.json"), load_texts("mm18_induce_coef_texts.json")
    split = pd.read_csv(OUT / "mm18_induce_split.csv")
    held = split[split["split"] == "heldout"]
    assert set(drafts) == set(coef) == set(held["model"]), "drafts / texts / split cover different models"
    d = load_mm(with_text=True)
    d["pair_id"] = d["original_question_idx"].astype(str) + "-" + d["counterfactual_question_idx"].astype(str)
    rows = d.merge(held[["model", "pair_id"]], on=["model", "pair_id"])
    assert len(rows) == len(held), f"held-out rows found {len(rows)} of {len(held)}"

    for sim in G.SIMULATORS:
        prev = pd.read_csv(PATH) if PATH.exists() else pd.DataFrame(columns=["pair_id", "condition", "simulator"])
        done = set(zip(prev.loc[prev["simulator"] == sim, "pair_id"], prev.loc[prev["simulator"] == sim, "condition"]))
        jobs = jobs_for(rows, drafts, coef, done)
        print(f"{sim}: {len(jobs)} calls to make")
        if args.dry_run:
            print(pd.Series([j[2] for j in jobs]).str.replace(r"_\d", "_k", regex=True).value_counts().to_string())
            print(jobs[0][3][-2500:])
            continue
        for k in range(0, len(jobs), 100):
            chunk = jobs[k:k + 100]
            t0 = time.time()
            outs = asyncio.run(G.call(sim, [j[3] for j in chunk]))
            pd.DataFrame(G._records(TAG, sim, chunk, outs)).to_csv(PATH, mode="a", header=not PATH.exists(), index=False)
            spent = G.cost(pd.read_csv(PATH), price)
            print(f"  calls {k + 1}-{k + len(chunk)} of {len(jobs)} in {time.time() - t0:.0f}s; estimated spend ${spent:.3f}")
            if spent > args.budget:
                print(f"*** BUDGET FLAG: estimated ${spent:.2f} over the ${args.budget:.2f} cap; stopped. Re-run to resume.")
                sys.exit(2)
    if not args.dry_run:
        res = pd.read_csv(PATH)
        print(f"done: {len(res)} rows, parsed {res['answer'].notna().mean():.1%}, estimated spend ${G.cost(res, price):.3f}")
