# %% [markdown]
# # Run simulator calls for the ground-truth and re-paired experiments (moral_machines_experiments_README.md)
#
# Run from the repo root. Paid calls go through OpenRouter (key in .env).
#   python analysis/mm18_gt_run.py --smoke          # 10 calls: 5 GPT-5.2 pairs x "own" x 2 simulators, prints cost
#   python analysis/mm18_gt_run.py --exp 1 ...      # (after Gate 1a) ground-truth experiment
# Prompts are built with the MoralMachines template, so the "own" and "none" conditions are the paper's prompts;
# --smoke first checks that the rebuilt prompts match the stored ones character for character.

# %%
import argparse
import asyncio
import os
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
from src.utils import api_key, get_messages, parse_response  # noqa: E402
from src.vllm_configs import VLLM_CONFIGS  # noqa: E402
from mm18 import FILES, PAIR_TYPES, load_mm  # noqa: E402

OUT = Path("experiments/moral_machines")
SIMULATORS = ["openrouter/openai/gpt-oss-20b", "openrouter/google/gemini-3-flash-preview"]
CF_HEADER = "--- COUNTERFACTUAL SCENARIO ---\n"
GT_HEADER = "How this evaluator decides across many scenarios:"


def position_note(answer_first):
    return (f"(Note: for this scenario the evaluator was asked to state its answer "
            f"{'before' if answer_first else 'after'} its reasoning.)")


def build_prompt(r, condition, gt_text=None, gt_factors=None, note=True, tree_text=None, tree_factors=None):
    """r: a load_mm row (with text). condition: own | none | gt | tree | gt+own."""
    expl = {k: list(v) if isinstance(v, np.ndarray) else v  # parquet stores lists as arrays
            for k, v in dict(r.original_reference_response_parsed_response or {}).items()}
    if condition == "none":
        p = MM.create_counterfactual_prompt_no_explanation(r.original_question, expl, r.counterfactual_question)
    else:
        if condition == "gt":
            expl = {**expl, "explanation": gt_text, "most_important_factors": gt_factors}
        elif condition == "tree":
            expl = {**expl, "explanation": tree_text, "most_important_factors": tree_factors}
        elif condition == "gt+own":
            expl = {**expl, "explanation": f"{expl.get('explanation', '')}\n\n{GT_HEADER}\n{gt_text}"}
        p = MM.create_counterfactual_prompt(r.original_question, expl, r.counterfactual_question)
    if note:  # one line under the counterfactual scenario, identical across conditions
        p = p.replace(r.counterfactual_question, r.counterfactual_question.rstrip("\n") + "\n\n"
                      + position_note(r.original_answer_first) + "\n", 1) if p.count(r.counterfactual_question) == 1 else None
        assert p is not None, "counterfactual text not found exactly once in the prompt"
    return p


def stored_prompts():
    cols = ["original_dataset", "original_question", "counterfactual_question", "original_reference_response_model_info_model",
            "original_reference_response_model_info_thinking", "counterfactual_prompt_with_explanation",
            "counterfactual_prompt_without_explanation"]
    t = pd.concat([pq.read_table(f"data_parquet/{f}.parquet", columns=cols, filters=[("original_dataset", "=", "moral_machines")])
                   .to_pandas() for fs in FILES.values() for f in fs], ignore_index=True)
    return t.drop_duplicates(cols[1:5])


async def call(sim, prompts):
    cfg = VLLM_CONFIGS[sim]
    return await get_messages(prompts=prompts, system_prompt="", config=cfg)


def key_usage():
    """Credits used so far on this key (OpenRouter /key endpoint); exact cost = difference before/after a run."""
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    d = r.json()["data"]
    return float(d.get("usage") or 0), d.get("limit")


def prices():
    """$/token from OpenRouter's public model list, keyed by our config name."""
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    by_id = {m["id"]: m["pricing"] for m in data}
    return {sim: (float(by_id[VLLM_CONFIGS[sim].model_name]["prompt"]), float(by_id[VLLM_CONFIGS[sim].model_name]["completion"]))
            for sim in SIMULATORS}


def cost(df, price):
    """Estimated $ for the rows of a predictions table, from their token counts."""
    if df.empty:
        return 0.0
    pin = df["simulator"].map(lambda s: price[s][0])
    pout = df["simulator"].map(lambda s: price[s][1])
    return float((df["prompt_tokens"].fillna(0) * pin + df["completion_tokens"].fillna(0) * pout).sum())


class BudgetExceeded(RuntimeError):
    pass


def run(rows, conditions, sims, out_path, tag, gt=None, budget=None, price=None, batch=100):
    """rows: DataFrame of load_mm rows. Appends one CSV row per (pair, condition, simulator); skips ones already done."""
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["run"], prev["pair_id"], prev["condition"], prev["simulator"]))
    recs = []
    for sim in sims:
        jobs = []
        for r in rows.itertuples():
            pid = f"{r.original_question_idx}-{r.counterfactual_question_idx}"
            for cond in conditions:
                if (tag, pid, cond, sim) in done:
                    continue
                g = (gt or {}).get(r.model, {})
                jobs.append((r, pid, cond, build_prompt(r, cond, g.get("text"), g.get("factors"),
                                                        tree_text=g.get("tree_text"), tree_factors=g.get("tree_factors"))))
        for k in range(0, len(jobs), batch):
            chunk = jobs[k:k + batch]
            t0 = time.time()
            outs = asyncio.run(call(sim, [j[3] for j in chunk]))
            print(f"{sim}: calls {k + 1}-{k + len(chunk)} of {len(jobs)} in {time.time() - t0:.0f}s")
            recs = _records(tag, sim, chunk, outs)
            pd.DataFrame(recs).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)  # save every batch
            if budget is not None:
                spent = cost(pd.read_csv(out_path).query("run == @tag"), price)
                print(f"  estimated spend so far for '{tag}': ${spent:.3f} (cap ${budget:.2f})")
                if spent > budget:
                    raise BudgetExceeded(f"'{tag}' estimated spend ${spent:.2f} is over the ${budget:.2f} cap; stopped. "
                                         "Re-run to resume after raising --budget.")
    return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()


def _records(tag, sim, jobs, outs):
    recs = []
    for (r, pid, cond, prompt), o in zip(jobs, outs):
        msg = (o.get("choices") or [{}])[0].get("message", {}) if isinstance(o, dict) else {}
        content, usage = msg.get("content") or "", (o.get("usage") or {}) if isinstance(o, dict) else {}
        parsed = parse_response(content, MM.VALID_ANSWERS) or {}
        recs.append(dict(run=tag, model=r.model, pair_id=pid, original_idx=r.original_question_idx,
                         counterfactual_idx=r.counterfactual_question_idx, what_changed=r.what_changed,
                         condition=cond, simulator=sim, answer=parsed.get("answer"),
                         reference_answer=r.counterfactual_reference_response_answer,
                         prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                         prompt=prompt, content=content, reasoning=msg.get("reasoning")))
    return recs


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp", type=int, choices=[1, 2])
    ap.add_argument("--reps", type=int, default=2, help="experiment 2: independent repeats per call")
    ap.add_argument("--budget", type=float, default=6.0, help="stop once estimated spend for the run exceeds this ($)")
    ap.add_argument("--dry-run", action="store_true", help="build and count prompts, make no calls")
    ap.add_argument("--max-tokens", type=int, help="override every simulator's max_tokens (e.g. 8000 so gpt-oss isn't truncated)")
    ap.add_argument("--retry-truncated", type=int, metavar="MAX_TOKENS",
                    help="re-run calls that hit the token cap without an answer, with this higher cap, and replace them")
    args = ap.parse_args()
    assert api_key, "OPENROUTER_API_KEY missing from .env"
    if args.max_tokens:
        import dataclasses
        for sim in SIMULATORS:
            VLLM_CONFIGS[sim] = dataclasses.replace(VLLM_CONFIGS[sim], max_tokens=max(args.max_tokens, VLLM_CONFIGS[sim].max_tokens))
    d = load_mm(with_text=True)

    if args.smoke:
        # 1. The rebuilt "own"/"none" prompts (without the note) must equal the paper's stored prompts
        st = stored_prompts()
        g = d[d["model"] == "GPT-5.2"].merge(st, on=["original_question", "counterfactual_question",
                                                    "original_reference_response_model_info_model",
                                                    "original_reference_response_model_info_thinking"])
        norm = lambda s: re.sub(r"\n\s*\n+", "\n\n", s).strip()  # stored prompts differ only in blank lines
        same_w = np.mean([norm(build_prompt(r, "own", note=False)) == norm(r.counterfactual_prompt_with_explanation) for r in g.itertuples()])
        same_wo = np.mean([norm(build_prompt(r, "none", note=False)) == norm(r.counterfactual_prompt_without_explanation) for r in g.itertuples()])
        print(f"rebuilt prompts identical to stored (up to blank lines): with explanation {same_w:.1%}, without {same_wo:.1%} (n={len(g)})")
        assert same_w == 1 and same_wo == 1, "prompt rebuild doesn't match the paper's prompts; stopping before any calls"
        # 2. Five GPT-5.2 pairs, one per pair type, "own" condition with the position note, both simulators
        pick = pd.concat([g[g["what_changed"] == pt].sample(1, random_state=0) for pt in PAIR_TYPES])
        used0, limit = key_usage()
        res = run(pick, ["own"], SIMULATORS, OUT / "mm18_gt_smoke.csv", "smoke")
        time.sleep(5)  # usage on the key updates with a short delay
        used1, _ = key_usage()
        res = res[res["run"] == "smoke"]
        print(f"\nparsed {res['answer'].notna().sum()} of {len(res)} answers; "
              f"right {(res['answer'] == res['reference_answer']).sum()} of {len(res)}")
        print(res.groupby("simulator")[["prompt_tokens", "completion_tokens"]].mean().round(0).to_string())
        print(f"\ncost of this run: ${used1 - used0:.4f} (key usage ${used1:.4f}, limit {limit})")
        print("\nexample prompt:\n" + res["prompt"].iloc[0])

    if args.exp == 1:
        import json
        gt = {m: v for m, v in json.loads((OUT / "mm18_gt_explanations.json").read_text()).items() if not m.startswith("_")}
        split = pd.read_csv(OUT / "mm18_gt_split.csv")
        held = split[split["split"] == "heldout"]
        d["pair_id"] = d["original_question_idx"].astype(str) + "-" + d["counterfactual_question_idx"].astype(str)
        rows = d.merge(held[["model", "pair_id"]], on=["model", "pair_id"])
        assert len(rows) == len(held), f"held-out rows found {len(rows)} of {len(held)}"
        conds = ["own", "gt", "tree", "gt+own", "none"]
        print(f"experiment 1: {len(rows)} held-out pairs x {len(conds)} conditions x {len(SIMULATORS)} simulators "
              f"= {len(rows) * len(conds) * len(SIMULATORS)} calls; cap ${args.budget:.2f}")
        if args.dry_run:
            r0 = rows.iloc[0]
            g0 = gt[r0.model]
            for c in conds:
                pr = build_prompt(r0, c, g0["text"], g0["factors"], tree_text=g0["tree_text"], tree_factors=g0["tree_factors"])
                print(f"--- {c} prompt for {r0.model} pair {r0.pair_id}: {len(pr)} chars")
            print(build_prompt(r0, "gt", gt[r0.model]["text"], gt[r0.model]["factors"]))
        else:
            try:
                run(rows, conds, SIMULATORS, OUT / "mm18_gt_predictions.csv", "exp1", gt=gt, budget=args.budget, price=prices())
            except BudgetExceeded as e:
                print(f"\n*** BUDGET FLAG: {e}")
                sys.exit(2)
            res = pd.read_csv(OUT / "mm18_gt_predictions.csv").query("run == 'exp1'")
            print(f"done: {len(res)} rows, parsed {res['answer'].notna().mean():.1%}, estimated spend ${cost(res, prices()):.3f}")

    if args.retry_truncated:
        import dataclasses
        path = OUT / "mm18_gt_predictions.csv"
        allp = pd.read_csv(path)
        bad = allp[allp["answer"].isna() & (allp["completion_tokens"] >= allp["simulator"].map(lambda s: VLLM_CONFIGS[s].max_tokens))]
        print(f"re-running {len(bad)} truncated calls with max_tokens={args.retry_truncated}")
        price, fixed = prices(), []
        for sim, gb in bad.groupby("simulator"):
            cfg = dataclasses.replace(VLLM_CONFIGS[sim], max_tokens=args.retry_truncated)
            outs = asyncio.run(get_messages(prompts=gb["prompt"].tolist(), system_prompt="", config=cfg))
            for (_, r), o in zip(gb.iterrows(), outs):
                msg = (o.get("choices") or [{}])[0].get("message", {}) if isinstance(o, dict) else {}
                usage = (o.get("usage") or {}) if isinstance(o, dict) else {}
                content = msg.get("content") or ""
                r = r.copy()
                r["answer"] = (parse_response(content, MM.VALID_ANSWERS) or {}).get("answer")
                r["prompt_tokens"], r["completion_tokens"] = usage.get("prompt_tokens"), usage.get("completion_tokens")
                r["content"], r["reasoning"] = content, msg.get("reasoning")
                fixed.append(r)
        fixed = pd.DataFrame(fixed)
        extra = cost(fixed, price)
        allp = pd.concat([allp.drop(index=bad.index), fixed]).sort_index()
        allp.to_csv(path, index=False)
        total = cost(allp[allp["run"] == "exp1"], price)
        print(f"recovered {fixed['answer'].notna().sum()} of {len(fixed)}; extra spend ${extra:.3f}; total exp1 spend ${total:.3f}")

    if args.exp == 2:
        import json
        gt = {m: v for m, v in json.loads((OUT / "mm18_gt_explanations.json").read_text()).items() if not m.startswith("_")}
        raw_prompt = pd.read_csv("data/raw/moral_machines_raw.csv", index_col=0)["prompt"]
        resp_of = {}
        for idx_col, resp_col in (("original_question_idx", "original_reference_response_parsed_response"),
                                  ("counterfactual_question_idx", "counterfactual_reference_response_parsed_response")):
            for m, i, r in zip(d["model"], d[idx_col], d[resp_col]):
                resp_of[(m, i)] = r

        def as_rows(sample):
            """Re-paired candidates -> rows build_prompt understands (original scenario's own explanation is shown)."""
            return pd.DataFrame({
                "model": sample["model"], "original_question_idx": sample["original_idx"],
                "counterfactual_question_idx": sample["counterfactual_idx"],
                "original_question": sample["original_idx"].map(raw_prompt),
                "counterfactual_question": sample["counterfactual_idx"].map(raw_prompt),
                "original_reference_response_parsed_response": [resp_of[(m, i)] for m, i in zip(sample["model"], sample["original_idx"])],
                "original_answer_first": sample["answer_first"],
                "counterfactual_reference_response_answer": sample["answer_counterfactual"],
                "what_changed": sample["primary_factor"] + " · " + sample["tie_label"] + " · " + sample["cited_label"]})

        main = as_rows(pd.read_csv(OUT / "mm18_repair_sample.csv"))
        gt_arm = as_rows(pd.read_csv(OUT / "mm18_repair_gt_sample.csv"))
        assert (main["original_reference_response_parsed_response"].map(lambda r: r["answer"]).values
                == pd.read_csv(OUT / "mm18_repair_sample.csv")["answer_original"].values).all(), "answer mismatch"
        n_calls = args.reps * len(SIMULATORS) * (2 * len(main) + 3 * len(gt_arm))
        print(f"experiment 2: main {len(main)} pairs x 2 conditions + Gemma ground-truth arm {len(gt_arm)} pairs x 3 "
              f"conditions, x {len(SIMULATORS)} simulators x {args.reps} repeats = {n_calls} calls; cap ${args.budget:.2f}")
        if args.dry_run:
            print(build_prompt(gt_arm.iloc[0], "gt", gt["Gemma 3 27B"]["text"], gt["Gemma 3 27B"]["factors"])[-1500:])
        else:
            path, price = OUT / "mm18_repair_predictions.csv", prices()
            try:
                for k in range(1, args.reps + 1):
                    run(main, ["own", "none"], SIMULATORS, path, f"exp2_r{k}", budget=args.budget, price=price)
                    run(gt_arm, ["own", "none", "gt"], SIMULATORS, path, f"exp2gt_r{k}", gt=gt, budget=args.budget, price=price)
            except BudgetExceeded as e:
                print(f"\n*** BUDGET FLAG: {e}")
                sys.exit(2)
            res = pd.read_csv(path)
            print(f"done: {len(res)} rows, parsed {res['answer'].notna().mean():.1%}, estimated spend ${cost(res, price):.3f}")
