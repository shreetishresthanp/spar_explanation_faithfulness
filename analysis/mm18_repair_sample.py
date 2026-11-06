# %% [markdown]
# # Experiment 2 sample: re-paired counterfactuals, near-tie vs clear (headline), cited as a label only
#
# Run from the repo root after `python analysis/mm18_repair.py` (which builds the candidate file):
# `python analysis/mm18_repair_sample.py`. No API calls. Design (decided 2026-10-27):
# - per model, 50 near-tie + 50 clear pairs, single-rule-feature changes only
# - traffic-light (law) changes capped at half of each group; each original and each counterfactual scenario used once
# - half flips / half non-flips where possible; Gemma also balanced on answer format (25 answer-first per group)
# - Gemma ground-truth arm: separate sample whose two scenarios both come from experiment 1's HELD-OUT pairs, so the
#   ground-truth text (written from training pairs) is never tested on scenarios it was written from
# - 20 "cited" pairs drawn for the Gate 2a hand check
# Writes experiments/moral_machines/mm18_repair_sample.csv, mm18_repair_gt_sample.csv, mm18_repair_check.csv.

# %%
import os
from pathlib import Path

import sys

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18 import load_mm  # noqa: E402

OUT = Path("experiments/moral_machines")
N_PER_GROUP, N_CHECK, SEED = 50, 20, 0
cand = pd.read_csv(OUT / "mm18_repair_candidates.csv")
pool = cand[cand["change_size"] == "one rule feature"].copy()
pool["law_change"] = pool["changed_factors"].str.split(",").map(lambda f: "law" in f)
# The candidate file's "cited" counts a pair as cited if ANY changed factor is cited, and almost every pair also changes
# head counts ("numbers"), which almost every factor list mentions. Re-label by the one rule feature that changed.
pool["primary_factor"] = pool["changed_factors"].str.split(",").map(lambda f: ([x for x in f if x != "numbers"] or ["numbers"])[0])
_src = Path("analysis/mm18_repair.py").read_text()  # reuse the same keyword map without re-running that script
_ns = {"np": np}
exec(_src[_src.index("KEYWORDS = {"):_src.index("# %% Held-out P(Case 1)")], _ns)
import re
# Gate 2a found false matches from substrings ("red" in "predictability"/"compared") and from "right-of-way" (pedestrian
# priority, not signals). Match keywords only at the start of a word, and drop right-of-way.
_KW = {f: [k for k in kws if "right" not in k] for f, kws in _ns["KEYWORDS"].items()}
_RX = {f: re.compile(r"\b(?:" + "|".join(map(re.escape, kws)) + ")", re.I) for f, kws in _KW.items()}


def cited_fn(factors):
    text = " ".join(map(str, factors if isinstance(factors, (list, tuple, np.ndarray)) else []))
    return {f for f, rx in _RX.items() if rx.search(text)}
d = load_mm(with_text=True)
factors_of = {}
for side, idx_col, resp_col in (("o", "original_question_idx", "original_reference_response_parsed_response"),
                                ("c", "counterfactual_question_idx", "counterfactual_reference_response_parsed_response")):
    for m, i, r in zip(d["model"], d[idx_col], d[resp_col]):
        factors_of[(m, i)] = r.get("most_important_factors") if isinstance(r, dict) else None
pool["original_factors"] = [factors_of.get((m, i)) for m, i in zip(pool["model"], pool["original_idx"])]
pool["cited_keywords"] = [",".join(sorted(cited_fn(f))) for f in pool["original_factors"]]
pool["changed_cited"] = [pf in ck.split(",") for pf, ck in zip(pool["primary_factor"], pool["cited_keywords"])]
pool["cited_label"] = np.where(pool["changed_cited"], "cited", "uncited")
pool["original_factors"] = pool["original_factors"].map(lambda f: " | ".join(map(str, f)) if f is not None else "")
rng = np.random.default_rng(SEED)


def draw(g, n, balance_format=False):
    """n pairs from g: law changes <= n/2, half flips where possible, no scenario reused (as original or counterfactual)."""
    g = g.sample(frac=1, random_state=SEED)
    picked, used = [], set()
    quotas = [(af, fl) for af in ([True, False] if balance_format else [None]) for fl in (True, False)]
    per_quota = n // len(quotas)
    law_cap = n // 2
    n_law = 0

    def take(rows, k):
        nonlocal n_law
        out = []
        for r in rows.itertuples():
            if len(out) >= k:
                break
            if r.original_idx in used or r.counterfactual_idx in used or (r.law_change and n_law >= law_cap):
                continue
            used.update([r.original_idx, r.counterfactual_idx])
            n_law += r.law_change
            out.append(r.Index)
        return out

    for af, fl in quotas:
        sub = g[(g["flipped"] == fl) & ((g["answer_first"] == af) if af is not None else True)]
        sub = pd.concat([sub[~sub["law_change"]], sub[sub["law_change"]]]).sample(frac=1, random_state=SEED)
        picked += take(sub, per_quota)
    picked += take(g.drop(picked), n - len(picked))  # top up if a quota ran short
    return g.loc[picked]


main = pd.concat([draw(g, N_PER_GROUP, balance_format=(m == "Gemma 3 27B"))
                  for (m, tie), g in pool.groupby(["model", "tie_label"])], ignore_index=True)

# Gemma ground-truth arm: both scenarios from experiment 1's held-out pairs
split = pd.read_csv(OUT / "mm18_gt_split.csv")
held_pairs = split[(split["model"] == "Gemma 3 27B") & (split["split"] == "heldout")]["pair_id"]
held_scen = {int(x) for p in held_pairs for x in p.split("-")}
gpool = pool[(pool["model"] == "Gemma 3 27B") & pool["original_idx"].isin(held_scen) & pool["counterfactual_idx"].isin(held_scen)]
gt_arm = pd.concat([draw(g, N_PER_GROUP, balance_format=True) for _, g in gpool.groupby("tie_label")], ignore_index=True)

# Gate 2a: 20 pairs the keyword match calls "cited", spread over models
cited = main[main["cited_label"] == "cited"]
check = pd.concat([g.sample(min(len(g), -(-N_CHECK // 3)), random_state=SEED) for _, g in cited.groupby("model")]).head(N_CHECK)

main.to_csv(OUT / "mm18_repair_sample.csv", index=False)
gt_arm.to_csv(OUT / "mm18_repair_gt_sample.csv", index=False)
if not (OUT / "mm18_repair_check.csv").exists():  # keep the already hand-checked Gate 2a set
    check.to_csv(OUT / "mm18_repair_check.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    summ = lambda s, by: s.groupby(by).agg(pairs=("flipped", "size"), flips=("flipped", "mean"), law=("law_change", "mean"),
                                           cited=("changed_cited", "mean"), answer_first=("answer_first", "mean")).round(2)
    print("Main sample:"); print(summ(main, ["model", "tie_label"]), "\n")
    print(f"Gemma ground-truth arm (held-out scenarios only; {len(gpool)} candidates):"); print(summ(gt_arm, ["tie_label"]), "\n")
    print("Changed rule feature in the main sample:")
    print(main.groupby("model")["primary_factor"].value_counts().unstack(fill_value=0), "\n")
    print("Cited (primary factor in the original's factor list), by model x tie:")
    print(main.groupby(["model", "tie_label"])["changed_cited"].mean().unstack().round(2), "\n")
    print(f"Gate 2a check: {len(check)} pairs", check["model"].value_counts().to_dict())
