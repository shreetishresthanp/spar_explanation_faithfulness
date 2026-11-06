# %% [markdown]
# # Experiment 2 analysis: re-paired counterfactuals, near-tie vs clear (headline), cited vs uncited (secondary)
#
# Run from the repo root after `python analysis/mm18_gt_run.py --exp 2`: `python analysis/mm18_repair_analysis.py`.
# Accuracy pools both simulators and both repeats. NSG = (acc_own - acc_none) / (1 - acc_none) within a group.
# Intervals: bootstrap over pairs (2,000 replicates), keeping each pair's answers (all simulators/repeats) together.
# Writes experiments/moral_machines/mm18_repair_results.csv and mm18_repair_contrasts.csv.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)

OUT = Path("experiments/moral_machines")
N_BOOT, SEED = 2000, 0
pr = pd.read_csv(OUT / "mm18_repair_predictions.csv")
pr["arm"] = np.where(pr["run"].str.startswith("exp2gt"), "gemma ground-truth arm", "main")
pr["correct"] = np.where(pr["answer"].isna(), np.nan, (pr["answer"] == pr["reference_answer"]).astype(float))
lab = pr["what_changed"].str.split(" · ", expand=True)
pr["factor"], pr["tie"], pr["cited"] = lab[0], lab[1], lab[2]
samp = pd.concat([pd.read_csv(OUT / "mm18_repair_sample.csv").assign(arm="main"),
                  pd.read_csv(OUT / "mm18_repair_gt_sample.csv").assign(arm="gemma ground-truth arm")])
samp["pair_id"] = samp["original_idx"].astype(str) + "-" + samp["counterfactual_idx"].astype(str)
pr = pr.merge(samp[["arm", "model", "pair_id", "answer_first", "flipped"]], on=["arm", "model", "pair_id"], how="left")
pr["format"] = np.where(pr["answer_first"], "answer first", "answer last")
rng = np.random.default_rng(SEED)


def mats(g, conds):
    c = g.pivot_table(index="pair_id", columns="condition", values="correct", aggfunc="sum").reindex(columns=conds).fillna(0).to_numpy()
    n = g.assign(ok=g["correct"].notna()).pivot_table(index="pair_id", columns="condition", values="ok", aggfunc="sum").reindex(columns=conds).fillna(0).to_numpy()
    return c, n


def nsg_of(c, n):
    acc = c.sum(0) / n.sum(0)
    return acc, (acc[1:] - acc[0]) / (1 - acc[0])


def summarise(g, conds=("none", "own")):
    c, n = mats(g, list(conds))
    acc, nsg = nsg_of(c, n)
    boots = [nsg_of(c[ix], n[ix]) for ix in (rng.integers(0, len(c), len(c)) for _ in range(N_BOOT))]
    bacc = np.array([b[0] for b in boots]); bnsg = np.array([b[1] for b in boots])
    out = dict(pairs=len(c), flip_rate=g.drop_duplicates("pair_id")["flipped"].mean())
    for k, cond in enumerate(conds):
        out[f"acc_{cond}"] = acc[k]
        if k:
            out[f"nsg_{cond}"], out[f"nsg_{cond}_lo"], out[f"nsg_{cond}_hi"] = nsg[k - 1], *np.percentile(bnsg[:, k - 1], [2.5, 97.5])
    return out, bnsg


rows, contrasts = [], []
main = pr[pr["arm"] == "main"]
frontier = ["GPT-5.2", "Gemini 3 Flash"]
scopes = {m: main[main["model"] == m] for m in main["model"].unique()}
scopes["GPT-5.2 + Gemini 3 Flash"] = main[main["model"].isin(frontier)]
for scope, g in scopes.items():
    by = {}
    for split, col in (("tie", "tie"), ("cited", "cited")):
        for level, gl in g.groupby(col):
            s, b = summarise(gl)
            rows.append(dict(arm="main", scope=scope, split=split, level=level, **s))
            by[(split, level)] = (s, b)
    if scope == "Gemma 3 27B":
        for (fmt, tie), gl in g.groupby(["format", "tie"]):
            s, _ = summarise(gl)
            rows.append(dict(arm="main", scope=scope, split="format x tie", level=f"{fmt} · {tie}", **s))
    for split, a, b_ in (("tie", "near-tie", "clear"), ("cited", "cited", "uncited")):
        if (split, a) in by and (split, b_) in by:
            (sa, ba), (sb, bb) = by[(split, a)], by[(split, b_)]
            d = ba[:, 0] - bb[:, 0]  # independent groups: difference of independent bootstrap draws
            contrasts.append(dict(scope=scope, contrast=f"{a} - {b_}", nsg_diff=sa["nsg_own"] - sb["nsg_own"],
                                  lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                  acc_own_diff=sa["acc_own"] - sb["acc_own"]))
gt = pr[pr["arm"] == "gemma ground-truth arm"]
for level, gl in [("all", gt)] + list(gt.groupby("tie")):
    s, _ = summarise(gl, ("none", "own", "gt"))
    rows.append(dict(arm="gemma ground-truth arm", scope="Gemma 3 27B", split="tie", level=level, **s))
res, con = pd.DataFrame(rows), pd.DataFrame(contrasts)
# simulator consistency between the two repeats
w = pr.pivot_table(index=["arm", "model", "pair_id", "condition", "simulator"], columns="run", values="answer", aggfunc="first")
rep = []
for arm, prefix in (("main", "exp2_r"), ("gemma ground-truth arm", "exp2gt_r")):
    cols = [c for c in w.columns if c.startswith(prefix)]
    if len(cols) == 2:
        x = w.loc[arm, cols].dropna()
        rep.append(x.assign(agree=x[cols[0]] == x[cols[1]]).groupby(level=["simulator"])["agree"].mean().rename(arm))
res.to_csv(OUT / "mm18_repair_results.csv", index=False)
con.to_csv(OUT / "mm18_repair_contrasts.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"{len(pr)} predictions, parsed {pr['answer'].notna().mean():.1%}\n")
    cols = ["scope", "level", "pairs", "flip_rate", "acc_none", "acc_own", "nsg_own", "nsg_own_lo", "nsg_own_hi"]
    print("Near-tie vs clear (main sample):"); print(res.query("arm == 'main' and split == 'tie'")[cols].round(3).to_string(index=False), "\n")
    print("Cited vs uncited (main sample):"); print(res.query("arm == 'main' and split == 'cited'")[cols].round(3).to_string(index=False), "\n")
    print("Gemma by answer format x tie:"); print(res.query("split == 'format x tie'")[cols].round(3).to_string(index=False), "\n")
    print("Contrasts (NSG own, difference between groups):"); print(con.round(3).to_string(index=False), "\n")
    g = res[res["arm"] == "gemma ground-truth arm"]
    print("Gemma ground-truth arm (held-out scenarios):")
    print(g[["level", "pairs", "acc_none", "acc_own", "acc_gt", "nsg_own", "nsg_gt", "nsg_gt_lo", "nsg_gt_hi"]].round(3).to_string(index=False), "\n")
    print("Repeat agreement (same prompt, two calls):"); print(pd.concat(rep, axis=1).round(3) if rep else "n/a")
