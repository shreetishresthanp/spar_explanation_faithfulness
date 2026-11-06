# %% [markdown]
# # Experiment 1 analysis: does a ground-truth description of the model beat its own explanation?
#
# Run from the repo root after `python analysis/mm18_gt_run.py --exp 1`: `python analysis/mm18_gt.py`.
# Writes to experiments/moral_machines/:
# - mm18_gt_results.csv: accuracy and NSG per model x condition (pooled over simulators, and per simulator), 95% CIs
# - mm18_gt_contrasts.csv: paired NSG differences (ground truth - own, tree - own, gt+own - gt) with 95% CIs
# - mm18_gt_drift.csv: agreement of the new "own"/"none" answers with the paper's stored answers (same simulator)
# NSG for a condition = (acc_condition - acc_none) / (1 - acc_none), on the same held-out pairs and simulators.
# Intervals: bootstrap over held-out pairs (2,000 replicates), keeping each pair's answers together.

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18 import P, load_mm  # noqa: E402

OUT = Path("experiments/moral_machines")
N_BOOT, SEED = 2000, 0
CONDS = ["none", "own", "gt", "tree", "gt+own"]
LABEL = {"none": "no explanation", "own": "model's own explanation", "gt": "ground truth (written)",
         "tree": "ground truth (decision tree)", "gt+own": "written ground truth + own"}
SIM_SHORT = {"openrouter/openai/gpt-oss-20b": "gpt-oss-20b", "openrouter/google/gemini-3-flash-preview": "gemini-3-flash"}

pr = pd.read_csv(OUT / "mm18_gt_predictions.csv").query("run == 'exp1'")
pr["correct"] = (pr["answer"] == pr["reference_answer"]).astype(float)
pr.loc[pr["answer"].isna(), "correct"] = np.nan  # unparsed answers are dropped, as in the paper's NSG
pr["sim"] = pr["simulator"].map(SIM_SHORT)
ceiling = pd.read_csv(OUT / "mm18_gt_heldout_rule.csv")
ceiling["rule_right"] = ceiling["rule_answer"] == ceiling["reference_answer"]


def table(g):
    """pair x condition matrix of (correct count, parsed count), summed over the simulators in g."""
    c = g.pivot_table(index="pair_id", columns="condition", values="correct", aggfunc="sum").reindex(columns=CONDS)
    n = g.assign(ok=g["correct"].notna()).pivot_table(index="pair_id", columns="condition", values="ok", aggfunc="sum").reindex(columns=CONDS)
    return c.fillna(0).to_numpy(), n.fillna(0).to_numpy()


def stats(c, n):
    acc = c.sum(0) / n.sum(0)
    nsg = (acc - acc[0]) / (1 - acc[0])
    return acc, nsg


rows, contrasts = [], []
rng = np.random.default_rng(SEED)
for m, gm in pr.groupby("model"):
    for scope, g in [("both simulators", gm)] + [(s, gs) for s, gs in gm.groupby("sim")]:
        c, n = table(g)
        acc, nsg = stats(c, n)
        b_acc, b_nsg = [], []
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(c), len(c))
            a, s_ = stats(c[ix], n[ix])
            b_acc.append(a), b_nsg.append(s_)
        b_acc, b_nsg = np.array(b_acc), np.array(b_nsg)
        for k, cond in enumerate(CONDS):
            rows.append(dict(model=m, simulator=scope, condition=cond, label=LABEL[cond], pairs=len(c),
                             answers=int(n[:, k].sum()), acc=acc[k], acc_lo=np.percentile(b_acc[:, k], 2.5),
                             acc_hi=np.percentile(b_acc[:, k], 97.5), nsg=nsg[k] if k else np.nan,
                             nsg_lo=np.percentile(b_nsg[:, k], 2.5) if k else np.nan,
                             nsg_hi=np.percentile(b_nsg[:, k], 97.5) if k else np.nan))
        for a_, b_ in [("gt", "own"), ("tree", "own"), ("gt+own", "gt"), ("gt", "tree")]:
            ia, ib = CONDS.index(a_), CONDS.index(b_)
            dif = b_nsg[:, ia] - b_nsg[:, ib]
            contrasts.append(dict(model=m, simulator=scope, contrast=f"{a_} - {b_}", nsg_diff=nsg[ia] - nsg[ib],
                                  lo=np.percentile(dif, 2.5), hi=np.percentile(dif, 97.5), p_le_0=(dif <= 0).mean()))
res, con = pd.DataFrame(rows), pd.DataFrame(contrasts)
ceil = ceiling.groupby("model")["rule_right"].mean().rename("rule_acc_heldout")

# Drift: new answers vs the paper's stored answers for the same simulator, pair and condition
d = load_mm()
d["pair_id"] = d["original_question_idx"].astype(str) + "-" + d["counterfactual_question_idx"].astype(str)
d = d[d["pair_id"].isin(pr["pair_id"]) & d["model"].isin(pr["model"].unique())]
stored = []
for _, r in d.iterrows():
    for nm, aw, awo in zip(r[P + "with_explanation_predictor_names"], r[P + "with_explanation_predictor_answers"],
                           r[P + "without_explanation_predictor_answers"]):
        short = {"openai/gpt-oss-20b": "gpt-oss-20b", "google/gemini-3-flash-preview": "gemini-3-flash"}.get(nm)
        if short:
            stored += [dict(model=r["model"], pair_id=r["pair_id"], sim=short, condition="own", stored=aw),
                       dict(model=r["model"], pair_id=r["pair_id"], sim=short, condition="none", stored=awo)]
st = pd.DataFrame(stored).merge(pr[["model", "pair_id", "sim", "condition", "answer", "reference_answer"]],
                                on=["model", "pair_id", "sim", "condition"])
st["agree"] = st["stored"] == st["answer"]
st["stored_right"], st["new_right"] = st["stored"] == st["reference_answer"], st["answer"] == st["reference_answer"]
drift = st.groupby(["model", "sim", "condition"]).agg(n=("agree", "size"), agreement=("agree", "mean"),
                                                       stored_acc=("stored_right", "mean"), new_acc=("new_right", "mean")).reset_index()

res.to_csv(OUT / "mm18_gt_results.csv", index=False)
con.to_csv(OUT / "mm18_gt_contrasts.csv", index=False)
drift.to_csv(OUT / "mm18_gt_drift.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"{len(pr)} predictions, parsed {pr['answer'].notna().mean():.1%}\n")
    t = res[res["simulator"] == "both simulators"].copy()
    t["acc"] = t.apply(lambda r: f"{r.acc:.1%} [{r.acc_lo:.1%}, {r.acc_hi:.1%}]", axis=1)
    t["nsg"] = t.apply(lambda r: "—" if np.isnan(r.nsg) else f"{r.nsg:+.1%} [{r.nsg_lo:+.1%}, {r.nsg_hi:+.1%}]", axis=1)
    print("Pooled over both simulators:")
    print(t.pivot(index="condition", columns="model", values="nsg").reindex(CONDS).to_string(), "\n")
    print(t.pivot(index="condition", columns="model", values="acc").reindex(CONDS).to_string(), "\n")
    print("Rule (training-only logistic) accuracy on the same held-out pairs:", ceil.round(3).to_dict(), "\n")
    c = con.copy()
    c["diff"] = c.apply(lambda r: f"{r.nsg_diff:+.1%} [{r.lo:+.1%}, {r.hi:+.1%}]", axis=1)
    print("Paired NSG differences:")
    print(c.pivot_table(index=["contrast", "simulator"], columns="model", values="diff", aggfunc="first").to_string(), "\n")
    print("Drift vs the paper's stored answers:")
    print(drift.round(3).to_string(index=False))
