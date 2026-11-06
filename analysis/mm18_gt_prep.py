# %% [markdown]
# # Experiment 1 prep: held-out split, training-only rule, and the material for writing ground-truth explanations
#
# Run from the repo root: `python analysis/mm18_gt_prep.py`. No API calls. Writes to experiments/moral_machines/:
# - mm18_gt_split.csv: (model, pair_id, split) with 20 held-out pairs per pair type per model
# - mm18_gt_train_coefs.csv: rule refit on training pairs only (standardised coefs + 95% pair-bootstrap intervals)
# - mm18_gt_heldout_rule.csv: the training-only rule's prediction on each held-out pair (the ceiling)
# - mm18_gt_train_material.md: coefficients + 40 sampled TRAINING decisions per model, the only input used to write
#   the ground-truth explanations (held-out pairs never appear in it)

# %%
import ast
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import mm18_rule as R  # noqa: E402  (fits the full rule on import; we only reuse its data and helpers)
from mm18 import PAIR_TYPES  # noqa: E402

OUT = Path("experiments/moral_machines")
GT_MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
N_HELD_PER_TYPE, N_SHOW_PER_TYPE, N_BOOT, SEED = 20, 8, 200, 0
raw = pd.read_csv(R.RAW_CSV, index_col=0)


def describe(idx):
    """Compact one-line rendering of a scenario: who dies in each case, where, and the lights."""
    r = raw.loc[idx]
    c1, c2 = ast.literal_eval(r["count_dict_1"]), ast.literal_eval(r["count_dict_2"])
    light = ast.literal_eval(r["traffic_light_pattern"])
    fmt = lambda c: ", ".join(f"{v} {k}" for k, v in c.items())
    lt = lambda l: {"green": " (crossing legally)", "red": " (crossing illegally)", "NA": ""}[l]
    act1, act2 = ("swerve", "stay") if r["is_interventionism"] else ("stay", "swerve")
    where1 = "passengers" if r["is_in_car"] else "pedestrians"
    return (f"Case 1 = {act1}, kills {fmt(c1)} [{where1}{lt(light[0])}] | "
            f"Case 2 = {act2}, kills {fmt(c2)} [pedestrians{lt(light[1])}]")


split_rows, coef_rows, held_rows, material = [], [], [], ["# Training material for the ground-truth explanations\n",
    "Coefficients are from a rule fitted on TRAINING pairs only (standardised; positive = pushes towards Case 1).",
    "Decisions are a sample of training scenarios. Held-out pairs never appear here.\n"]
rng = np.random.default_rng(SEED)
for m in GT_MODELS:
    g = R.d[R.d["model"] == m].reset_index(drop=True)
    held = pd.concat([g[g["what_changed"] == pt].sample(N_HELD_PER_TYPE, random_state=SEED) for pt in PAIR_TYPES])
    g["split"] = np.where(g["pair_id"].isin(held["pair_id"]), "heldout", "train")
    split_rows.append(g[["model", "pair_id", "what_changed", "split"]])
    dec = R.decisions(g)
    dec["split"] = dec["pair_id"].map(g.set_index("pair_id")["split"])
    tr, te = dec[dec["split"] == "train"], dec[dec["split"] == "heldout"]
    cols = [c for c in R.FEATURES if tr[c].std() > 0]
    full = R.fit(tr[cols], tr["y"])
    pairs = tr["pair_id"].unique()
    by_pair = tr.groupby("pair_id").indices
    boots = []
    for _ in range(N_BOOT):
        ix = np.concatenate([by_pair[p] for p in rng.choice(pairs, len(pairs))])
        boots.append(R.fit(tr.iloc[ix][cols], tr["y"].iloc[ix])[-1].coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ct = pd.DataFrame(dict(model=m, feature=cols, label=[R.FEATURE_LABEL[c] for c in cols], coef=full[-1].coef_[0], lo=lo, hi=hi))
    coef_rows.append(ct)
    # held-out ceiling: training-only rule on the held-out counterfactual scenarios
    cf = te[te["side"] == "c"].copy()
    cf["rule_p_case1"] = full.predict_proba(cf[cols])[:, 1]
    gh = g[g["split"] == "heldout"].set_index("pair_id")
    cf["reference_answer"] = gh.loc[cf["pair_id"], "counterfactual_reference_response_answer"].to_numpy()
    cf["rule_answer"] = np.where(cf["rule_p_case1"] > 0.5, "CASE 1", "CASE 2")
    held_rows.append(cf.assign(model=m)[["model", "pair_id", "rule_p_case1", "rule_answer", "reference_answer"]])
    # training material: coefficients + 40 sampled training decisions (8 per pair type, one side each)
    gt = g[g["split"] == "train"]
    show = pd.concat([gt[gt["what_changed"] == pt].sample(N_SHOW_PER_TYPE, random_state=SEED) for pt in PAIR_TYPES])
    acc = (cf["rule_answer"] == cf["reference_answer"]).mean()
    material += [f"\n## {m}\n", f"Training pairs: {len(gt)}. Training-rule accuracy on held-out counterfactuals: {acc:.1%} "
                 "(reported as the ceiling; not used to write the text).\n", "| Factor | coef | 95% CI |", "|---|---|---|"]
    material += [f"| {r.label} | {r.coef:+.2f} | [{r.lo:+.2f}, {r.hi:+.2f}] |"
                 for r in ct.reindex(ct["coef"].abs().sort_values(ascending=False).index).itertuples()]
    material += ["", f"Intercept (not a clean Case 1 bias): {full[-1].intercept_[0]:+.2f}", "",
                 "Answer-first share of training prompts: "
                 f"{gt['o_answer_first'].mean():.0%}. Case 1 rate: answer first "
                 f"{tr[tr['answer_first'] == 1]['y'].mean():.0%}, answer last {tr[tr['answer_first'] == -1]['y'].mean():.0%}.\n",
                 "### 40 training decisions (theme · answer position · scenario → choice)\n"]
    for k, r in enumerate(show.itertuples(), 1):
        side_idx = r.original_question_idx if k % 2 else r.counterfactual_question_idx
        ans = r.original_reference_response_answer if k % 2 else r.counterfactual_reference_response_answer
        pos = "answer first" if r.o_answer_first else "answer last"
        material.append(f"{k}. {r.dimension} · {pos} · {describe(side_idx)} → **{ans}**")

pd.concat(split_rows).to_csv(OUT / "mm18_gt_split.csv", index=False)
pd.concat(coef_rows).to_csv(OUT / "mm18_gt_train_coefs.csv", index=False)
pd.concat(held_rows).to_csv(OUT / "mm18_gt_heldout_rule.csv", index=False)
(OUT / "mm18_gt_train_material.md").write_text("\n".join(material) + "\n")

if __name__ == "__main__":
    s = pd.concat(split_rows)
    print(s.groupby(["model", "split"]).size().unstack())
    h = pd.concat(held_rows)
    print("\ntraining-only rule accuracy on held-out counterfactuals:")
    print(h.assign(ok=h["rule_answer"] == h["reference_answer"]).groupby("model")["ok"].mean().round(3).to_string())
