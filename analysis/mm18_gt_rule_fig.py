# %% [markdown]
# # Experiment 1: held-out accuracy of the training-only logistic rule, against simple baselines
#
# Run from the repo root: `python analysis/mm18_gt_rule_fig.py`. No API calls. Reads mm18_gt_split.csv and
# mm18_gt_heldout_rule.csv (from mm18_gt_prep.py) and refits only the baselines. Writes
# figures/moral_machines/exp1_rule_heldout.png and experiments/moral_machines/mm18_gt_rule_baselines.csv.
# Scored on the counterfactual scenario of each model's 100 held-out pairs; intervals are 2,000 bootstrap resamples.

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import mm18_rule as R  # noqa: E402

IN, FIG = Path("experiments/moral_machines"), Path("figures/moral_machines")
FIG.mkdir(parents=True, exist_ok=True)
GT_MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
C1 = "#2a78d6"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "legend.frameon": False})
N_BOOT, SEED = 2000, 0
rng = np.random.default_rng(SEED)

split = pd.read_csv(IN / "mm18_gt_split.csv")
rule = pd.read_csv(IN / "mm18_gt_heldout_rule.csv")


def ci(right):
    right = np.asarray(right, dtype=float)
    boots = rng.choice(right, (N_BOOT, len(right))).mean(axis=1)
    return right.mean(), *np.percentile(boots, [2.5, 97.5])


rows = []
for m in GT_MODELS:
    g = R.d[R.d["model"] == m].reset_index(drop=True)
    dec = R.decisions(g)
    dec["split"] = dec["pair_id"].map(split[split["model"] == m].set_index("pair_id")["split"])
    tr, te = dec[dec["split"] == "train"], dec[(dec["split"] == "heldout") & (dec["side"] == "c")]
    majority = int(tr["y"].mean() > 0.5)
    position = (R.fit(tr[["answer_first"]], tr["y"]).predict(te[["answer_first"]]) if tr["answer_first"].std() > 0
                else np.full(len(te), majority))
    r = rule[rule["model"] == m]
    preds = {"Logistic rule (all features)": r["rule_answer"] == r["reference_answer"],
             "Answer position only": position == te["y"].to_numpy(),
             "Always the majority answer": te["y"].to_numpy() == majority}
    for name, right in preds.items():
        acc, lo, hi = ci(right)
        rows.append(dict(model=m, predictor=name, n=len(right), acc=acc, lo=lo, hi=hi,
                         train_pairs=tr["pair_id"].nunique(), majority=f"CASE {2 - majority}"))
res = pd.DataFrame(rows)
res.to_csv(IN / "mm18_gt_rule_baselines.csv", index=False)

# %% Figure: one row per model, rule as the filled dot, majority baseline as the open square.
# The answer-position-only rule predicts Case 1 everywhere for all three models, so it matches the majority baseline
# exactly; it stays in the CSV but not the figure.
fig, ax = plt.subplots(figsize=(8.5, 3.4))
style = {"Logistic rule (all features)": dict(color=C1, marker="o", s=90, zorder=4),
         "Always the majority answer": dict(facecolor="white", edgecolor=GREY, marker="s", s=60, linewidth=2, zorder=2)}
offset = {"Logistic rule (all features)": -0.14, "Always the majority answer": 0.14}
for i, m in enumerate(GT_MODELS):
    for name, st in style.items():
        r = res[(res["model"] == m) & (res["predictor"] == name)].iloc[0]
        y = i + offset[name]
        ax.plot([r.lo, r.hi], [y, y], color=st.get("color", st.get("edgecolor")), linewidth=2, solid_capstyle="round",
                zorder=st["zorder"] - 1)
        ax.scatter(r.acc, y, label=name if i == 0 else None, **st)
        ax.annotate(f"{r.acc:.0%}", (r.hi, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=11.5, color=INK if st["zorder"] == 4 else MUTED,
                    fontweight="bold" if st["zorder"] == 4 else "normal")
ax.set_yticks(range(len(GT_MODELS)))
ax.set_yticklabels([f"{m}\n{res[res['model'] == m]['train_pairs'].iloc[0]} train · 100 held-out" for m in GT_MODELS],
                   fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0.35, 1.05)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Accuracy on held-out counterfactual answers (95% bootstrap interval)")
ax.grid(axis="y", visible=False)
ax.set_title("Experiment 1: training-only logistic rule vs. majority baseline", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=10.5, handletextpad=0.3, columnspacing=1.2)
fig.savefig(FIG / "exp1_rule_heldout.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

if __name__ == "__main__":
    print(res.round(3).to_string(index=False))
