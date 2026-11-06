# %% [markdown]
# # Predicting Gemma 3 27B on Moral Machines: accuracy and NSG with and without its explanation
#
# Run from the repo root: `python analysis/gemma_stress_test/gemma27b_predictor_nsg.py`. No API calls.
# Reads data_parquet/gemma_3_predictions.parquet, keeps moral_machines rows with google/gemma-3-27b-it as the
# reference, both answer positions pooled (original_answer_first is not split). Rows with an invalid counterfactual
# reference answer are dropped; an invalid predictor answer counts as wrong. All five predictors are read from the
# *_predictor_names / *_predictor_answers lists (the scalar *_answer columns hold only the first predictor).
# Per predictor: accuracy with and without the explanation (answer equals counterfactual_reference_response_answer)
# and NSG = (acc_with - acc_without) / (1 - acc_without). 95% intervals: bootstrap over scenarios, one resample shared
# by both conditions so NSG stays paired.
# Writes figures/moral_machines/gemma_stress_test/gemma27b_predictor_nsg.png and
# experiments/moral_machines/gemma_stress_test/gemma27b_predictor_nsg.csv.

# %%
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if Path.cwd().name in {"analysis", "gemma_stress_test"}:
    os.chdir(Path(__file__).resolve().parents[2])

DATA = Path("data_parquet/gemma_3_predictions.parquet")
FIG = Path("figures/moral_machines/gemma_stress_test")
OUT = Path("experiments/moral_machines/gemma_stress_test")
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "google/gemma-3-27b-it"
CF_ANS = "counterfactual_reference_response_answer"
PRED = "counterfactual_predictor_response_{}_explanation_predictor_{}"
CONDITIONS = ["with", "without"]
N_BOOT, SEED = 2000, 0

COND_COLORS = {"with": "#4292c6", "without": "#deebf7"}  # ColorBrewer Blues, as in description_test.py
NSG_COLOR = "#4292c6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 300, "legend.frameon": True,
                     "legend.edgecolor": "0.8"})

def short(name):
    """'google/gemma-3-27b-it' -> 'gemma-3-27b'; 'google/gemini-3-flash-preview' -> 'gemini-3-flash'."""
    return name.split("/")[-1].removesuffix("-it").removesuffix("-preview")


# %% one row per scenario, one correct-or-not column per (predictor, condition)
cols = ["original_dataset", "original_reference_response_model_info_model", "counterfactual_question_idx", CF_ANS]
cols += [PRED.format(c, f) for c in CONDITIONS for f in ["names", "answers"]]
t = pq.read_table(DATA, columns=cols, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
t = t[(t["original_reference_response_model_info_model"] == TARGET) & t[CF_ANS].notna()]
t = t.drop_duplicates("counterfactual_question_idx").reset_index(drop=True)

correct = {}
for cond in CONDITIONS:
    for names, answers, target, i in zip(t[PRED.format(cond, "names")], t[PRED.format(cond, "answers")], t[CF_ANS],
                                         range(len(t))):
        for pred, a in zip(names, answers):
            correct.setdefault((pred, cond), np.zeros(len(t)))[i] = a == target
predictors = list(dict.fromkeys(p for p, _ in correct))
print(f"{TARGET}: {len(t)} counterfactual scenarios (both answer positions); predictors {[short(p) for p in predictors]}")


# %% accuracy and NSG with paired bootstrap intervals
def nsg(w, wo):
    return (w.mean() - wo.mean()) / (1 - wo.mean())


rng = np.random.default_rng(SEED)
idx = [rng.integers(0, len(t), len(t)) for _ in range(N_BOOT)]
rows = []
for pred in predictors:
    w, wo = correct[(pred, "with")], correct[(pred, "without")]
    row = {"predictor": short(pred), "n": len(t)}
    for name, fn in [("acc_with", lambda i: w[i].mean()), ("acc_without", lambda i: wo[i].mean()),
                     ("nsg", lambda i: nsg(w[i], wo[i]))]:
        reps = [fn(i) for i in idx]
        row[name], (row[f"{name}_lo"], row[f"{name}_hi"]) = fn(slice(None)), np.percentile(reps, [2.5, 97.5])
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(OUT / "gemma27b_predictor_nsg.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))

# %% figure: accuracy (a), NSG (b); plain matplotlib style for the paper
ERR = {"ecolor": "black", "elinewidth": 1.2, "capsize": 3}
EDGE = {"edgecolor": "0.25", "linewidth": 0.8, "zorder": 2}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
x, w = np.arange(len(res)), 0.38
for k, cond in enumerate(["without", "with"]):
    v, lo, hi = 100 * res[f"acc_{cond}"], 100 * res[f"acc_{cond}_lo"], 100 * res[f"acc_{cond}_hi"]
    ax.bar(x + (k - 0.5) * w, v, w, yerr=[v - lo, hi - v], color=COND_COLORS[cond], label=f"{cond} explanation",
           error_kw=ERR, **EDGE)
ax.set_xlim(-0.5, len(res) - 0.5)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("(a) accuracy", labelpad=10)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9.5, framealpha=1)

v, lo, hi = res["nsg"], res["nsg_lo"], res["nsg_hi"]
bx.bar(x, v, 0.6, yerr=[v - lo, hi - v], color=NSG_COLOR, error_kw=ERR, **EDGE)
bx.axhline(0, color="black", lw=1, zorder=3)
bx.set_xlim(-0.5, len(res) - 0.5)
bx.set_ylim(np.floor(min(lo.min(), 0) * 4) / 4, np.ceil(max(hi.max(), 0) * 4) / 4)
bx.set_xticks(x, res["predictor"], rotation=20, ha="right")
ax.set_xticks(x, res["predictor"], rotation=20, ha="right")
bx.set_ylabel("NSG")
bx.set_xlabel("(b) normalised simulatability gain", labelpad=10)
fig.tight_layout()
for ext in ["png", "pdf"]:  # 300 dpi PNG and vector PDF for the paper
    fig.savefig(FIG / f"gemma27b_predictor_nsg.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nWrote {FIG / 'gemma27b_predictor_nsg.png'}, .pdf and {OUT / 'gemma27b_predictor_nsg.csv'}")
