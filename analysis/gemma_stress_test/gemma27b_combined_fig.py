# %% [markdown]
# # One figure for section 4.4: Gemma 3 27B's explanation vs an accurate description of its behaviour
#
# Run from the repo root: `python analysis/gemma_stress_test/gemma27b_combined_fig.py [strict|lenient]`. No API calls.
# Combines two saved results as grouped bar panels so the section needs one figure instead of two:
#   (a) all Gemma 3 27B counterfactuals in the paper's run (both answer positions), accuracy without and with its own
#       explanation, NSG above each group: experiments/.../gemma27b_predictor_nsg.csv (gemma27b_predictor_nsg.py);
#   (b) the description test, held-out answer-last items, accuracy with no explanation, the own explanation, the
#       accurate description and the decoy: experiments/.../description_test_summary_<parsing>.csv (description_test.py).
# Same simulator order and condition colours in both panels. Error bars are the saved 95% bootstrap intervals.
# Simulators marked with a dagger were added to the description test after the preregistered three.
# Writes figures/moral_machines/gemma_stress_test/gemma27b_combined_<parsing>.{png,pdf}.

# %%
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

if Path.cwd().name in {"analysis", "gemma_stress_test"}:
    os.chdir(Path(__file__).resolve().parents[2])

PARSING = sys.argv[1] if len(sys.argv) > 1 else "strict"
OUT = Path("experiments/moral_machines/gemma_stress_test")
FIG = Path("figures/moral_machines/gemma_stress_test")
FIG.mkdir(parents=True, exist_ok=True)

# x order; keys are the names in the two CSVs (Fig. 9 file, description-test file)
SIMS = [("GPT-5 Mini", "gpt-5-mini", "openai/gpt-5-mini"),
        ("Gemini 3 Flash", "gemini-3-flash", "google/gemini-3-flash-preview"),
        ("Qwen3-32B†", "Qwen3-32B", "qwen/qwen3-32b"),
        ("gpt-oss-20b†", "gpt-oss-20b", "openai/gpt-oss-20b"),
        ("Gemma 3 27B", "gemma-3-27b", "google/gemma-3-27b-it")]
# condition: description-test label, legend label, colour
CONDS = {"none": ("A no explanation", "no explanation", "#b5b3a8"),
         "explanation": ("B own explanation", "Gemma's own explanation", "#db7043"),
         "accurate": ("C accurate description", "accurate description", "#4176cf"),
         "decoy": ("D decoy description", "decoy description", "#1f9e89")}

INK, MUTED, GRID, AXIS = "#3d3d3a", "#6b6a64", "#ebeae6", "#c9c8bf"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": AXIS,
                     "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
                     "grid.linewidth": 1.0, "axes.axisbelow": True, "figure.dpi": 300, "axes.titlesize": 10.5,
                     "axes.titlecolor": INK, "legend.frameon": False})
ERR = {"ecolor": INK, "elinewidth": 0.8, "capsize": 1.8}

# %%
full = pd.read_csv(OUT / "gemma27b_predictor_nsg.csv").set_index("predictor")
desc = pd.read_csv(OUT / f"description_test_summary_{PARSING}.csv")
ref = desc[desc["simulator"] == "reference"].set_index("metric")["value"]
always1 = ref["always CASE 1 baseline"]
n_full = int(full["n"].iloc[0])
n_test = int(desc[desc["metric"] == "accuracy, A no explanation"]["n"].max())


def acc(sim, label):
    r = desc[(desc["simulator"] == sim) & (desc["metric"] == f"accuracy, {label}")].iloc[0]
    return r["value"], r["lo"], r["hi"]


def bars(ax, x, vals, cond, fontsize):
    v, lo, hi = (np.array(a) for a in zip(*vals))
    b = ax.bar(x, v, width, yerr=[v - lo, hi - v], color=CONDS[cond][2], error_kw=ERR)
    ax.bar_label(b, labels=[f"{a:.0%}" for a in v], fontsize=fontsize, color=MUTED, padding=4)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.5],
                                                                                  "wspace": 0.06})
x = np.arange(len(SIMS))

# (a) paper's run: no explanation vs own explanation, NSG above each pair
width = 0.36
for k, (cond, col) in enumerate([("none", "acc_without"), ("explanation", "acc_with")]):
    vals = [(full.loc[s[1], col], full.loc[s[1], f"{col}_lo"], full.loc[s[1], f"{col}_hi"]) for s in SIMS]
    bars(ax, x + (k - 0.5) * width, vals, cond, 7)
for xi, s in zip(x, SIMS):
    r = full.loc[s[1]]
    color = CONDS["explanation"][2] if r["nsg_hi"] < 0 else CONDS["accurate"][2] if r["nsg_lo"] > 0 else INK
    ax.text(xi, 1.03, f"NSG\n{r['nsg']:+.2f}".replace("-", "−"), ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=color, linespacing=1.1)
ax.set_title(f"(a) Paper's run, all counterfactuals (n = {n_full})", loc="left", pad=8)

# (b) description test: four conditions
width = 0.2
for k, cond in enumerate(CONDS):
    vals = [acc(s[2], CONDS[cond][0]) for s in SIMS]
    bars(bx, x + (k - 1.5) * width, vals, cond, 6)
bx.axhline(always1, color=INK, lw=0.8, ls="--", zorder=1)
bx.set_title(f"(b) Held-out answer-last items (n = {n_test})", loc="left", pad=8)

for a in (ax, bx):
    a.set_xticks(x, [s[0] for s in SIMS], rotation=20, ha="right")
    a.set_xlim(-0.6, len(SIMS) - 0.4)
ax.set_ylim(0, 1.18)
ax.set_yticks(np.linspace(0, 1, 6))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("Accuracy vs Gemma's counterfactual answer")

fig.legend(handles=[Patch(color=c, label=lab) for _, lab, c in CONDS.values()], loc="upper center",
           bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=9.5, handlelength=1.2, columnspacing=1.8)
fig.text(0.01, -0.06, f"Error bars are 95% bootstrap intervals. Dashed line in (b): always predicting CASE 1 "
         f"({always1:.0%}). NSG in (a) is coloured where its interval excludes 0.\n† added to the description test (b) "
         "after the preregistered simulators.", fontsize=8, color=MUTED, va="top")
for ext in ["png", "pdf"]:
    fig.savefig(FIG / f"gemma27b_combined_{PARSING}.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {FIG / f'gemma27b_combined_{PARSING}.png'} and .pdf")
