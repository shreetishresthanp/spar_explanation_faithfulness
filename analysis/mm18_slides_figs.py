# %% [markdown]
# # Slide figures for the 10/27 update (debrief/10_27_update.md)
#
# Run from the repo root after the rule, coverage and experiment analyses: `python analysis/mm18_slides_figs.py`.
# Writes 16:9-friendly PNGs to debrief/10_27_figures/. Reads only result CSVs in experiments/moral_machines/.

# %%
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)

IN, OUT = Path("experiments/moral_machines"), Path("debrief/10_27_figures")
OUT.mkdir(parents=True, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pct(v):
    """Whole-number percent, rounded half up (0.885 -> 89%)."""
    return f"{int(np.floor(v * 100 + 0.5 + 1e-9))}%"


def bars(ax, x, vals, lo, hi, colors, fmt=None, width=0.8):
    ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=2, zorder=2)
    if lo is not None:
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                    ecolor=MUTED, elinewidth=1.2, capsize=0, zorder=3)
    for xi, v, h in zip(x, vals, hi if hi is not None else vals):
        ax.text(xi, max(v, 0) + (0.02 if hi is None else max(h - v, 0) + 0.02), pct(v), ha="center",
                va="bottom", fontsize=12, color=INK)


# %% 1. What drives each model's choices (fitted rule, |effect| of the top factors, plain-English direction)
LABEL = {  # feature -> (label if coef > 0, label if coef < 0)
    "c1_spares_lawful": ("Spare people crossing legally", "Spare people crossing illegally"),
    "c1_kills_passengers": ("Sacrifice passengers, not pedestrians", "Protect passengers over pedestrians"),
    "c1_swerves": ("Prefer swerving", "Avoid swerving (stay in lane)"),
    "lives_saved": ("Save more lives", "Save fewer lives"),
    "pets_saved": ("Save more animals", "Save fewer animals"),
    "theme_species": ("Save animals over humans", "Save humans over animals"),
    "theme_utilitarianism": ("Save the larger group", "Save the smaller group"),
    "theme_social_value": ("Spare higher-status people", "Spare lower-status people"),
    "theme_age": ("Spare older people", "Spare younger people"),
    "theme_gender": ("Spare men", "Spare women"),
    "theme_fitness": ("Spare fitter people", "Spare less fit people"),
    "answer_first": ("Answer format (Case 1 if answering first)", "Answer format (Case 1 if answering last)"),
}
co = pd.read_csv(IN / "mm18_rule_coefs.csv")
co = co[co["feature"] != "intercept"]
models = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
fig, axs = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
for ax, m, color in zip(axs, models, (C1, C1, C1)):
    t = co[co["model"] == m].assign(size=lambda x: x["coef"].abs()).nlargest(5, "size").iloc[::-1]
    labels = [LABEL[f][0 if c > 0 else 1] for f, c in zip(t["feature"], t["coef"])]
    cols = [C2 if f == "answer_first" else color for f in t["feature"]]
    ax.barh(labels, t["size"], color=cols, edgecolor="white", linewidth=2, zorder=2)
    ax.set_title(m, loc="left")
    ax.grid(axis="y", visible=False)
axs[-1].set_xlabel("strength of effect (fitted rule; orange = a cause the explanations never mention)")
fig.subplots_adjust(hspace=0.45)
fig.suptitle("What actually drives each model's choices (top 5 factors of the fitted rule)", x=0.01, ha="left",
             fontsize=15, fontweight="bold", y=1.03)
save(fig, "fig1_what_drives.png")

# %% 2. The near-tie gap: seen on the paper's pairs, not on purpose-built pairs
b = pd.read_csv(IN / "mm18_rule_boundary.csv")
b = b[(b["scope"] == "top 3 pooled") & (b["split"] == "near-tie (margin < 0.2)")].set_index("level")
r2 = pd.read_csv(IN / "mm18_repair_results.csv")
r2 = r2.query("arm == 'main' and split == 'tie' and scope == 'GPT-5.2 + Gemini 3 Flash'").set_index("level")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, t, col, lo, hi, title in (
        (axs[0], b, "nsg", "nsg_lo", "nsg_hi", "Paper's random pairs\n(top 3 models, 2,835 pairs)"),
        (axs[1], r2, "nsg_own", "nsg_own_lo", "nsg_own_hi", "Purpose-built pairs, experiment 2\n(GPT-5.2 + Gemini 3 Flash, 200 pairs)")):
    lv = ["clear", "near-tie"]
    bars(ax, [0, 1], [t.loc[l, col] for l in lv], [t.loc[l, lo] for l in lv], [t.loc[l, hi] for l in lv], [C1, C2])
    ax.set_xticks([0, 1], ["clear", "near-tie"])
    ax.set_title(title, loc="left", fontsize=13)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("explanation gain (NSG)")
axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axs[0].set_ylim(0, 0.6)
fig.suptitle("Near-ties looked harder to explain, but the gap didn't replicate", x=0.01, ha="left",
             fontsize=15, fontweight="bold", y=1.04)
save(fig, "fig2_near_tie.png")

# %% 3. Hand labels: what separates GPT-5.2 failures from successes
cl = pd.read_csv(IN / "mm18_coverage_labels_scored.csv")
cl = cl[cl["model"] == "GPT-5.2"]
f_, s_ = cl[cl["outcome"] == "failure"], cl[cl["outcome"] == "success"]
rows = [("Explanation cites the\nchanged factor", f_["in_scope"].mean(), s_["in_scope"].mean()),
        ("Near-tie under the\nfitted rule", f_["near_tie"].mean(), s_["near_tie"].mean()),
        ("Explanation, read literally,\npoints the wrong way", (f_["literal_reading"] == "wrong").mean(),
         (s_["literal_reading"] == "wrong").mean())]
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(rows))
for k, (lab, color, off) in enumerate((("failures (20)", C2, -0.2), ("successes (10)", C1, 0.2))):
    vals = [r[1 + k] for r in rows]
    bars(ax, x + off, vals, None, None, color, width=0.38)
ax.set_xticks(x, [r[0] for r in rows])
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.12)
ax.grid(axis="x", visible=False)
ax.legend(handles=[matplotlib.patches.Patch(color=C2, label="failures (20)"), matplotlib.patches.Patch(color=C1, label="successes (10)")],
          loc="upper right")
ax.set_ylabel("share of hand-labelled pairs")
ax.set_title("GPT-5.2 hand labels: citing the changed factor doesn't separate failures from successes",
             loc="left", fontsize=14)
save(fig, "fig3_hand_labels.png")

# %% 4. Experiment 1: accuracy by what the simulator sees
g = pd.read_csv(IN / "mm18_gt_results.csv").query("simulator == 'both simulators'")
conds = [("none", "no explanation", GREY), ("own", "model's own explanation", C1),
         ("gt", "description of its behaviour (written)", C3), ("tree", "description (automatic decision tree)", C4)]
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(models))
w = 0.2
for k, (cond, lab, color) in enumerate(conds):
    t = g[g["condition"] == cond].set_index("model").reindex(models)
    off = (k - 1.5) * w
    ax.bar(x + off, t["acc"], width=w, color=color, edgecolor="white", linewidth=2, label=lab, zorder=2)
    ax.errorbar(x + off, t["acc"], yerr=[t["acc"] - t["acc_lo"], t["acc_hi"] - t["acc"]], fmt="none", ecolor=MUTED,
                elinewidth=1.1, zorder=3)
    for xi, v in zip(x + off, t["acc"]):
        ax.text(xi, 0.505, pct(v), ha="center", va="bottom", fontsize=10.5, color="white", fontweight="bold")
ax.set_xticks(x, models)
ax.set_ylim(0.5, 1.0)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("simulator accuracy on the counterfactual")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=11)
ax.set_title("Experiment 1: a description of the model's behaviour only helps for Gemma", loc="left", fontsize=15)
save(fig, "fig4_exp1_accuracy.png")

# %% 5. Gemma: the description beats its own explanation on old and new pairs
ga = pd.read_csv(IN / "mm18_repair_results.csv").query("arm == 'gemma ground-truth arm' and level == 'all'").iloc[0]
gm = g[g["model"] == "Gemma 3 27B"].set_index("condition")
sets = [("Paper's pairs\n(experiment 1, 100 pairs)", gm.loc["none", "acc"], gm.loc["own", "acc"], gm.loc["gt", "acc"]),
        ("New purpose-built pairs\n(experiment 2, 50 pairs)", ga["acc_none"], ga["acc_own"], ga["acc_gt"])]
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(sets))
for k, (lab, color) in enumerate((("no explanation", GREY), ("Gemma's own explanation", C2), ("description of its behaviour", C3))):
    vals = [s[1 + k] for s in sets]
    bars(ax, x + (k - 1) * 0.26, vals, None, None, color, width=0.24)
ax.set_xticks(x, [s[0] for s in sets])
ax.set_ylim(0.5, 1.02)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("simulator accuracy")
ax.grid(axis="x", visible=False)
ax.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for l, c in
                   (("no explanation", GREY), ("Gemma's own explanation", C2), ("description of its behaviour", C3))],
          loc="upper right", fontsize=11)
ax.set_title("Gemma's explanation hides its real driver (answer format)", loc="left", fontsize=15)
save(fig, "fig5_gemma.png")
# %% 6. Gemma format test: same scenarios asked answer-first vs answer-last
fm = pd.read_csv(IN / "mm18_format_results.csv").set_index(["group", "metric"])
G, Cc = "Gemma 3 27B (fresh)", "Gemini 3 Flash control (fresh)"
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(2)
for k, (fmt, color) in enumerate((("answer first", C1), ("answer last", C2))):
    t = [fm.loc[(g_, f"Case 1 rate, {fmt}")] for g_ in (G, Cc)]
    bars(ax, x + (k - 0.5) * 0.36, [r["value"] for r in t], [r["lo"] for r in t], [r["hi"] for r in t], color, width=0.34)
ax.set_xticks(x, ["Gemma 3 27B", "Gemini 3 Flash (control)"])
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("share of answers that are Case 1")
ax.grid(axis="x", visible=False)
ax.legend(handles=[matplotlib.patches.Patch(color=C1, label="[ANSWER] asked first"), matplotlib.patches.Patch(color=C2, label="[ANSWER] asked last")],
          loc="upper right", fontsize=11)
ax.set_title("Same 100 scenarios, only the answer position changes", loc="left", fontsize=15)
save(fig, "fig6_gemma_format.png")
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
