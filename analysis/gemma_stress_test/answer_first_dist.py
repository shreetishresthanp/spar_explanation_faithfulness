# %% [markdown]
# # Gemma 3 on Moral Machines: answer position and reference answers
#
# Run from the repo root: `python analysis/gemma_stress_test/answer_first_dist.py`. No API calls.
# Reads data_parquet/gemma_3_predictions.parquet, keeps moral_machines rows (one row per (model, scenario)), and plots
#   1. how many scenarios have original_answer_first True vs False, overall and per Gemma model;
#   2. the reference answer (CASE 1 / CASE 2) split by original_answer_first, pooled over models and per model;
#   3. the same for the counterfactual reference answer (missing answers shown as "invalid"), per counterfactual model;
#   4. the predictors' counterfactual answers with and without the explanation, per predictor. The scalar
#      counterfactual_predictor_response_*_answer / *_model_info_model columns only hold the first predictor
#      (gpt-oss-20b), so all predictors are read from the *_predictor_names / *_predictor_answers lists;
#   5. each predictor's accuracy (its answer equals counterfactual_reference_response_answer) with and without the
#      explanation, by original_answer_first. Rows with an invalid counterfactual reference answer are dropped; an
#      invalid predictor answer counts as wrong. Error bars: 95% normal-approximation binomial intervals.
# Writes figures/moral_machines/gemma_stress_test/{answer_first_counts,answer_by_position,cf_answer_by_position,
# predictor_answer_by_position,predictor_accuracy_by_position}.png and prints the tables.

# %%
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

if Path.cwd().name in {"analysis", "gemma_stress_test"}:
    os.chdir(Path(__file__).resolve().parents[2])

DATA = Path("data_parquet/gemma_3_predictions.parquet")
FIG = Path("figures/moral_machines/gemma_stress_test")
FIG.mkdir(parents=True, exist_ok=True)
FAMILY = "Gemma 3"
MODELS = ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]

POS = "original_answer_first"
ANS, MODEL = "original_reference_response_answer", "original_reference_response_model_info_model"
CF_ANS, CF_MODEL = "counterfactual_reference_response_answer", "counterfactual_reference_response_model_info_model"
PRED = "counterfactual_predictor_response_{}_explanation_predictor_{}"
CONDITIONS = ["with", "without"]
POSITIONS = [False, True]
ANSWERS = ["CASE 1", "CASE 2"]
INVALID = "invalid"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY, TEAL = "#2a78d6", "#eb6834", "#8a8984", "#2a9d8f"
ANSWER_COLORS = {"CASE 1": BLUE, "CASE 2": ORANGE, INVALID: GREY}
POS_COLORS = {False: GREY, True: TEAL}
COND_COLORS = {"with": BLUE, "without": GREY}
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False})


def short(name):
    """'google/gemma-3-27b-it' -> 'gemma-3-27b'; 'google/gemini-3-flash-preview' -> 'gemini-3-flash'."""
    return name.split("/")[-1].removesuffix("-it").removesuffix("-preview")


def legend_right(axes, ax):
    """One legend for all panels (deduplicated by label), placed right of ax."""
    handles = {}
    for a in axes:
        for h, lab in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    ax.legend(handles.values(), handles.keys(), loc="upper left", bbox_to_anchor=(1.0, 1.0))


def save(fig, fname):
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG / fname}")


# %%
cols = ["original_dataset", POS, ANS, MODEL, CF_ANS, CF_MODEL]
cols += [PRED.format(c, f) for c in CONDITIONS for f in ["names", "answers"]]
df = pd.read_parquet(DATA, columns=cols)
mm = df[df["original_dataset"].str.contains("moral_machines", na=False)].copy()
models = [m for m in MODELS if m in set(mm[MODEL])] + sorted(set(mm[MODEL]) - set(MODELS))
print(f"moral_machines rows: {len(mm)} of {len(df)}")


# %% 1. original_answer_first counts, overall and per model
def plot_answer_first_counts(data, fname="answer_first_counts.png"):
    counts = pd.crosstab(data[MODEL], data[POS]).reindex(index=models, columns=POSITIONS, fill_value=0)
    counts.loc["All models"] = counts.sum()
    print("\noriginal_answer_first counts:\n", counts)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    for k, pos in enumerate(POSITIONS):
        xs = [i + (k - 0.5) * w for i in range(len(counts))]
        bars = ax.bar(xs, counts[pos], w, color=POS_COLORS[pos], label=f"answer_first = {pos}")
        ax.bar_label(bars, fontsize=9, color=MUTED, padding=2)
    ax.set_xticks(range(len(counts)), [short(r) for r in counts.index])
    ax.set_ylabel("Scenarios")
    ax.set_title(f"original_answer_first: Moral Machines, {FAMILY}")
    ax.legend(loc="upper left")
    save(fig, fname)


plot_answer_first_counts(mm)


# %% 2. reference answer by original_answer_first, pooled and per model
def answer_shares(sub, ans_col):
    """Rows: answer_first False/True. Columns: count and share of each answer (missing answers count as invalid)."""
    counts = pd.crosstab(sub[POS], sub[ans_col].fillna(INVALID)).reindex(index=POSITIONS, fill_value=0)
    counts = counts.reindex(columns=ANSWERS + ([INVALID] if INVALID in counts else []), fill_value=0)
    return counts, counts.div(counts.sum(axis=1).where(lambda s: s > 0), axis=0)


def stacked(ax, sub, ans_col, title):
    """Stacked 100% bars of answer shares, one bar per answer_first value, labelled with share and count."""
    counts, shares = answer_shares(sub, ans_col)
    labels = [f"{pos}\n(n={int(counts.loc[pos].sum())})" for pos in POSITIONS]
    bottom = pd.Series(0.0, index=POSITIONS)
    for ans in counts.columns:
        vals = shares[ans].fillna(0)
        bars = ax.bar(labels, vals, bottom=bottom, color=ANSWER_COLORS[ans], width=0.6, label=ans)
        for b, v, n in zip(bars, vals, counts[ans]):
            if v >= 0.06:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2, f"{v:.0%}\n({n})", ha="center",
                        va="center", fontsize=8.5, color="white")
        bottom += vals
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("original_answer_first")
    ax.set_title(title)


def plot_answer_by_position(data, ans_col, model_col, fname, what):
    """Stacked answer shares by original_answer_first: all models pooled, then one panel per model in model_col."""
    print(f"\n{ans_col} by original_answer_first, all models:\n", answer_shares(data, ans_col)[0])
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.2 * (len(models) + 1), 4.8), sharey=True)
    stacked(axes[0], data, ans_col, "All models")
    for ax, m in zip(axes[1:], models):
        sub = data[data[model_col] == m]
        print(f"\n{m}:\n", answer_shares(sub, ans_col)[0])
        stacked(ax, sub, ans_col, short(m))
    axes[0].set_ylabel(f"Share of {what} answers")
    legend_right(axes, axes[-1])
    fig.suptitle(f"{what.capitalize()} answer by answer position: Moral Machines, {FAMILY}", fontweight="bold")
    save(fig, fname)


plot_answer_by_position(mm, ANS, MODEL, "answer_by_position.png", "reference")

# %% 3. counterfactual reference answer by original_answer_first, pooled and per model
plot_answer_by_position(mm, CF_ANS, CF_MODEL, "cf_answer_by_position.png", "counterfactual reference")


# %% 4. predictor answers with / without explanation by original_answer_first, pooled and per predictor
def predictor_long(data):
    """One row per (record, condition, predictor) with columns POS, CF_ANS, predictor, answer, condition."""
    parts = []
    for cond in CONDITIONS:
        part = data[[POS, CF_ANS, PRED.format(cond, "names"), PRED.format(cond, "answers")]].copy()
        part.columns = [POS, CF_ANS, "predictor", "answer"]
        parts.append(part.explode(["predictor", "answer"]).assign(condition=cond))
    return pd.concat(parts, ignore_index=True)


def plot_predictor_answers_by_position(data, fname="predictor_answer_by_position.png"):
    """Rows: with / without explanation. Columns: all predictors pooled, then one panel per predictor."""
    long = predictor_long(data)
    predictors = list(dict.fromkeys(long["predictor"]))
    fig, axes = plt.subplots(len(CONDITIONS), len(predictors) + 1, figsize=(3.2 * (len(predictors) + 1), 9.6),
                             sharey=True)
    for row, cond in zip(axes, CONDITIONS):
        sub = long[long["condition"] == cond]
        print(f"\npredictor answer {cond} explanation by original_answer_first, all predictors:\n",
              answer_shares(sub, "answer")[0])
        stacked(row[0], sub, "answer", "All predictors")
        for ax, pred in zip(row[1:], predictors):
            psub = sub[sub["predictor"] == pred]
            print(f"\n{pred}, {cond} explanation:\n", answer_shares(psub, "answer")[0])
            stacked(ax, psub, "answer", short(pred))
        row[0].set_ylabel(f"{cond.upper()} EXPLANATION\nShare of predictor answers")
    legend_right(axes.flat, axes[0, -1])
    fig.suptitle(f"Predictor answer on counterfactual by answer position: Moral Machines, {FAMILY} references",
                 fontweight="bold")
    save(fig, fname)


plot_predictor_answers_by_position(mm)


# %% 5. predictor accuracy (matches the counterfactual reference answer) with / without explanation
def plot_predictor_accuracy_by_position(data, fname="predictor_accuracy_by_position.png"):
    """One panel per original_answer_first value; grouped bars per predictor (plus all pooled), with vs without."""
    long = predictor_long(data[data[CF_ANS].notna()])
    long["correct"] = long["answer"] == long[CF_ANS]
    order = ["All predictors"] + list(dict.fromkeys(long["predictor"]))
    acc = (pd.concat([long.assign(predictor="All predictors"), long])
           .groupby([POS, "predictor", "condition"])["correct"].agg(acc="mean", n="size"))
    acc["ci"] = 1.96 * (acc["acc"] * (1 - acc["acc"]) / acc["n"]) ** 0.5
    print("\npredictor accuracy vs counterfactual reference answer:\n",
          acc["acc"].unstack("condition").round(3).assign(gain=lambda t: (t["with"] - t["without"]).round(3)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    w = 0.38
    for ax, pos in zip(axes, POSITIONS):
        for k, cond in enumerate(CONDITIONS):
            t = acc.loc[pos].xs(cond, level="condition").reindex(order)
            xs = [i + (k - 0.5) * w for i in range(len(order))]
            bars = ax.bar(xs, t["acc"], w, yerr=t["ci"], color=COND_COLORS[cond], label=f"{cond} explanation",
                          error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in t["acc"]], fontsize=8, color=MUTED, padding=6)
        n = int(acc.loc[(pos, order[1], CONDITIONS[0]), "n"])
        ax.set_xticks(range(len(order)), [short(p) for p in order], rotation=30, ha="right")
        ax.set_title(f"answer_first = {pos} (n={n} each)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Accuracy vs counterfactual reference")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Predictor accuracy on counterfactuals by answer position: Moral Machines, {FAMILY} references",
                 fontweight="bold")
    save(fig, fname)


plot_predictor_accuracy_by_position(mm)
