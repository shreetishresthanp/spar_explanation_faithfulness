# %% [markdown]
# # Qwen 3 on Moral Machines: answer position and reference answers
#
# Run from the repo root: `python analysis/gemma_stress_test/answer_first_dist_qwen.py`. No API calls.
# Reads data_parquet/qwen_3_predictions_latest.parquet, keeps moral_machines rows (one row per (model, scenario)), and plots
#   1. how many scenarios have original_answer_first True vs False, overall and per Qwen model;
#   2. the reference answer (CASE 1 / CASE 2) split by original_answer_first, pooled over models and per model;
#   3. the same for the counterfactual reference answer (missing answers shown as "invalid"), per counterfactual model.
#   4. the predictors' counterfactual answers with and without the explanation, per predictor. The scalar
#      counterfactual_predictor_response_*_answer / *_model_info_model columns only hold the first predictor
#      (gpt-oss-20b), so all predictors are read from the *_predictor_names / *_predictor_answers lists.
#   5. each predictor's accuracy (its answer equals counterfactual_reference_response_answer) with and without the
#      explanation, by original_answer_first. Rows with an invalid counterfactual reference answer are dropped; an
#      invalid predictor answer counts as wrong. Error bars: 95% normal-approximation binomial intervals.
# Writes figures/moral_machines/qwen_stress_test/{answer_first_counts,answer_by_position,cf_answer_by_position,
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

FIG = Path("figures/moral_machines/qwen_stress_test")
FIG.mkdir(parents=True, exist_ok=True)
POS, ANS, MODEL = "original_answer_first", "original_reference_response_answer", "original_reference_response_model_info_model"
CF_ANS, CF_MODEL = "counterfactual_reference_response_answer", "counterfactual_reference_response_model_info_model"
PRED = "counterfactual_predictor_response_{}_explanation_predictor_{}"
CONDITIONS = ["with", "without"]
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
ANSWERS = ["CASE 1", "CASE 2"]
INVALID = "invalid"
ANSWER_COLORS = {"CASE 1": "#2a78d6", "CASE 2": "#eb6834", INVALID: "#8a8984"}
POS_COLORS = {False: "#8a8984", True: "#2a9d8f"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False})

# %%
df = pd.read_parquet("data_parquet/qwen_3_predictions_latest.parquet", columns=["original_dataset", POS, ANS, MODEL, CF_ANS, CF_MODEL]
                     + [PRED.format(c, f) for c in CONDITIONS for f in ["names", "answers"]])
mm = df[df["original_dataset"].str.contains("moral_machines", na=False)].copy()
models = [m for m in MODELS if m in set(mm[MODEL])] + sorted(set(mm[MODEL]) - set(MODELS))
short = {m: m.split("/")[-1].removesuffix("-it") for m in models}
print(f"moral_machines rows: {len(mm)} of {len(df)}")

# %% 1. original_answer_first counts, overall and per model
pos_counts = pd.crosstab(mm[MODEL], mm[POS]).reindex(index=models, columns=[False, True], fill_value=0)
pos_counts.loc["All models"] = pos_counts.sum()
print("\noriginal_answer_first counts:\n", pos_counts)

fig, ax = plt.subplots(figsize=(8, 4.5))
rows = list(pos_counts.index)
x = range(len(rows))
w = 0.38
for k, pos in enumerate([False, True]):
    vals = pos_counts[pos].values
    bars = ax.bar([i + (k - 0.5) * w for i in x], vals, w, color=POS_COLORS[pos], label=f"answer_first = {pos}")
    ax.bar_label(bars, fontsize=9, color=MUTED, padding=2)
ax.set_xticks(list(x), [short.get(r, r) for r in rows])
ax.set_ylabel("Scenarios")
ax.set_title("original_answer_first: Moral Machines, Qwen 3")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "answer_first_counts.png")
plt.close(fig)


# %% 2. reference answer by original_answer_first, pooled and per model
def answer_shares(sub, ans_col):
    """Rows: answer_first False/True. Columns: count and share of each answer (missing answers count as invalid)."""
    counts = pd.crosstab(sub[POS], sub[ans_col].fillna(INVALID)).reindex(index=[False, True], fill_value=0)
    counts = counts.reindex(columns=ANSWERS + ([INVALID] if INVALID in counts else []), fill_value=0)
    return counts, counts.div(counts.sum(axis=1).where(lambda s: s > 0), axis=0)


def stacked(ax, sub, ans_col, title):
    counts, shares = answer_shares(sub, ans_col)
    labels = [f"{pos}\n(n={int(counts.loc[pos].sum())})" for pos in [False, True]]
    bottom = pd.Series(0.0, index=[False, True])
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
    for m in models:
        print(f"\n{m}:\n", answer_shares(data[data[model_col] == m], ans_col)[0])

    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.2 * (len(models) + 1), 4.8), sharey=True)
    stacked(axes[0], data, ans_col, "All models")
    for ax, m in zip(axes[1:], models):
        stacked(ax, data[data[model_col] == m], ans_col, short[m])
    axes[0].set_ylabel(f"Share of {what} answers")
    handles, labels = {}, {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axes[-1].legend(handles.values(), handles.keys(), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{what.capitalize()} answer by answer position: Moral Machines, Qwen 3", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG}/{fname}")


plot_answer_by_position(mm, ANS, MODEL, "answer_by_position.png", "reference")

# %% 3. counterfactual reference answer by original_answer_first, pooled and per model
plot_answer_by_position(mm, CF_ANS, CF_MODEL, "cf_answer_by_position.png", "counterfactual reference")


# %% 4. predictor answers with / without explanation by original_answer_first, pooled and per predictor
def predictor_long(data):
    """One row per (record, condition, predictor): original_answer_first, CF reference answer, condition, predictor,
    answer."""
    parts = []
    for cond in CONDITIONS:
        part = data[[POS, CF_ANS, PRED.format(cond, "names"), PRED.format(cond, "answers")]].copy()
        part.columns = [POS, CF_ANS, "predictor", "answer"]
        part = part.explode(["predictor", "answer"])
        part["condition"] = cond
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def short_pred(name):
    return name.split("/")[-1].removesuffix("-it").removesuffix("-preview")


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
            print(f"\n{pred}, {cond} explanation:\n", answer_shares(sub[sub["predictor"] == pred], "answer")[0])
            stacked(ax, sub[sub["predictor"] == pred], "answer", short_pred(pred))
        row[0].set_ylabel(f"{cond.upper()} EXPLANATION\nShare of predictor answers")
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axes[0, -1].legend(handles.values(), handles.keys(), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Predictor answer on counterfactual by answer position: Moral Machines, Qwen 3 references",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG}/{fname}")


plot_predictor_answers_by_position(mm)


# %% 5. predictor accuracy (matches the counterfactual reference answer) with / without explanation
def plot_predictor_accuracy_by_position(data, fname="predictor_accuracy_by_position.png"):
    """One panel per original_answer_first value; grouped bars per predictor (plus all pooled), with vs without."""
    long = predictor_long(data[data[CF_ANS].notna()])
    long["correct"] = long["answer"] == long[CF_ANS]
    predictors = list(dict.fromkeys(long["predictor"]))
    pooled = long.assign(predictor="All predictors")
    acc = (pd.concat([pooled, long]).groupby([POS, "predictor", "condition"])["correct"].agg(["mean", "size"])
           .rename(columns={"mean": "acc", "size": "n"}))
    acc["ci"] = 1.96 * (acc["acc"] * (1 - acc["acc"]) / acc["n"]) ** 0.5
    print("\npredictor accuracy vs counterfactual reference answer:\n",
          acc["acc"].unstack("condition").round(3).assign(gain=lambda t: (t["with"] - t["without"]).round(3)))

    order = ["All predictors"] + predictors
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    w = 0.38
    for ax, pos in zip(axes, [False, True]):
        for k, (cond, color) in enumerate(zip(CONDITIONS, ["#2a78d6", "#8a8984"])):
            t = acc.loc[pos].xs(cond, level="condition").reindex(order)
            xs = [i + (k - 0.5) * w for i in range(len(order))]
            bars = ax.bar(xs, t["acc"], w, yerr=t["ci"], color=color, label=f"{cond} explanation",
                          error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in t["acc"]], fontsize=8, color=MUTED, padding=6)
        n = int(acc.loc[(pos, order[1], CONDITIONS[0]), "n"])
        ax.set_xticks(range(len(order)), [short_pred(p) for p in order], rotation=30, ha="right")
        ax.set_title(f"answer_first = {pos} (n={n} each)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Accuracy vs counterfactual reference")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Predictor accuracy on counterfactuals by answer position: Moral Machines, Qwen 3 references",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG}/{fname}")


plot_predictor_accuracy_by_position(mm)
print(f"\nWrote {FIG}/answer_first_counts.png")
