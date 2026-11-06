# %% [markdown]
# # Gemma's Case 1 rate by answer position, next to other open-weight models on the same scenarios (paper's data)
#
# Run from the repo root: `python analysis/mm18_gemma_format_fig.py`. No API calls. Uses the paper's stored reference
# answers (data_parquet/, pair set B) as parsed by the paper, one row per (model, scenario), both sides of each pair.
# Only scenarios Gemma 3 27B also answered are kept, so every model faces the same questions.
# Writes figures/moral_machines/gemma_format_vs_models.png and experiments/moral_machines/mm18_gemma_format_vs_models.csv.
# Also figures/moral_machines/gemma_format_fresh.png: the fresh runs (analysis/mm18_format_run.py), the same 100 scenarios
# asked both ways, Gemma 3 27B (3 samples per format) vs Gemini 3 Flash (1 sample), read from mm18_format_results.csv
# (default parsing of analysis/mm18_format.py: every reply counts, tagged or not).
# Intervals: 95% bootstrap over scenarios (2,000 replicates).

# %%
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18 import CONFIG_TO_MODEL, FILES, load_mm  # noqa: E402

OUT, FIG = Path("experiments/moral_machines"), Path("figures/moral_machines")
FIG.mkdir(parents=True, exist_ok=True)
SHOW = ["Gemma 3 27B", "Qwen 3 32B", "Qwen 3 14B", "Qwen 3 8B"]
C1, C2 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "legend.frameon": False})
N_BOOT, SEED = 2000, 0
rng = np.random.default_rng(SEED)

# %% One row per (model, scenario): the answer and whether [ANSWER] was requested before [EXPLANATION]
d = load_mm()
cols = ["original_dataset", "original_question", "counterfactual_question", "original_reference_response_model_info_model",
        "original_reference_response_model_info_thinking", "counterfactual_question_prompt"]
t = pd.concat([pq.read_table(f"data_parquet/{f}.parquet", columns=cols, filters=[("original_dataset", "=", "moral_machines")])
               .to_pandas() for fs in FILES.values() for f in fs], ignore_index=True)
t["model"] = (t["original_reference_response_model_info_model"].str.split("/").str[-1] + " ("
              + t["original_reference_response_model_info_thinking"].astype(str) + ")").map(CONFIG_TO_MODEL)
t["c_answer_first"] = t["counterfactual_question_prompt"].map(lambda p: p.rfind("[ANSWER]") < p.rfind("[EXPLANATION]"))
keys = ["model", "original_question", "counterfactual_question"]
d = d.merge(t.dropna(subset=["model"]).drop_duplicates(keys)[keys + ["c_answer_first"]], on=keys, how="left")
sides = [d[["model", q, f, a]].set_axis(["model", "q", "answer_first", "answer"], axis=1) for q, f, a in
         [("original_question", "original_answer_first", "original_reference_response_answer"),
          ("counterfactual_question", "c_answer_first", "counterfactual_reference_response_answer")]]
s = pd.concat(sides).dropna(subset=["answer"]).drop_duplicates(["model", "q"])
s["case1"] = s["answer"].astype(str).str.contains("1")
s = s[s["model"].isin(SHOW) & s["q"].isin(set(s.loc[s["model"] == "Gemma 3 27B", "q"]))]


def ci(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), *np.percentile(rng.choice(x, (N_BOOT, len(x))).mean(axis=1), [2.5, 97.5])


rows = []
for m in SHOW:
    for first, lab in [(True, "answer first"), (False, "answer last")]:
        x = s[(s["model"] == m) & (s["answer_first"] == first)]["case1"]
        rows.append(dict(model=m, position=lab, n=len(x), **dict(zip(["rate", "lo", "hi"], ci(x)))))
res = pd.DataFrame(rows)
res.to_csv(OUT / "mm18_gemma_format_vs_models.csv", index=False)

# %% Grouped bars: two per model (answer first / last)
fig, ax = plt.subplots(figsize=(9, 4.4))
w, x0 = 0.38, np.arange(len(SHOW))
for k, (lab, col) in enumerate([("answer first", C1), ("answer last", C2)]):
    r = res[res["position"] == lab].set_index("model").loc[SHOW]
    xs = x0 + (k - 0.5) * (w + 0.02)
    ax.bar(xs, r["rate"], w, color=col, label=f"[ANSWER] asked {lab.split()[1]}")
    ax.vlines(xs, r["lo"], r["hi"], color=INK, linewidth=1.2)
    for xi, v, hi in zip(xs, r["rate"], r["hi"]):
        ax.annotate(f"{v:.0%}", (xi, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11.5,
                    color=INK, fontweight="bold" if v > 0.9 else "normal")
n = s.groupby("model")["q"].nunique()
ax.set_xticks(x0)
ax.set_xticklabels([f"{m}\n{n[m]:,} scenarios" for m in SHOW], fontsize=11.5)
ax.tick_params(axis="x", length=0)
ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("share of answers that are Case 1")
ax.grid(axis="x", visible=False)
ax.set_title("Same scenarios: only Gemma's choice depends on the answer position", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=11)
fig.savefig(FIG / "gemma_format_vs_models.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# %% Fresh runs: same 100 scenarios, both answer positions, Gemma vs Gemini 3 Flash control
fr = pd.read_csv(OUT / "mm18_format_results.csv")
FRESH = [("Gemma 3 27B (fresh)", "Gemma 3 27B\n3 samples per format"), ("Gemini 3 Flash control (fresh)", "Gemini 3 Flash\n1 sample per format")]
fig, ax = plt.subplots(figsize=(8, 4.4))
x0 = np.arange(len(FRESH))
for k, (pos, col) in enumerate([("first", C1), ("last", C2)]):
    xs = x0 + (k - 0.5) * (w + 0.02)
    for xi, (grp, _) in zip(xs, FRESH):
        r = fr[(fr["group"] == grp) & (fr["metric"] == f"Case 1 rate, answer {pos}")].iloc[0]
        ax.bar(xi, r["value"], w, color=col, label=f"[ANSWER] asked {pos}" if xi == xs[0] else None)
        ax.vlines(xi, r["lo"], r["hi"], color=INK, linewidth=1.2)
        ax.annotate(f"{r['value']:.0%}", (xi, r["hi"]), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=11.5, color=INK, fontweight="bold" if r["value"] > 0.9 else "normal")
for xi, (grp, _) in zip(x0, FRESH):
    f = fr[(fr["group"] == grp) & (fr["metric"] == "scenarios whose majority answer flips with the format")].iloc[0]
    ax.text(xi, 1.1, f"{f['value']:.0%} of scenarios flip", ha="center", fontsize=11, color=MUTED)
ax.set_xticks(x0)
ax.set_xticklabels([lab for _, lab in FRESH], fontsize=11.5)
ax.tick_params(axis="x", length=0)
ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("share of answers that are Case 1")
ax.grid(axis="x", visible=False)
ax.set_title("Fresh runs, same 100 scenarios: only the answer position changes", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=11)
fig.savefig(FIG / "gemma_format_fresh.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

if __name__ == "__main__":
    print(res.round(3).to_string(index=False))
