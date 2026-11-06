# %% [markdown]
# # Slide figures for the 10/28 update (debrief/10_28_slides_README.md)
#
# Run from the repo root: `python analysis/mm18_slides_1028_figs.py`. No API calls; reads result CSVs in
# experiments/moral_machines/ and the paper's stored predictions (via mm18.load_mm). Writes PNGs to debrief/10_28_figures/
# and prints every number quoted in the README.
#   Slide 1 (fitted rule):        s1_rule_vs_simulators.png  (+ reuses debrief/10_27_figures/fig1_what_drives.png)
#   Slide 2 (Gemma stress test):  s2a_gemma_baselines.png, s2b_gemma_full_data.png, s2c_gemma_format.png
#   Slide 3 (LLM ground truth):   s3_llm_ground_truth.png
#   Slide 4 (hand labels):        s4_hand_labels.png

# %%
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18 import P, load_mm  # noqa: E402

IN, OUT = Path("experiments/moral_machines"), Path("debrief/10_28_figures")
OUT.mkdir(parents=True, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, GREY, DARK = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c", "#3d3c39"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "legend.frameon": False})
PCT = matplotlib.ticker.PercentFormatter(1.0)
N_BOOT, SEED = 2000, 0
rng = np.random.default_rng(SEED)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pct(v):
    return f"{int(np.floor(v * 100 + 0.5 + 1e-9))}%"


def boot_ci(per_pair):
    """95% CI of the mean of per-pair values (pair bootstrap)."""
    a = np.asarray(per_pair, float)
    b = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def bar_group(ax, x, vals, los, his, colors, width, labels_inside=False):
    ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=2, zorder=2)
    if los is not None:
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none",
                    ecolor=MUTED, elinewidth=1.1, zorder=3)
    for xi, v, h in zip(x, vals, his if his is not None else vals):
        ax.text(xi, h + 0.012, pct(v), ha="center", va="bottom", fontsize=10.5, color=INK)


# %% Slide 1: fitted rule (out-of-fold) vs simulators, on the paper's counterfactuals, 9 models
ce = pd.read_csv(IN / "mm18_rule_ceiling.csv")
rp = pd.read_csv(IN / "mm18_rule_pairs.csv")
rp["cf_case1"] = ((rp["rule_p_case1"] > 0.5) == rp["rule_right"]).astype(int)
maj = rp.groupby("model")["cf_case1"].mean()
ce["cf_acc_majority"] = ce["model"].map(np.maximum(maj, 1 - maj))
order = ["GPT-5.2", "Gemini 3 Flash", "Gemini 3 Pro", "Claude Opus 4.5", "Claude Sonnet 4.5", "GPT-5 Mini",
         "Claude Haiku 4.5", "GPT-5 Nano", "Gemma 3 27B"]
ce = ce.set_index("model").reindex(order)
series = [("cf_acc_majority", "always the model's majority answer", GREY),
          ("cf_acc_predictors_without", "simulators, no explanation", DARK),
          ("cf_acc_predictors_with", "simulators, model's own explanation", C1),
          ("cf_acc_rule", "fitted rule (5-fold, out of sample)", C3)]
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(order))
w = 0.2
for k, (col, lab, color) in enumerate(series):
    ax.bar(x + (k - 1.5) * w, ce[col], width=w, color=color, edgecolor="white", linewidth=1.5, label=lab, zorder=2)
for xi, v in zip(x + 1.5 * w, ce["cf_acc_rule"]):
    ax.text(xi, v + 0.01, pct(v), ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in order], fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylabel("accuracy on the counterfactual answer")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=11)
ax.set_title("A 12-feature logistic rule predicts every model's counterfactual answers better than simulators do",
             loc="left", fontsize=15)
save(fig, "s1_rule_vs_simulators.png")
print("Slide 1 table:\n", ce[["pairs", "decisions", "acc_all_decisions_rule", "acc_all_decisions_majority",
                              "acc_all_decisions_position_only", "cf_acc_majority", "cf_acc_predictors_without",
                              "cf_acc_predictors_with", "cf_acc_rule", "nsg_explanation", "nsg_rule"]].round(3).to_string())
co = pd.read_csv(IN / "mm18_rule_coefs.csv").query("feature == 'answer_first'").set_index("model")
print("answer_first coef:\n", co[["coef", "lo", "hi"]].round(2).to_string())

# %% Slide 2a: Gemma, Exp 1 held-out pairs: trivial baselines vs simulator conditions, by answer format
pr = pd.read_csv(IN / "mm18_gt_predictions.csv").query("run == 'exp1' and model == 'Gemma 3 27B'").copy()
pr["ok"] = (pr["answer"] == pr["reference_answer"]).astype(float)
d = load_mm()
d["pair_id"] = d["original_question_idx"].astype(str) + "-" + d["counterfactual_question_idx"].astype(str)
gem = d[d["model"] == "Gemma 3 27B"].set_index("pair_id")
held = gem.loc[pr["pair_id"].unique()].copy()
rule = pd.read_csv(IN / "mm18_gt_heldout_rule.csv").query("model == 'Gemma 3 27B'").set_index("pair_id")
per_pair = pr.pivot_table(index="pair_id", columns="condition", values="ok", aggfunc="mean")  # mean over 2 simulators
per_pair["always_case1"] = (held["counterfactual_reference_response_answer"] == "CASE 1").astype(float)
per_pair["rule"] = (rule["rule_answer"] == rule["reference_answer"]).astype(float)
per_pair["fmt"] = held["answer_format"]
cols2 = [("always_case1", "always say Case 1", GREY), ("none", "simulator: no explanation", DARK),
         ("own", "simulator: Gemma's own explanation", C2), ("gt", "simulator: LLM-written description", C3),
         ("rule", "fitted rule (training split)", C4)]
panels = [("all", "All 100 held-out pairs", per_pair),
          ("answer first", "Answer asked first (39 pairs)", per_pair[per_pair["fmt"] == "answer first"]),
          ("answer last", "Answer asked last (61 pairs)", per_pair[per_pair["fmt"] == "answer last"])]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
s2a = {}
for ax, (key, title, t) in zip(axs, panels):
    vals, los, his = [], [], []
    for c, _, _ in cols2:
        lo, hi = boot_ci(t[c])
        vals.append(t[c].mean()), los.append(lo), his.append(hi)
    s2a[key] = dict(zip([c for c, _, _ in cols2], vals))
    bar_group(ax, np.arange(len(cols2)), vals, los, his, [c for _, _, c in cols2], 0.8)
    ax.set_xticks([])
    ax.set_title(title, loc="left", fontsize=13)
    ax.grid(axis="x", visible=False)
axs[0].set_ylim(0.4, 1.05)
axs[0].yaxis.set_major_formatter(PCT)
axs[0].set_ylabel("accuracy on Gemma's counterfactual answer")
fig.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for _, l, c in cols2], loc="lower center",
           bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=11)
fig.suptitle("Gemma 3 27B: the 92% description is no better than always guessing Case 1", x=0.01, ha="left",
             fontsize=15, fontweight="bold", y=1.02)
save(fig, "s2a_gemma_baselines.png")
print("\nSlide 2a:", pd.DataFrame(s2a).round(3).to_string())
print("held-out answer-last share", (held["answer_format"] == "answer last").mean(),
      "full-data answer-last share", (gem["answer_format"] == "answer last").mean().round(3))
print("held-out cf Case1 by fmt", held.groupby("answer_format")["counterfactual_reference_response_answer"]
      .apply(lambda s: (s == "CASE 1").mean()).round(3).to_dict(),
      "| full data", gem.groupby("answer_format")["counterfactual_reference_response_answer"]
      .apply(lambda s: (s == "CASE 1").mean()).round(3).to_dict())
for a, b in [("gt", "always_case1"), ("own", "none"), ("own", "always_case1"), ("gt", "own")]:
    diff = per_pair[a] - per_pair[b]
    lo, hi = boot_ci(diff)
    print(f"  {a} - {b}: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
by_sim = pr.assign(sim=pr["simulator"].str.split("/").str[-1]).pivot_table(index="condition", columns="sim", values="ok")
print(by_sim.round(3).to_string())

# %% Slide 2b: Gemma on the paper's full data, own explanation vs none, per simulator (all 999 pairs)
rows = []
for _, r in gem.iterrows():
    for nm, aw, awo in zip(r[P + "with_explanation_predictor_names"], r[P + "with_explanation_predictor_answers"],
                           r[P + "without_explanation_predictor_answers"]):
        t = r["counterfactual_reference_response_answer"]
        rows.append(dict(sim=nm.split("/")[-1], pair=r.name, w=np.nan if aw is None else float(aw == t),
                         wo=np.nan if awo is None else float(awo == t)))
fd = pd.DataFrame(rows)
summ = fd.groupby("sim")[["w", "wo"]].mean()
summ["nsg"] = (summ["w"] - summ["wo"]) / (1 - summ["wo"])
summ["n"] = fd.groupby("sim").size()
cis = {s: boot_ci((g["w"] - g["wo"]).dropna()) for s, g in fd.groupby("sim")}
summ = summ.sort_values("nsg")
fig, ax = plt.subplots(figsize=(11, 4.6))
y = np.arange(len(summ))
for yi, (s, r) in zip(y, summ.iterrows()):
    color = C2 if r["w"] < r["wo"] else C3
    ax.plot([r["wo"], r["w"]], [yi, yi], color=color, lw=3, zorder=2)
    ax.scatter([r["wo"]], [yi], s=70, color=DARK, zorder=3)
    ax.scatter([r["w"]], [yi], s=70, color=color, zorder=3)
    ax.text(0.905, yi, f"NSG {r['nsg']:+.0%}", va="center", fontsize=11.5, color=color)
always1 = (gem["counterfactual_reference_response_answer"] == "CASE 1").mean()
ax.axvline(always1, color=GREY, ls="--", lw=1.3)
ax.text(always1 + 0.003, -0.55, f"always Case 1 ({pct(always1)})", color=MUTED, fontsize=10.5)
ax.set_yticks(y, summ.index)
ax.set_xlim(0.6, 0.97)
ax.set_ylim(-0.8, len(summ) - 0.5)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("simulator accuracy (● dark = no explanation, ● colour = with Gemma's explanation)")
ax.grid(axis="y", visible=False)
ax.set_title(f"Paper's data ({len(gem)} pairs): Gemma's explanation hurts 3 of 5 simulators", loc="left", fontsize=15)
save(fig, "s2b_gemma_full_data.png")
print("\nSlide 2b:\n", summ.round(3).to_string(), "\n paired acc diff CIs:", {k: np.round(v, 3).tolist() for k, v in cis.items()})
print(" full-data always-Case1:", round((gem["counterfactual_reference_response_answer"] == "CASE 1").mean(), 3),
      "copy original:", round((~gem["ref_flipped"]).mean(), 3))

# %% Slide 2c: format test, lenient vs strict parsing
fl = pd.read_csv(IN / "mm18_format_results.csv").set_index(["group", "metric"])
fs = pd.read_csv(IN / "mm18_format_results_strict.csv").set_index(["group", "metric"])
G, Cc = "Gemma 3 27B (fresh)", "Gemini 3 Flash control (fresh)"
groups = [("Gemma 3 27B\nlenient parsing", fl, G), ("Gemma 3 27B\nstrict parsing", fs, G), ("Gemini 3 Flash\n(control)", fl, Cc)]
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(groups))
for k, (fmt, color) in enumerate((("answer first", C1), ("answer last", C2))):
    t = [tab.loc[(g_, f"Case 1 rate, {fmt}")] for _, tab, g_ in groups]
    bar_group(ax, x + (k - 0.5) * 0.36, [r["value"] for r in t], [r["lo"] for r in t], [r["hi"] for r in t], color, 0.34)
for xi, (_, tab, g_) in zip(x, groups):
    r = tab.loc[(g_, "scenarios whose majority answer flips with the format")]
    ax.text(xi, 1.13, f"{pct(r['value'])} of {int(r['n'])} scenarios flip", ha="center", fontsize=11, color=MUTED)
ax.set_xticks(x, [g[0] for g in groups])
ax.set_ylim(0, 1.2)
ax.yaxis.set_major_formatter(PCT)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("share of answers that are Case 1")
ax.grid(axis="x", visible=False)
ax.legend(handles=[matplotlib.patches.Patch(color=C1, label="[ANSWER] asked first"),
                   matplotlib.patches.Patch(color=C2, label="[ANSWER] asked last")], loc="upper left",
          bbox_to_anchor=(0, -0.13), ncol=2, fontsize=11)
ax.set_title("Causal check: same scenarios, fresh runs, only the answer position changes", loc="left", fontsize=15)
save(fig, "s2c_gemma_format.png")

# %% Slide 3: Exp 1, LLM-written description vs own explanation, decision tree, rule and majority, 3 models
res = pd.read_csv(IN / "mm18_gt_results.csv").query("simulator == 'both simulators'")
hr = pd.read_csv(IN / "mm18_gt_heldout_rule.csv")
models = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
conds = [("none", "no explanation", DARK), ("own", "model's own explanation", C1),
         ("gt", "LLM-written description", C3), ("tree", "auto description (decision tree)", C4)]
fig, ax = plt.subplots(figsize=(14, 5.2))
x = np.arange(len(models))
w = 0.19
for k, (cond, lab, color) in enumerate(conds):
    t = res[res["condition"] == cond].set_index("model").reindex(models)
    off = (k - 1.5) * w
    ax.bar(x + off, t["acc"], width=w, color=color, edgecolor="white", linewidth=1.5, label=lab, zorder=2)
    ax.errorbar(x + off, t["acc"], yerr=[t["acc"] - t["acc_lo"], t["acc_hi"] - t["acc"]], fmt="none", ecolor=MUTED,
                elinewidth=1.1, zorder=3)
    for xi, v in zip(x + off, t["acc"]):
        ax.text(xi, 0.415, pct(v), ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
s3 = {}
for xi, m in zip(x, models):
    g = hr[hr["model"] == m]
    c1 = (g["reference_answer"] == "CASE 1").mean()
    s3[m] = dict(majority=max(c1, 1 - c1), rule=(g["rule_answer"] == g["reference_answer"]).mean())
    for key, ls, lab in (("majority", "--", "always majority answer"), ("rule", "-", "fitted rule")):
        ax.hlines(s3[m][key], xi - 2 * w, xi + 2 * w, colors=INK if key == "rule" else MUTED, linestyles=ls, lw=1.8, zorder=4)
        ax.text(xi + 2 * w + 0.015, s3[m][key], pct(s3[m][key]), va="center", fontsize=10, color=INK if key == "rule" else MUTED)
ax.set_xticks(x, models)
ax.set_xlim(-0.5, len(models) - 0.35)
ax.set_ylim(0.4, 1.02)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylabel("simulator accuracy, 100 held-out pairs")
ax.grid(axis="x", visible=False)
h, _ = ax.get_legend_handles_labels()
h += [matplotlib.lines.Line2D([], [], color=INK, lw=1.8, label="fitted rule (training split)"),
      matplotlib.lines.Line2D([], [], color=MUTED, lw=1.8, ls="--", label="always the majority answer")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=11)
ax.set_title("An LLM-written description of behaviour loses to the rule it was written from, and to frontier models' own explanations",
             loc="left", fontsize=14)
save(fig, "s3_llm_ground_truth.png")
print("\nSlide 3 baselines on held-out:", {m: {k: round(v, 3) for k, v in s.items()} for m, s in s3.items()})
print(res.pivot(index="condition", columns="model", values="acc").round(3).to_string())

# %% Slide 4: GPT-5.2 hand labels: what separates failures from successes, and how the 20 failures fail
cl = pd.read_csv(IN / "mm18_coverage_labels_scored.csv").query("model == 'GPT-5.2'")
f_, s_ = cl[cl["outcome"] == "failure"], cl[cl["outcome"] == "success"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
rows4 = [("Explanation cites the\nchanged factor", f_["in_scope"].mean(), s_["in_scope"].mean()),
         ("Near-tie under the\nfitted rule", f_["near_tie"].mean(), s_["near_tie"].mean()),
         ("Explanation, read literally,\npoints the wrong way", (f_["literal_reading"] == "wrong").mean(),
          (s_["literal_reading"] == "wrong").mean())]
x = np.arange(len(rows4))
bar_group(a1, x - 0.2, [r[1] for r in rows4], None, None, C2, 0.38)
bar_group(a1, x + 0.2, [r[2] for r in rows4], None, None, C1, 0.38)
a1.set_xticks(x, [r[0] for r in rows4], fontsize=11)
a1.set_ylim(0, 1.12)
a1.yaxis.set_major_formatter(PCT)
a1.grid(axis="x", visible=False)
a1.legend(handles=[matplotlib.patches.Patch(color=C2, label=f"failures ({len(f_)})"),
                   matplotlib.patches.Patch(color=C1, label=f"successes ({len(s_)})")], loc="upper right", fontsize=11)
a1.set_title("Citing the changed factor doesn't separate them", loc="left", fontsize=13)
short = {"explanation points the wrong way: stated rule overridden": "GPT-5.2 broke its own stated rule",
         "underdetermined: explanation doesn't settle the counterfactual": "Explanation's rule doesn't apply\n(its condition isn't met)",
         "explanation settles it correctly: predictors misread": "Explanation was right;\nsimulators misread it"}
m = f_["mode"].map(short).value_counts().reindex(list(short.values()))
a2.barh(m.index[::-1], m.values[::-1], color=[GREY, C4, C2], edgecolor="white", linewidth=2, zorder=2)
for yi, v in enumerate(m.values[::-1]):
    a2.text(v + 0.15, yi, f"{v} of {len(f_)}", va="center", fontsize=12, color=INK)
a2.set_xlim(0, 11)
a2.grid(axis="y", visible=False)
a2.set_xlabel("failures (hand-labelled)")
a2.set_title("How the 20 failures fail", loc="left", fontsize=13)
fig.suptitle("GPT-5.2 hand labels (30 pairs): failures come from conditional rules, not missing factors", x=0.01,
             ha="left", fontsize=15, fontweight="bold", y=1.03)
fig.subplots_adjust(wspace=0.45)
save(fig, "s4_hand_labels.png")
print("\nSlide 4:", [(r[0].replace(chr(10), " "), round(r[1], 2), round(r[2], 2)) for r in rows4], m.to_dict(),
      "egregious failures", int(f_["egregious"].sum()))
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
