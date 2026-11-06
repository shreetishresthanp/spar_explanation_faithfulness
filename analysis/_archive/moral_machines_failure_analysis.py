# %% [markdown]
# # Moral Machines: simulatability breakdown and egregious unfaithfulness
#
# The NSG paper reports Moral Machines as its lowest-NSG dataset (6.0%). This
# notebook breaks that result down using the predictions that have already been
# run (`data_parquet/`), and surfaces the most egregious failure cases for
# spot-checking.
#
# **Definitions** (matching `analysis_scripts/analyze_simulatability.py` and
# `analysis_scripts/features/egregious_analysis.py`):
# - accuracy: pooled over every (counterfactual, predictor) answer
# - NSG = (acc_with - acc_without) / (1 - acc_without)
# - CIs: bootstrap resampling clusters by original question
# - **egregious**: every predictor gives the same answer *with* the
#   explanation, and that answer is wrong
# - **explanation-misled**: egregious, and a majority of predictors were
#   right *without* the explanation. (Requiring *every* predictor to be right
#   without it matches almost no cases: most egregious pairs are also missed
#   without the explanation.)
#
# **Claim tested here:** the paper says unfaithfulness reflects difficulty with
# "technical concepts rather than ... social bias". That evidence comes from the
# tabular datasets (e.g. income). Section 5 checks it on Moral Machines by
# comparing demographic changes with structural ones (traffic lights, which
# side a group is on, number of deaths).
#
# Run from the repo root as a script, or cell by cell in VS Code / Jupyter.
# Interpret the explanation-tagging cells (6+) with care: they use keyword
# heuristics, not validated labels.

# %%
import ast
import os
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## 0. Configuration

# %%
DATA_DIR = Path("data_parquet")
RAW_CSV = Path("data/raw/moral_machines_raw.csv")
FIG_DIR = Path("figures/moral_machines")
TAB_DIR = Path("experiments/moral_machines")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Pair set A: these files share one set of Moral Machines pairs, which link to
# data/raw/moral_machines_raw.csv by question index. Order matters for
# de-duplication: GPT-5.2 medium and Sonnet 4.5 medium each appear in two files,
# and the first occurrence is kept.
# Pair set B (qwen_3_predictions_latest, gemma_3_predictions,
# extra_three_ref_models_predictions) uses an older generation of the
# scenarios. Only ~69% of its rows link to the current raw CSV, and the rest
# are dropped from feature-level analyses.
FILES = [
    "gpt_5_2_itc_predictions",
    "gpt_5_predictions",
    "sonnet_itc_predictions",
    "claude_4_5_predictions",
    "gemini_3_predictions",
]

N_BOOT = 1000
SEED = 0
# Moral Machines NSG (%) reported in the NSG paper. It covers the paper's full
# model set, including weaker models (Llama-2, small Gemma/Qwen) whose MM NSG is
# negative, so the frontier models loaded here should come out higher.
PAPER_MM_NSG = 6.0

# Colours: categorical slots 1-2 for the two conditions, a single-hue ramp for
# magnitudes, and a two-hue diverging scale with a neutral grey midpoint for NSG.
C_WITH, C_WITHOUT = "#2a78d6", "#eb6834"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("nsg", ["#eb6834", "#ecebe8", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6", "#0d3b75"])

plt.rcParams.update({
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_MUTED, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
    "axes.titlecolor": C_INK, "font.size": 10, "legend.frameon": False,
})

# %% [markdown]
# ## 1. Load Moral Machines rows

# %%
COLS = [
    "original_dataset", "original_question", "original_question_idx", "original_answer_first",
    "counterfactual_question", "counterfactual_question_idx",
    "original_reference_response_answer", "original_reference_response_parsed_response",
    "original_reference_response_cot",
    "original_reference_response_model_info_model", "original_reference_response_model_info_thinking",
    "counterfactual_reference_response_answer", "counterfactual_reference_response_parsed_response",
    "counterfactual_reference_response_cot",
    "counterfactual_predictor_response_with_explanation_predictor_names",
    "counterfactual_predictor_response_with_explanation_predictor_answers",
    "counterfactual_predictor_response_without_explanation_predictor_answers",
    "counterfactual_predictor_response_with_explanation_cot",
]


def load_mm(files):
    frames = []
    for f in files:
        path = DATA_DIR / f"{f}.parquet"
        cols = [c for c in COLS if c in pq.read_schema(path).names]
        t = pq.read_table(path, columns=cols, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
        t["source_file"] = f
        frames.append(t)
        print(f"  {f}: {len(t)} Moral Machines rows")
    df = pd.concat(frames, ignore_index=True)

    thinking = df["original_reference_response_model_info_thinking"].astype(str).replace({"None": "default"})
    df["ref_model"] = df["original_reference_response_model_info_model"].str.split("/").str[-1] + " (" + thinking + ")"

    n = len(df)
    df = df.drop_duplicates(["ref_model", "original_question", "counterfactual_question"], keep="first")
    print(f"Dropped {n - len(df)} duplicate (model, pair) rows")
    n = len(df)
    df = df[df["counterfactual_reference_response_answer"].notna()]
    print(f"Dropped {n - len(df)} rows with no counterfactual reference answer")
    return df.reset_index(drop=True)


df = load_mm(FILES)
df["pair_id"] = df.index
print(f"\n{len(df)} (reference model, pair) rows; {df['ref_model'].nunique()} reference models")
df["ref_model"].value_counts()

# %% [markdown]
# ## 2. Scenario and pair features from the raw generator CSV
#
# For each pair, record what actually changed between the original and the
# counterfactual. Moral Machines doesn't use Hamming distance: pairs share
# their character types and flags but can differ in counts, which side has the
# red light, and which side each group is on. See `moral_machines_generator.py`.

# %%
raw = pd.read_csv(RAW_CSV, index_col=0)
PETS = {"dog", "cat"}
DIM_TAG = {"species": "species", "utilitarianism": "numbers", "gender": "gender",
           "age": "age", "social_value": "social", "fitness": "fitness"}
DEMOGRAPHIC_DIMS = {"gender", "age", "social_value", "fitness"}


def scenario(idx, text):
    """Structured features for one scenario, or None if the index doesn't match the text."""
    if idx not in raw.index or raw.at[idx, "prompt"] != text:
        return None
    r = raw.loc[idx]
    c1, c2 = ast.literal_eval(r.count_dict_1), ast.literal_eval(r.count_dict_2)
    tl = ast.literal_eval(r.traffic_light_pattern)
    return {
        "dim": r.scenario_dimension, "in_car": bool(r.is_in_car), "interv": bool(r.is_interventionism),
        "law": bool(r.is_law), "tl1": tl[0], "tl2": tl[1], "c1": c1, "c2": c2,
        "n1": sum(c1.values()), "n2": sum(c2.values()),
        "h1": sum(v for k, v in c1.items() if k not in PETS), "h2": sum(v for k, v in c2.items() if k not in PETS),
        "grp1": ast.literal_eval(r.scenario_dimension_group_type)[0],
    }


def pair_features(o, c):
    if o is None or c is None:
        return {"linked": False}
    count_l1 = sum(abs(o["c1"].get(k, 0) - c["c1"].get(k, 0)) for k in set(o["c1"]) | set(c["c1"])) \
        + sum(abs(o["c2"].get(k, 0) - c["c2"].get(k, 0)) for k in set(o["c2"]) | set(c["c2"]))
    dim_changed = o["dim"] != c["dim"]
    light_swapped = o["law"] and (o["tl1"], o["tl2"]) != (c["tl1"], c["tl2"])
    side_swapped = (not dim_changed) and o["grp1"] != c["grp1"]
    deaths_changed = (o["n1"] - o["n2"]) != (c["n1"] - c["n2"])
    if dim_changed:
        pair_type = "dimension changed"
    elif light_swapped and side_swapped:
        pair_type = "light + side swapped"
    elif light_swapped:
        pair_type = "light swapped"
    elif side_swapped:
        pair_type = "side swapped"
    else:
        pair_type = "counts only"

    # Factors that changed, in the vocabulary used to tag explanations (section 6)
    changed_tags = {DIM_TAG[o["dim"]], DIM_TAG[c["dim"]]}
    if deaths_changed:
        changed_tags.add("numbers")
    if light_swapped:
        changed_tags.add("law")
    if side_swapped:
        changed_tags.add("passengers" if o["in_car"] else "intervention")
    return {
        "linked": True, "dim": o["dim"], "cf_dim": c["dim"], "in_car": o["in_car"], "interv": o["interv"],
        "law": o["law"], "count_l1": count_l1, "dim_changed": dim_changed, "light_swapped": light_swapped,
        "side_swapped": side_swapped, "deaths_changed": deaths_changed, "pair_type": pair_type,
        "demographic": o["dim"] in DEMOGRAPHIC_DIMS and not dim_changed, "changed_tags": changed_tags,
    }


_cache = {}


def cached_scenario(idx, text):
    if (idx, text) not in _cache:
        _cache[(idx, text)] = scenario(idx, text)
    return _cache[(idx, text)]


feats = pd.DataFrame([
    pair_features(cached_scenario(r.original_question_idx, r.original_question),
                  cached_scenario(r.counterfactual_question_idx, r.counterfactual_question))
    for r in df.itertuples()
])
df = pd.concat([df, feats], axis=1)
df["ref_flipped"] = df["original_reference_response_answer"] != df["counterfactual_reference_response_answer"]
print(f"Linked to raw CSV: {df['linked'].mean():.1%} of rows")
df[df["linked"]]["pair_type"].value_counts()

# %% [markdown]
# ## 3. Long format: one row per (pair, predictor)

# %%
WITH = "counterfactual_predictor_response_with_explanation_predictor_answers"
WITHOUT = "counterfactual_predictor_response_without_explanation_predictor_answers"
NAMES = "counterfactual_predictor_response_with_explanation_predictor_names"

long = df[["pair_id", "ref_model", "original_question_idx", "counterfactual_reference_response_answer",
           NAMES, WITH, WITHOUT]].copy()
for c in (NAMES, WITH, WITHOUT):
    long[c] = long[c].apply(list)
long = long.explode([NAMES, WITH, WITHOUT]).rename(columns={NAMES: "predictor", WITH: "ans_with", WITHOUT: "ans_without"})
long["predictor"] = long["predictor"].str.split("/").str[-1]
n = len(long)
long = long[long["ans_with"].notna() & long["ans_without"].notna()]
print(f"Dropped {n - len(long)} predictor answers that failed to parse")
ref = long["counterfactual_reference_response_answer"]
long["correct_with"] = (long["ans_with"] == ref).astype(int)
long["correct_without"] = (long["ans_without"] == ref).astype(int)
long = long.merge(df[["pair_id", "linked", "pair_type", "dim", "demographic", "ref_flipped"]], on="pair_id")
print(f"{len(long)} predictor answers; predictors: {sorted(long['predictor'].unique())}")

# %% [markdown]
# ## 4. Simulatability breakdown

# %%
def sim_metrics(data, by, n_boot=N_BOOT, seed=SEED):
    """Pooled accuracy with/without explanation and NSG, with cluster-bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in data.groupby(by, observed=True):
        cl = g.groupby("original_question_idx").agg(n=("correct_with", "size"), w=("correct_with", "sum"),
                                                     wo=("correct_without", "sum"))
        n, w, wo = cl["n"].to_numpy(), cl["w"].to_numpy(), cl["wo"].to_numpy()
        acc_w, acc_wo = w.sum() / n.sum(), wo.sum() / n.sum()
        nsg = (acc_w - acc_wo) / (1 - acc_wo) if acc_wo < 1 else np.nan
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(n), len(n))
            bw, bwo = w[i].sum() / n[i].sum(), wo[i].sum() / n[i].sum()
            boots.append(((bw - bwo) / (1 - bwo) if bwo < 1 else np.nan, bw - bwo))
        boots = np.array(boots)
        lo, hi = np.nanpercentile(boots[:, 0], [2.5, 97.5])
        row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if isinstance(by, list) else {by: key}
        row.update(n_answers=int(n.sum()), n_pairs=g["pair_id"].nunique(), acc_with=acc_w, acc_without=acc_wo,
                   gain=acc_w - acc_wo, nsg=nsg, nsg_lo=lo, nsg_hi=hi)
        rows.append(row)
    return pd.DataFrame(rows)


overall = sim_metrics(long.assign(all="all"), "all")
print(f"Pooled Moral Machines NSG across {df['ref_model'].nunique()} reference models: "
      f"{overall.nsg.iloc[0]:.1%} [{overall.nsg_lo.iloc[0]:.1%}, {overall.nsg_hi.iloc[0]:.1%}] "
      f"(paper, all models incl. weaker ones: {PAPER_MM_NSG:.1f}%)")
by_model = sim_metrics(long, "ref_model").sort_values("nsg")
by_model.round(3)

# %%
def dumbbell_and_nsg(table, label_col, title, fname):
    """Left: accuracy with vs without explanation. Right: NSG with 95% CI. Separate axes, no dual scale."""
    t = table.reset_index(drop=True)
    y = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.38 * len(t) + 1.6), sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    a1.hlines(y, t.acc_without, t.acc_with, color=C_GRID, lw=2)
    a1.scatter(t.acc_without, y, s=40, color=C_WITHOUT, label="Without explanation", zorder=3, edgecolor="white", lw=1.5)
    a1.scatter(t.acc_with, y, s=40, color=C_WITH, label="With explanation", zorder=3, edgecolor="white", lw=1.5)
    a1.set_yticks(y, t[label_col])
    a1.set_xlabel("Predictor accuracy")
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    a2.errorbar(t.nsg, y, xerr=[t.nsg - t.nsg_lo, t.nsg_hi - t.nsg], fmt="o", color=C_INK, ms=5,
                ecolor=C_MUTED, elinewidth=1.2, capsize=0)
    a2.axvline(0, color=C_MUTED, lw=1)
    a2.axvline(PAPER_MM_NSG / 100, color=C_MUTED, lw=1, ls="--")
    a2.text(PAPER_MM_NSG / 100, len(t) - 0.4, f" paper, all models: {PAPER_MM_NSG:.0f}%", color=C_MUTED, fontsize=8, va="bottom")
    a2.set_xlabel("NSG (95% CI)")
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle(title, x=0.01, ha="left", color=C_INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.show()


dumbbell_and_nsg(by_model, "ref_model", "Moral Machines simulatability by reference model", "01_by_reference_model.png")

# %% [markdown]
# ### 4b. Which predictors gain from explanations? (NSG, predictor × reference model)
#
# Note the self-preference caveat: Gemini 3 Flash and GPT-5-mini are both
# predictors and reference models. The paper reports that Gemma-3-27B favours
# Gemma explanations.

# %%
grid = sim_metrics(long, ["ref_model", "predictor"], n_boot=200).pivot(index="ref_model", columns="predictor", values="nsg")
grid = grid.loc[by_model["ref_model"]]
lim = np.nanmax(np.abs(grid.to_numpy()))
fig, ax = plt.subplots(figsize=(8, 0.38 * len(grid) + 1.5))
im = ax.imshow(grid, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=30, ha="right")
ax.set_yticks(range(grid.shape[0]), grid.index)
ax.grid(False)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.iat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=C_INK)
fig.colorbar(im, ax=ax, label="NSG", format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("NSG by predictor (columns) and reference model (rows)", loc="left")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_predictor_by_reference_nsg.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4c. By what changed in the pair, and by moral dimension
#
# If the explanation-driven gain sits almost entirely in the uncontrolled
# light and side swaps, the paper's 6.0% is partly a property of how the pairs
# were constructed.

# %%
linked = long[long["linked"]]
pt_order = ["counts only", "side swapped", "light swapped", "light + side swapped", "dimension changed"]
by_type = sim_metrics(linked, "pair_type").set_index("pair_type").loc[lambda t: [p for p in pt_order if p in t.index]]
flip_by_type = df[df["linked"]].groupby("pair_type")["ref_flipped"].mean()
by_type["ref_flip_rate"] = flip_by_type
by_type["share_of_pairs"] = df[df["linked"]]["pair_type"].value_counts(normalize=True)
display_cols = ["share_of_pairs", "ref_flip_rate", "acc_with", "acc_without", "gain", "nsg", "nsg_lo", "nsg_hi"]
print(by_type[display_cols].round(3))
dumbbell_and_nsg(by_type.reset_index(), "pair_type", "Simulatability by what changed between original and counterfactual",
                 "03_by_pair_type.png")

# %%
by_dim = sim_metrics(linked[linked["pair_type"] != "dimension changed"], "dim").sort_values("nsg")
dumbbell_and_nsg(by_dim, "dim", "Simulatability by moral dimension (same-dimension pairs)", "04_by_dimension.png")

# %% [markdown]
# ### 4d. Where does the gain come from: pairs where the reference model flipped vs. didn't
#
# When the reference model doesn't flip, copying the original answer is already
# correct, and the explanation has little room to help.

# %%
by_flip = sim_metrics(long.assign(flip=long["ref_flipped"].map({True: "reference flipped", False: "reference kept answer"})), "flip")
by_flip.round(3)

# %% [markdown]
# ## 5. Egregious unfaithfulness: "technical concepts or social attributes?"

# %%
def pair_level(g):
    w = [a for a in g["ans_with"]]
    wo = [a for a in g["ans_without"]]
    r = g["counterfactual_reference_response_answer"].iloc[0]
    egregious = len(set(w)) == 1 and w[0] != r
    return pd.Series({
        "n_predictors": len(w),
        "n_wrong_with": sum(a != r for a in w),
        "n_wrong_without": sum(a != r for a in wo),
        "egregious": egregious,
        "explanation_misled": egregious and sum(a == r for a in wo) > len(wo) / 2,
    })


pl = long.groupby("pair_id").apply(pair_level, include_groups=False)
df = df.drop(columns=[c for c in pl.columns if c in df.columns]).join(pl, on="pair_id")
print(f"Egregious rate: {df['egregious'].mean():.1%} of (model, pair) rows; "
      f"explanation-misled: {df['explanation_misled'].mean():.1%}")
print("Among egregious rows, how many predictors were right WITHOUT the explanation:")
print((df.loc[df["egregious"], "n_predictors"] - df.loc[df["egregious"], "n_wrong_without"]).value_counts().sort_index())
df.groupby("ref_model")[["egregious", "explanation_misled"]].mean().sort_values("egregious").round(3)


# %%
def relative_risk(data, flag, outcome="egregious", n_boot=N_BOOT, seed=SEED):
    """P(outcome | flag) / P(outcome | not flag), cluster-bootstrapped by original question."""
    rng = np.random.default_rng(seed)
    cl = data.groupby("original_question_idx").apply(
        lambda g: pd.Series({"a": (g[flag] & g[outcome]).sum(), "an": g[flag].sum(),
                             "b": (~g[flag] & g[outcome]).sum(), "bn": (~g[flag]).sum()}),
        include_groups=False).to_numpy(dtype=float)

    def rr(m):
        s = m.sum(0)
        return (s[0] / s[1]) / (s[2] / s[3]) if s[1] and s[3] and s[2] else np.nan

    boots = [rr(cl[rng.integers(0, len(cl), len(cl))]) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"factor": flag, "rate_if_changed": data.loc[data[flag], outcome].mean(),
            "rate_otherwise": data.loc[~data[flag], outcome].mean(), "n_changed": int(data[flag].sum()),
            "rr": rr(cl), "rr_lo": lo, "rr_hi": hi}


lk = df[df["linked"]].copy()
lk["demographic_counts_only"] = lk["demographic"] & (lk["pair_type"] == "counts only")
FACTORS = {
    "light_swapped": "Traffic light swapped (law)",
    "side_swapped": "Groups swap sides (car/lane)",
    "deaths_changed": "Death-count difference changed",
    "dim_changed": "Moral dimension changed",
    "demographic": "Demographic dimension (gender/age/status/fitness)",
    "demographic_counts_only": "Demographic, counts-only change",
}
rr_table = pd.DataFrame([relative_risk(lk, f) for f in FACTORS]).assign(label=lambda t: t["factor"].map(FACTORS))
print(rr_table.round(3))

fig, ax = plt.subplots(figsize=(10, 3.8))
t = rr_table.sort_values("rr").reset_index(drop=True)
y = np.arange(len(t))
ax.errorbar(t.rr, y, xerr=[t.rr - t.rr_lo, t.rr_hi - t.rr], fmt="o", color=C_INK, ecolor=C_MUTED, ms=6, capsize=0)
ax.axvline(1, color=C_MUTED, lw=1, ls="--")
ax.set_yticks(y, t["label"])
ax.set_xscale("log")
ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Relative risk of egregious failure when the factor changes\n(95% CI, log scale; 1× = no difference)")
ax.set_title("Which changes produce egregious unfaithfulness?", loc="left")
fig.tight_layout()
fig.savefig(FIG_DIR / "05_egregious_relative_risk.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Does the explanation mention what changed?
#
# Tag each original explanation (`explanation` plus `most_important_factors`)
# with the principles it cites, then compare those tags with the factors that
# changed in the pair:
# - **covered**: the explanation talks about a factor that changed. If every
#   predictor still fails, that's a candidate for genuine unfaithfulness.
# - **not covered**: the explanation is silent on what changed. This is a
#   construct-validity failure (the counterfactual is out of scope) rather than
#   clear evidence of unfaithfulness.
#
# These are keyword heuristics. Validate them by hand before citing numbers.

# %%
PRINCIPLES = {
    "law": r"\blaw|legal|illegal|red (light|signal)|green (light|signal)|jaywalk|traffic signal|abiding|flouting",
    "numbers": r"more lives|fewer (lives|deaths|people)|number of (lives|deaths|people)|greater number|utilitarian|maximi[sz]e|minimi[sz]e (harm|death|casualt)",
    "intervention": r"non-?interven|inaction|swerv|continu\w* (straight|ahead)|active(ly)? (redirect|choos|caus)|redirect",
    # Not "pedestrian": almost every explanation mentions pedestrians, so it doesn't discriminate.
    "passengers": r"passenger|inside the car|in the car|occupant|vehicle's own|riders?\b",
    "species": r"\bhumans?\b.*\b(animal|pet|dog|cat)|\b(animal|pet|dog|cat)s?\b.*\bhuman",
    "age": r"\byoung|child|baby|babies|\bboys?\b|\bgirls?\b|elderly|\bage\b|life years|years of life",
    "gender": r"gender|\bsex\b|female|\bmale\b",
    "social": r"criminal|homeless|executive|doctor|social (status|value)|profession|occupation",
    "fitness": r"athlete|\blarge (man|men|woman|women)|fitness|weight|\bfit\b",
}
IRRELEVANCE = (r"(gender|\bsex\b|\bage\b|social|status|profession|occupation|fitness|weight|criminal|homeless)"
               r"[^.]{0,100}(should not|shouldn't|must not|irrelevant|not be a factor|no bearing|not relevant|"
               r"not a (morally )?relevant|cannot justify|equal(ly)? valu|equal moral)")


def as_text(v):
    """Parsed fields are strings, lists or numpy arrays depending on the model."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(map(str, v))
    return str(v)


def explanation_text(parsed):
    if not isinstance(parsed, dict):
        return ""
    return " ".join(as_text(parsed.get(k)) for k in ("explanation", "most_important_factors", "other_relevant_info")).lower()


df["expl_text"] = df["original_reference_response_parsed_response"].apply(explanation_text)
df["cited"] = df["expl_text"].apply(lambda s: {p for p, rx in PRINCIPLES.items() if re.search(rx, s)})
df["claims_irrelevance"] = df["expl_text"].str.contains(IRRELEVANCE, regex=True)
df["covered"] = [bool(c & ch) if isinstance(ch, set) else np.nan for c, ch in zip(df["cited"], df["changed_tags"])]

lk = df[df["linked"]].copy()
lk["covered"] = lk["covered"].astype(bool)
cov = lk.groupby("covered")[["egregious", "explanation_misled", "ref_flipped"]].mean()
cov["n"] = lk.groupby("covered").size()
print("Egregious rate by whether the explanation mentions a changed factor:\n", cov.round(3))
print("\nHow often explanations cite each principle:")
print(pd.Series([p for s in df["cited"] for p in s]).value_counts(normalize=False).div(len(df)).round(2))

# %% [markdown]
# ## 7. Egregious cases shared across reference models
#
# A pair that is egregious for most reference models points to a problem with
# the pair or the predictors, not to one model's unfaithfulness. A pair that is
# egregious for one model alone is a candidate for that model's idiosyncratic
# behaviour.

# %%
per_pair = lk.groupby(["original_question", "counterfactual_question"]).agg(
    n_models=("ref_model", "size"), n_egregious=("egregious", "sum"), pair_type=("pair_type", "first"),
    dim=("dim", "first"))
per_pair["egregious_share"] = per_pair["n_egregious"] / per_pair["n_models"]
fig, ax = plt.subplots(figsize=(8, 3))
counts = per_pair["n_egregious"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color=C_WITH, width=0.7)
ax.set_xlabel("Number of reference models for which the pair is egregious")
ax.set_ylabel("Pairs")
ax.set_yscale("log")
ax.set_title("Is egregious unfaithfulness model-specific or pair-specific?", loc="left")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_egregious_across_models.png", bbox_inches="tight")
plt.show()
per_pair.sort_values("n_egregious", ascending=False).head(10)

# %% [markdown]
# ## 8. Ranked candidates for spot-checking
#
# Score (higher = more interesting):
# - +3 explanation-misled: most predictors were right without the explanation
#   and all were wrong with it, so the explanation actively misled them
# - +2 covered: the explanation talks about a factor that changed
# - +2 claims a demographic attribute is irrelevant, in a demographic pair
# - +1 high stated confidence
# - +1 the pair is egregious for at most 2 reference models (model-specific)

# %%
eg = lk[lk["egregious"]].copy()
eg = eg.join(per_pair["n_egregious"].rename("pair_n_egregious"), on=["original_question", "counterfactual_question"])
eg["confidence"] = eg["original_reference_response_parsed_response"].apply(
    lambda p: (p or {}).get("confidence") if isinstance(p, dict) else None)
eg["score"] = (3 * eg["explanation_misled"] + 2 * eg["covered"]
               + 2 * (eg["claims_irrelevance"] & eg["demographic"])
               + (eg["confidence"].astype(str).str.upper() == "HIGH") + (eg["pair_n_egregious"] <= 2))
candidates = eg.sort_values(["score", "ref_model"], ascending=[False, True])
show_cols = ["pair_id", "ref_model", "pair_type", "dim", "score", "explanation_misled", "covered",
             "claims_irrelevance", "confidence", "pair_n_egregious", "n_wrong_without"]
candidates[show_cols].to_csv(TAB_DIR / "egregious_candidates.csv", index=False)
print(f"{len(candidates)} egregious rows -> {TAB_DIR / 'egregious_candidates.csv'}")
candidates[show_cols].head(20)

# %%
def short_scenario(prompt):
    """Drop the shared instruction preamble and keep the Case 1 / Case 2 text."""
    i = prompt.find("Case 1.")
    return prompt[i:].strip() if i >= 0 else prompt


def describe_change(row):
    o = cached_scenario(row.original_question_idx, row.original_question)
    c = cached_scenario(row.counterfactual_question_idx, row.counterfactual_question)
    if o is None or c is None:
        return "  (not linked to raw CSV)"
    lines = [f"  dimension: {o['dim']} -> {c['dim']} | in car: {o['in_car']} | Case 1 swerves: {o['interv']} | law: {o['law']}"]
    lines.append(f"  Case 1 victims: {o['c1']} [{o['tl1']}]  ->  {c['c1']} [{c['tl1']}]")
    lines.append(f"  Case 2 victims: {o['c2']} [{o['tl2']}]  ->  {c['c2']} [{c['tl2']}]")
    return "\n".join(lines)


def wrap(s, width=110, indent="    "):
    return textwrap.fill(str(s), width=width, initial_indent=indent, subsequent_indent=indent)


def show_case(pair_id, cot_chars=1500):
    row = df.loc[pair_id]
    p = row.original_reference_response_parsed_response or {}
    pc = row.counterfactual_reference_response_parsed_response or {}
    print("=" * 120)
    print(f"pair_id={pair_id} | reference model: {row.ref_model} | pair type: {row.pair_type} | "
          f"answer first: {row.original_answer_first}")
    print("WHAT CHANGED\n" + describe_change(row))
    print("\nORIGINAL SCENARIO\n" + wrap(short_scenario(row.original_question)))
    print(f"\nREFERENCE ON ORIGINAL: {row.original_reference_response_answer} (confidence {p.get('confidence')})")
    print("  explanation:\n" + wrap(as_text(p.get("explanation"))))
    print("  most important factors:\n" + wrap(as_text(p.get("most_important_factors"))))
    print(f"  cited principles (heuristic): {sorted(row.cited)} | claims irrelevance: {row.claims_irrelevance}")
    print("\nCOUNTERFACTUAL SCENARIO\n" + wrap(short_scenario(row.counterfactual_question)))
    print(f"\nREFERENCE ON COUNTERFACTUAL: {row.counterfactual_reference_response_answer} (confidence {pc.get('confidence')})")
    print("  its explanation on the counterfactual (compare with the original):\n" + wrap(as_text(pc.get("explanation"))))
    print("\nPREDICTORS             with explanation   without explanation")
    for name, w, wo in zip(row[NAMES], row[WITH], row[WITHOUT]):
        print(f"  {name.split('/')[-1]:22s} {str(w):18s} {wo}")
    cot = row.get("counterfactual_predictor_response_with_explanation_cot")
    if isinstance(cot, str) and cot.strip():
        print(f"\nFIRST PREDICTOR'S REASONING (with explanation, first {cot_chars} chars):\n" + wrap(cot[:cot_chars]))
    print()


for pid in candidates["pair_id"].head(5):
    show_case(pid)

# %% [markdown]
# ### Look at specific slices
# Uncomment one line at a time, or pass any `pair_id` to `show_case`.

# %%
# Demographic pair: the model says the attribute is irrelevant, then flips anyway
# for pid in candidates.query("demographic and claims_irrelevance and ref_flipped")["pair_id"].head(3): show_case(pid)

# Counts-only pairs that still produce an egregious failure (close counterfactuals)
# for pid in candidates.query("pair_type == 'counts only'")["pair_id"].head(3): show_case(pid)

# Egregious failures where the explanation never mentions what changed (out-of-scope counterfactuals)
# for pid in candidates.query("not covered")["pair_id"].head(3): show_case(pid)

# One pair across every reference model
# q = candidates.iloc[0]; [show_case(i) for i in df[(df.original_question == q.original_question) & (df.counterfactual_question == q.counterfactual_question)].pair_id]

# %% [markdown]
# ## 9. Questions to probe while spot-checking
#
# 1. **Construct validity:** Does the explanation-driven gain sit almost
#    entirely in the light and side swaps (section 4c)? If so, how much of the
#    paper's 6.0% describes the pairs rather than the models?
# 2. **Technical vs. social:** Do demographic changes carry a higher or lower
#    risk of egregious failure than structural ones (section 5)? Does the
#    paper's conclusion from the income data hold for Moral Machines?
# 3. **Stated vs. revealed principles:** In covered egregious cases, does the
#    model's explanation on the counterfactual contradict its explanation on
#    the original? `show_case` prints both.
# 4. **Coin-flip cases:** Are the "irrelevant attribute" flips really
#    unfaithfulness, or indifference? If gender truly doesn't matter, the
#    answer is near-random and no explanation can predict it. Resampling the
#    reference model would tell these apart.
# 5. **Predictor failure modes:** In explanation-misled cases, do predictors
#    over-apply the stated principle, or misread the scenario (for example,
#    which case is the swerve)? The first predictor's chain of thought is
#    printed when available.
# 6. **Pair-level vs. model-level:** Pairs egregious for many models
#    (section 7) suggest a broken pair, an ambiguous prompt, or a shared
#    predictor bias, not one model's unfaithfulness.
