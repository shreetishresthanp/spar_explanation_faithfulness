# %% [markdown]
# # Is Gemma 3 27B's answer-position effect on Moral Machines confounded by which scenarios got which format?
#
# Run from the repo root: `python analysis/gemma_stress_test/answer_first_confounding.py`. No API calls.
# In the stored data each scenario was asked in one format only (original_answer_first True or False), so the raw
# gap in P(CASE 1) between formats could come from the two scenario sets differing rather than from the format.
# Checks, all on the model in TARGET (one row per scenario):
#   1. Scenario features parsed from the prompt text for every row, all relative to Case 1 (Case 1 swerves, whom
#      Case 1 kills, net lives / character types Case 1 spares). When passengers are involved, Case 1 is always the
#      one that kills them, so "Case 1 kills passengers" also marks car-vs-pedestrian scenarios. Validated against data/raw/moral_machines_raw.csv
#      on the rows that link to it. `linked` itself is kept as a feature: it marks the generator version.
#   2. Balance: feature means by format, standardised mean differences (SMD), Welch t / chi-square p-values.
#   3. Propensity: 5-fold CV AUC of a logistic model predicting the format from the features (0.5 = as if random).
#   4. Gap in P(CASE 1), answer first minus answer last: raw; regression-adjusted (g-computation from a
#      standardised L2 logistic model, everyone set to first vs last); stratified (within cells of Case 1 kills
#      passengers x Case 1 swerves x legality, size-weighted); adjusted for scenario appeal (mean CASE 1 rate of the non-Gemma
#      placebo models on that scenario, added as a covariate). Also prints per-cell gaps, the gap within linked /
#      unlinked rows, and the gap by the raw CSV's scenario_dimension (linked rows), plus a regression-adjusted gap
#      with scenario_dimension added (linked rows only).
#   5. Placebo: every other reference model in data_parquet/ that answered >= 90% of the same scenarios with the
#      same format assignment. If the two scenario sets differed in how appealing Case 1 is, these models would
#      show a gap in the same direction too. A placebo model can still have its own format effect, so the
#      informative ones are those near 0 on this split. The appeal covariate also absorbs part of the placebo
#      models' own format effects, which biases the appeal-adjusted gap towards 0.
# Intervals: 95% bootstrap over scenarios (N_BOOT replicates).
# Writes figures/moral_machines/gemma_stress_test/answer_first_confounding.png and
# experiments/moral_machines/gemma_stress_test/answer_first_{balance,gaps}.csv.

# %%
import ast
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

if Path.cwd().name in {"analysis", "gemma_stress_test"}:
    os.chdir(Path(__file__).resolve().parents[2])

DATA_DIR = Path("data_parquet")
TARGET_FILE, TARGET = DATA_DIR / "gemma_3_predictions.parquet", "google/gemma-3-27b-it"
RAW_CSV = Path("data/raw/moral_machines_raw.csv")
FIG = Path("figures/moral_machines/gemma_stress_test")
OUT = Path("experiments/moral_machines/gemma_stress_test")
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT, SEED, MIN_OVERLAP = 1000, 0, 0.9

Q, POS, ANS = "original_question", "original_answer_first", "original_reference_response_answer"
MODEL, THINK = "original_reference_response_model_info_model", "original_reference_response_model_info_thinking"
COLS = ["original_dataset", "original_question_idx", Q, POS, ANS, MODEL, THINK]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False})


def read_mm(path):
    t = pq.read_table(path, columns=COLS, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
    t["config"] = t[MODEL] + np.where(t[THINK].isna(), "", " (" + t[THINK].astype(str) + ")")
    return t.drop_duplicates(["config", Q])


# %% 1. scenario features from the prompt text, relative to Case 1
CATEGORIES = {  # character type -> categories it counts towards
    "pets": {"dog", "cat"},
    "young": {"baby", "boy", "girl"},
    "elderly": {"elderly man", "elderly woman"},
    "female": {"woman", "girl", "elderly woman", "large woman", "female athlete", "female doctor", "female executive",
               "pregnant woman"},
    "large": {"large man", "large woman"},
    "athletes": {"male athlete", "female athlete"},
    "high_status": {"male doctor", "female doctor", "male executive", "female executive"},
    "low_status": {"criminal", "homeless person"},
    "pregnant": {"pregnant woman"},
}
CASE_RE = re.compile(r"will (?P<action>continue ahead|swerve).*?This will result in the death of (?P<killed>.+?), "
                     r"who (?:were|was) (?P<killed_who>.+?)\. Meanwhile, it will spare (?P<spared>.+?), "
                     r"who (?:were|was) (?P<spared_who>.+?)\.\s*$", re.S)


def singular(name):
    name = re.sub(r"men$", "man", name)
    name = {"homeless people": "homeless person", "homeless persons": "homeless person"}.get(name, name)
    return re.sub(r"ies$", "y", name) if name.endswith("ies") else re.sub(r"(?<!wo)(?<!m)s$", "", name)


def parse_group(text):
    """'1 elderly man, 2 female athletes, and 1 criminal' -> {'elderly man': 1, 'female athlete': 2, 'criminal': 1}"""
    out = {}
    for item in re.split(r",? and |, ", text):
        n, name = item.split(" ", 1)
        out[singular(name)] = out.get(singular(name), 0) + int(n)
    return out


def parse_case(text):
    m = CASE_RE.search(text)
    if m is None:
        raise ValueError(f"unparsed case: {text!r}")
    return {"swerve": m["action"] == "swerve", "killed": parse_group(m["killed"]), "spared": parse_group(m["spared"]),
            "killed_who": m["killed_who"]}


def features(question):
    """Case-1-relative features, plus Case 1's killed / spared groups (for validation against the raw CSV)."""
    case1 = parse_case(re.split(r"\n\nCase \d\.\n", question)[1])
    killed, spared, who = case1["killed"], case1["spared"], case1["killed_who"]
    f = {"c1_swerve": case1["swerve"],
         "c1_kills_passengers": "inside the car" in who,
         "c1_kills_lawful": "abiding by the law" in who,
         "c1_kills_lawbreakers": "flouting the law" in who,
         "net_lives": sum(spared.values()) - sum(killed.values())}
    for cat, members in CATEGORIES.items():
        f[f"net_{cat}"] = (sum(v for k, v in spared.items() if k in members)
                           - sum(v for k, v in killed.items() if k in members))
    return f, killed, spared


def groups_key(*groups):
    return sorted(sorted(g.items()) for g in groups)


df = read_mm(TARGET_FILE)
d = df[df["config"] == TARGET].dropna(subset=[ANS]).reset_index(drop=True)
d["case1"] = d[ANS].eq("CASE 1").astype(float)
parsed = [features(q) for q in d[Q]]
feat = pd.DataFrame([p[0] for p in parsed])

raw = pd.read_csv(RAW_CSV, index_col=0)
d["linked"] = [o in raw.index and raw.at[o, "prompt"] == q for o, q in zip(d["original_question_idx"], d[Q])]
feat["linked"] = d["linked"]
bad = sum(groups_key(killed, spared) != groups_key(ast.literal_eval(raw.at[o, "count_dict_1"]),
                                                    ast.literal_eval(raw.at[o, "count_dict_2"]))
          for (_, killed, spared), o, ok in zip(parsed, d["original_question_idx"], d["linked"]) if ok)
print(f"{TARGET}: {len(d)} scenarios; parser vs raw CSV mismatches on {d['linked'].sum()} linked rows: {bad}")
assert bad == 0, "parser disagrees with the raw CSV"
FEATURES = list(feat.columns)
pos = d[POS].astype(bool).values
y = d["case1"].values


# %% 2. balance between the two scenario sets
def smd(a, b):
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


rows = []
for f in FEATURES:
    a, b = feat.loc[pos, f].astype(float), feat.loc[~pos, f].astype(float)
    if feat[f].dtype == bool:
        p = stats.chi2_contingency(pd.crosstab(feat[f], pos))[1] if feat[f].nunique() > 1 else 1.0
    else:
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    rows.append({"feature": f, "mean_answer_first": a.mean(), "mean_answer_last": b.mean(), "smd": smd(a, b), "p": p})
balance = pd.DataFrame(rows).sort_values("smd", key=abs, ascending=False)
balance.to_csv(OUT / "answer_first_balance.csv", index=False)
print("\nbalance (answer first vs answer last), |SMD| > 0.1 is a common imbalance flag:\n",
      balance.round(3).to_string(index=False))

# %% 3. can the features predict the format?
X = feat.astype(float).values
logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
prop = cross_val_predict(logit, X, pos, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         method="predict_proba")[:, 1]
auc = roc_auc_score(pos, prop)
print(f"\npropensity CV AUC (features -> answer_first): {auc:.3f}  (0.5 = as if randomly assigned)")


# %% 4. raw, regression-adjusted, stratified and appeal-adjusted gaps
def raw_gap(y, pos):
    return y[pos].mean() - y[~pos].mean()


def adjusted_gap(Xc, y, pos):
    """g-computation: mean predicted P(CASE 1) with everyone answer-first minus everyone answer-last."""
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    m.fit(np.column_stack([pos, Xc]), y)
    return (m.predict_proba(np.column_stack([np.ones_like(pos), Xc]))[:, 1].mean()
            - m.predict_proba(np.column_stack([np.zeros_like(pos), Xc]))[:, 1].mean())


STRATA = feat["c1_kills_passengers"].astype(str) + "|" + feat["c1_swerve"].astype(str) + "|" \
    + (feat["c1_kills_lawful"].astype(int) - feat["c1_kills_lawbreakers"].astype(int)).astype(str)


def stratified_gap(y, pos, strata):
    t = pd.DataFrame({"y": y, "pos": pos, "s": strata})
    cells = t.groupby(["s", "pos"])["y"].agg(["mean", "size"]).unstack("pos").dropna()
    w = cells[("size", True)] + cells[("size", False)]
    return ((cells[("mean", True)] - cells[("mean", False)]) * w).sum() / w.sum()


def boot(fn, n, rng):
    return np.array([fn(rng.integers(0, n, n)) for _ in range(N_BOOT)])


def ci(est, reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"gap": est, "lo": lo, "hi": hi}


rng = np.random.default_rng(SEED)
n = len(d)
gaps = {
    "raw": ci(raw_gap(y, pos), boot(lambda i: raw_gap(y[i], pos[i]), n, rng)),
    "regression-adjusted": ci(adjusted_gap(X, y, pos), boot(lambda i: adjusted_gap(X[i], y[i], pos[i]), n, rng)),
    "stratified": ci(stratified_gap(y, pos, STRATA.values),
                     boot(lambda i: stratified_gap(y[i], pos[i], STRATA.values[i]), n, rng)),
}
print(f"\nstratified gap uses {STRATA.nunique()} cells (Case 1 kills passengers x Case 1 swerves x legality)")


def gap_table(by):
    """P(CASE 1) by format within each level of `by`, with the gap in points."""
    t = d.assign(by=by, pos=pos).groupby(["by", "pos"])["case1"].agg(["mean", "size"]).unstack("pos")
    t.columns = [f"{s}_{'first' if p else 'last'}" for s, p in t.columns]
    return t.assign(gap_pts=(t["mean_first"] - t["mean_last"]) * 100).round(3)


print("\nper-cell gaps (kills passengers | swerve | legality: 1 = kills lawful, -1 = kills lawbreakers, 0 = no light):\n",
      gap_table(STRATA.values).to_string())
# `linked` (row matches the current raw CSV) is the largest imbalance and marks the generator version
print("\ngap within linked / unlinked scenarios:\n", gap_table(d["linked"].values).to_string())
dims = np.where(d["linked"], [raw.at[o, "scenario_dimension"] if ok else "" for o, ok in
                              zip(d["original_question_idx"], d["linked"])], "not linked")
print("\ngap by scenario_dimension (raw CSV, linked rows only):\n", gap_table(dims).to_string())
lk = d["linked"].values
Xd = np.column_stack([X[lk], pd.get_dummies(dims[lk]).astype(float).values])
gaps["adjusted + dimension (linked)"] = ci(adjusted_gap(Xd, y[lk], pos[lk]),
                                           boot(lambda i: adjusted_gap(Xd[i], y[lk][i], pos[lk][i]), lk.sum(), rng))

# %% 5. placebo models: same scenarios, same format assignment
target_pos = d.set_index(Q)[POS].astype(bool)
others = []
for path in sorted(glob.glob(str(DATA_DIR / "*.parquet"))):
    t = read_mm(path)
    for cfg, s in t.groupby("config"):
        if cfg == TARGET:
            continue
        s = s[s[Q].isin(target_pos.index)].dropna(subset=[ANS])
        if len(s) >= MIN_OVERLAP * len(d) and (s[POS].astype(bool).values == target_pos.loc[s[Q]].values).all():
            others.append(s.assign(file=Path(path).stem))
placebo = pd.concat(others, ignore_index=True)
placebo["case1"] = placebo[ANS].eq("CASE 1").astype(float)
placebo_gaps = {}
for cfg, s in placebo.groupby("config"):
    yy, pp = s["case1"].values, s[POS].astype(bool).values
    placebo_gaps[cfg] = ci(raw_gap(yy, pp), boot(lambda i: raw_gap(yy[i], pp[i]), len(s), rng))

non_gemma = placebo[~placebo["config"].str.contains("gemma")]
appeal = non_gemma.groupby(Q)["case1"].mean().reindex(d[Q]).values
has = ~np.isnan(appeal)
Xa = np.column_stack([X[has], appeal[has]])
ya, pa = y[has], pos[has]
gaps["appeal-adjusted"] = ci(adjusted_gap(Xa, ya, pa), boot(lambda i: adjusted_gap(Xa[i], ya[i], pa[i]), has.sum(), rng))
print(f"appeal = mean CASE 1 rate on the scenario across {non_gemma['config'].nunique()} non-Gemma placebo models "
      f"({has.sum()} of {n} scenarios covered)")

out = pd.DataFrame([{"model": TARGET, "estimate": k, **v} for k, v in gaps.items()]
                   + [{"model": k, "estimate": "placebo raw", **v} for k, v in placebo_gaps.items()])
out.to_csv(OUT / "answer_first_gaps.csv", index=False)
print("\ngap in P(CASE 1), answer first minus answer last (points):")
print((out.set_index(["model", "estimate"])[["gap", "lo", "hi"]] * 100).round(1).to_string())
print(f"\nSUMMARY {TARGET}: raw {gaps['raw']['gap']:+.1%}, regression-adjusted {gaps['regression-adjusted']['gap']:+.1%}, "
      f"stratified {gaps['stratified']['gap']:+.1%}, appeal-adjusted {gaps['appeal-adjusted']['gap']:+.1%}; "
      f"propensity AUC {auc:.2f}; max |SMD| {balance['smd'].abs().max():.2f}; placebo gaps "
      f"{min(v['gap'] for v in placebo_gaps.values()):+.1%} to {max(v['gap'] for v in placebo_gaps.values()):+.1%}")


# %% figure: gaps (left) and balance (right)
def short(cfg):
    return cfg.split("/")[-1].removesuffix("-it").removesuffix("-001")


fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
labels = [f"{short(TARGET)}: {k}" for k in gaps] + [short(k) for k in placebo_gaps]
vals = list(gaps.values()) + list(placebo_gaps.values())
colors = [BLUE] * len(gaps) + [ORANGE if "gemma" in k else GREY for k in placebo_gaps]
ys = np.arange(len(vals))[::-1]
for yy, v, c in zip(ys, vals, colors):
    ax.errorbar(v["gap"] * 100, yy, xerr=[[(v["gap"] - v["lo"]) * 100], [(v["hi"] - v["gap"]) * 100]], fmt="o",
                color=c, ecolor=c, capsize=3)
    ax.text(v["hi"] * 100 + 1.5, yy, f"{v['gap'] * 100:+.0f}", va="center", fontsize=9, color=MUTED)
ax.axvline(0, color=INK, lw=0.8)
ax.axhline(len(placebo_gaps) - 0.5, color=GRID, lw=1.5)
ax.set_yticks(ys, labels)
ax.set_xlabel("P(CASE 1): answer first minus answer last (points)")
ax.set_title(f"Gap, {short(TARGET)} vs placebo models on the same split")

b = balance.iloc[::-1]
bx.barh(b["feature"], b["smd"], color=[ORANGE if abs(v) > 0.1 else GREY for v in b["smd"]])
bx.axvspan(-0.1, 0.1, color=GRID, alpha=0.5, lw=0, zorder=0)
bx.axvline(0, color=INK, lw=0.8)
bx.set_xlabel("SMD, answer first minus answer last")
bx.set_title(f"Scenario balance (propensity AUC {auc:.2f})")
fig.suptitle(f"Is the answer-position effect confounded by scenario? Moral Machines, {short(TARGET)}",
             fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "answer_first_confounding.png", bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {FIG / 'answer_first_confounding.png'}, {OUT / 'answer_first_balance.csv'}, "
      f"{OUT / 'answer_first_gaps.csv'}")
