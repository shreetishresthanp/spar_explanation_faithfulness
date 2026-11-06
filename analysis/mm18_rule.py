# %% [markdown]
# # Per-model decision rule for Moral Machines (plan: moral_machines_rule_fit_plan.md)
#
# Run from the repo root: `python analysis/mm18_rule.py`. For each reference model, fits an L2 logistic regression of
# "chose Case 1" on the scenario factors, cross-validated by pair. Writes to experiments/moral_machines/:
# - mm18_rule_coefs.csv: standardised coefficients with 95% pair-bootstrap intervals
# - mm18_rule_ceiling.csv: held-out accuracy of the rule vs. baselines vs. predictors with/without the explanation
# - mm18_rule_pairs.csv: one row per (model, pair) with the rule's P(Case 1), P(flip) and margin on the counterfactual
# - mm18_rule_boundary.csv: NSG and flip rate by P(flip) quintile and by near-tie threshold
# - mm18_coverage_sample_rule.csv: the coverage sample joined to the rule columns

# %%
import ast
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18 import FILES, GROUP, ORDER, load_mm, summarise  # noqa: E402

# numpy 2 + macOS Accelerate emit spurious matmul warnings inside sklearn; the fitted values are finite
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

OUT = Path("experiments/moral_machines")
RAW_CSV = "data/raw/moral_machines_raw.csv"
MODELS = [m for m in ORDER if GROUP[m] == "proprietary"] + ["Gemma 3 27B"]
TOP3 = ["GPT-5.2", "Gemini 3 Flash", "Gemini 3 Pro"]
LOW_POWER = {"GPT-5 Nano"}
N_BOOT, N_FOLDS, C_REG, SEED = 200, 5, 1.0, 0
NEAR_TIE, NEAR_TIE_SENS = 0.2, [0.1, 0.2, 0.3]
PETS = {"dog", "cat"}
# First group per theme, in the generator's order (tabular_utils.scenario_dimension_group_types)
THEME_FIRST = {"species": "human", "social_value": "lower", "gender": "female", "age": "younger", "fitness": "lower",
               "utilitarianism": "less"}
THEME_LABEL = {"species": "kills the humans (vs. animals)", "social_value": "kills the lower-status group",
               "gender": "kills the women", "age": "kills the younger group", "fitness": "kills the less fit group",
               "utilitarianism": "kills the smaller group"}
LIGHT = {"green": 1, "red": -1, "NA": 0}
FEATURES = ["lives_saved", "pets_saved", "c1_swerves", "c1_kills_passengers", "c1_spares_lawful"] + \
           [f"theme_{t}" for t in THEME_FIRST] + ["answer_first"]
FEATURE_LABEL = {"lives_saved": "Case 1 saves more people (net)", "pets_saved": "Case 1 saves more animals (net)",
                 "c1_swerves": "Case 1 is the swerve", "c1_kills_passengers": "Case 1 kills the passengers",
                 "c1_spares_lawful": "Case 1 spares the lawful crossers", "answer_first": "[ANSWER] asked first",
                 **{f"theme_{t}": f"Case 1 {lab}" for t, lab in THEME_LABEL.items()}}


# %% Scenario features (all Case 1 relative to Case 2; Case 1 kills set 1 and spares set 2)
def scenario_features(raw):
    c1, c2 = raw["count_dict_1"].map(ast.literal_eval), raw["count_dict_2"].map(ast.literal_eval)
    humans = lambda c: sum(v for k, v in c.items() if k not in PETS)
    pets = lambda c: sum(v for k, v in c.items() if k in PETS)
    light = raw["traffic_light_pattern"].map(ast.literal_eval)
    first = raw["scenario_dimension_group_type"].map(lambda s: ast.literal_eval(s)[0])
    f = pd.DataFrame(index=raw.index)
    f["lives_saved"] = c2.map(humans) - c1.map(humans)
    f["pets_saved"] = c2.map(pets) - c1.map(pets)
    f["c1_swerves"] = np.where(raw["is_interventionism"], 1, -1)
    f["c1_kills_passengers"] = raw["is_in_car"].astype(int)
    f["c1_spares_lawful"] = light.map(lambda l: LIGHT[l[1]] - LIGHT[l[0]])
    for t, g in THEME_FIRST.items():
        f[f"theme_{t}"] = np.where(raw["scenario_dimension"] == t, np.where(first == g, 1, -1), 0)
    return f


def answer_first_flags():
    """The prompt format of each scenario, parsed from the prompt text: is [ANSWER] requested before [EXPLANATION]?"""
    cols = ["original_dataset", "original_question", "counterfactual_question", "original_reference_response_model_info_model",
            "original_reference_response_model_info_thinking", "original_question_prompt", "counterfactual_question_prompt"]
    frames = [pq.read_table(f"data_parquet/{f}.parquet", columns=cols, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
              for fs in FILES.values() for f in fs]
    t = pd.concat(frames, ignore_index=True)
    first = lambda p: p.rfind("[ANSWER]") < p.rfind("[EXPLANATION]")
    t["o_answer_first"], t["c_answer_first"] = t["original_question_prompt"].map(first), t["counterfactual_question_prompt"].map(first)
    keys = cols[1:5]
    return t.drop_duplicates(keys)[keys + ["o_answer_first", "c_answer_first"]], keys


# %% Load: one row per (model, pair), then one decision per scenario
d = load_mm()
d = d[d["model"].isin(MODELS) & d["linked"]].copy()
flags, keys = answer_first_flags()
d = d.merge(flags, on=keys, how="left", validate="many_to_one")
assert d["c_answer_first"].notna().all(), "missing prompt format for some rows"
mismatch = (d["o_answer_first"] != d["c_answer_first"]).mean()
assert (d["o_answer_first"] == d["original_answer_first"]).all(), "parsed format disagrees with original_answer_first"
d["pair_id"] = d["original_question_idx"].astype(str) + "-" + d["counterfactual_question_idx"].astype(str)

raw = pd.read_csv(RAW_CSV, index_col=0)
feat = scenario_features(raw)


def decisions(g):
    o = feat.loc[g["original_question_idx"]].reset_index(drop=True)
    c = feat.loc[g["counterfactual_question_idx"]].reset_index(drop=True)
    o["answer_first"] = np.where(g["o_answer_first"], 1, -1)
    c["answer_first"] = np.where(g["c_answer_first"], 1, -1)
    o["y"] = g["original_reference_response_answer"].astype(str).str.contains("1").to_numpy().astype(int)
    c["y"] = g["counterfactual_reference_response_answer"].astype(str).str.contains("1").to_numpy().astype(int)
    for x, side in ((o, "o"), (c, "c")):
        x["pair_id"], x["side"] = g["pair_id"].to_numpy(), side
    return pd.concat([o, c], ignore_index=True)


def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=2000)).fit(X, y)


def oof_proba(dec, cols):
    """Held-out P(Case 1) for every decision, with folds grouped by pair."""
    p = np.full(len(dec), np.nan)
    for tr, te in GroupKFold(N_FOLDS).split(dec, groups=dec["pair_id"]):
        p[te] = fit(dec.iloc[tr][cols], dec["y"].iloc[tr]).predict_proba(dec.iloc[te][cols])[:, 1]
    return p


# %% Fit every model
coef_rows, ceil_rows, pair_rows = [], [], []
rng = np.random.default_rng(SEED)
for m in MODELS:
    g = d[d["model"] == m].reset_index(drop=True)
    dec = decisions(g)
    cols = [c for c in FEATURES if dec[c].std() > 0]  # e.g. no answer-first rows for some models
    dec["p_full"], dec["p_pos"] = oof_proba(dec, cols), oof_proba(dec, ["answer_first"]) if dec["answer_first"].std() > 0 else dec["y"].mean()

    # Coefficients (standardised features) with a pair-level bootstrap
    full = fit(dec[cols], dec["y"])
    pairs = dec["pair_id"].unique()
    by_pair = {p: ix for p, ix in dec.groupby("pair_id").indices.items()}
    boots = []
    for _ in range(N_BOOT):
        ix = np.concatenate([by_pair[p] for p in rng.choice(pairs, len(pairs))])
        boots.append(fit(dec.iloc[ix][cols], dec["y"].iloc[ix])[-1].coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    for k, c in enumerate(cols):
        coef_rows.append(dict(model=m, feature=c, label=FEATURE_LABEL[c], coef=full[-1].coef_[0][k], lo=lo[k], hi=hi[k]))
    coef_rows.append(dict(model=m, feature="intercept", label="Baseline preference for Case 1",
                          coef=full[-1].intercept_[0], lo=np.nan, hi=np.nan))

    # Counterfactual side: rule prediction, P(flip), margin
    cf = dec[dec["side"] == "c"].set_index("pair_id").loc[g["pair_id"]]
    p_c = cf["p_full"].to_numpy()
    a_o_case1 = g["original_reference_response_answer"].astype(str).str.contains("1").to_numpy()
    y_c = cf["y"].to_numpy()
    g["rule_p_case1"] = p_c
    g["rule_p_flip"] = np.where(a_o_case1, 1 - p_c, p_c)
    g["rule_margin"] = np.abs(p_c - 0.5)
    g["rule_right"] = (p_c > 0.5).astype(int) == y_c
    pair_rows.append(g)

    acc_all = ((dec["p_full"] > 0.5).astype(int) == dec["y"]).mean()
    acc_pos = ((np.asarray(dec["p_pos"]) > 0.5).astype(int) == dec["y"]).mean()
    acc_wo, acc_w = g["c_wo"].sum() / g["n_wo"].sum(), g["c_w"].sum() / g["n_w"].sum()
    acc_rule = g["rule_right"].mean()
    ceil_rows.append(dict(
        model=m, group=GROUP[m], top3=m in TOP3, low_power=m in LOW_POWER, pairs=len(g), decisions=len(dec),
        acc_all_decisions_rule=acc_all, acc_all_decisions_majority=max(dec["y"].mean(), 1 - dec["y"].mean()),
        acc_all_decisions_position_only=acc_pos,
        cf_acc_copy_original=(~g["ref_flipped"]).mean(), cf_acc_predictors_without=acc_wo, cf_acc_predictors_with=acc_w,
        cf_acc_rule=acc_rule, nsg_explanation=(acc_w - acc_wo) / (1 - acc_wo), nsg_rule=(acc_rule - acc_wo) / (1 - acc_wo),
        flip_rate=g["ref_flipped"].mean(), format_mismatch_share=(g["o_answer_first"] != g["c_answer_first"]).mean()))

coefs, ceiling = pd.DataFrame(coef_rows), pd.DataFrame(ceil_rows)
pr = pd.concat(pair_rows, ignore_index=True)

# %% Sanity checks (plan: stop and report if one fails)
pos = coefs[coefs["feature"] == "answer_first"].set_index("model")
checks = []
for m in ["GPT-5.2", "Gemini 3 Flash"]:
    r = pos.loc[m] if m in pos.index else None
    checks.append((f"{m}: answer_first coefficient ~ 0 (|coef| < 0.25)", r is not None and abs(r["coef"]) < 0.25,
                   "n/a" if r is None else f"{r['coef']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"))
gm = "Gemma 3 27B"
checks.append((f"{gm}: answer_first coefficient large (|coef| > 0.5)", gm in pos.index and abs(pos.loc[gm, "coef"]) > 0.5,
               f"{pos.loc[gm, 'coef']:+.2f}" if gm in pos.index else "n/a (no format variation)"))
c = ceiling.set_index("model")
checks.append((f"{gm}: position-only accuracy close to full rule (gap < 5 pts)",
               c.loc[gm, "acc_all_decisions_rule"] - c.loc[gm, "acc_all_decisions_position_only"] < 0.05,
               f"rule {c.loc[gm, 'acc_all_decisions_rule']:.3f} vs position {c.loc[gm, 'acc_all_decisions_position_only']:.3f}"))
for m in TOP3:
    checks.append((f"{m}: full rule well above position-only (gap > 10 pts)",
                   c.loc[m, "acc_all_decisions_rule"] - c.loc[m, "acc_all_decisions_position_only"] > 0.10,
                   f"rule {c.loc[m, 'acc_all_decisions_rule']:.3f} vs position {c.loc[m, 'acc_all_decisions_position_only']:.3f}"))
checks = pd.DataFrame(checks, columns=["check", "passed", "value"])

# %% Boundary: NSG and flip rate by P(flip) quintile, and by near-tie threshold
pr["p_flip_bin"] = pr.groupby("model")["rule_p_flip"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
bound = []
scopes = {**{m: [m] for m in MODELS}, "top 3 pooled": TOP3, "8 proprietary pooled": [m for m in MODELS if GROUP[m] == "proprietary"]}
for scope, ms in scopes.items():
    s = pr[pr["model"].isin(ms)]
    for b, gb in s.groupby("p_flip_bin"):
        bound.append(dict(scope=scope, split="P(flip) quintile", level=int(b), mean_p_flip=gb["rule_p_flip"].mean(),
                          **summarise(gb, n_boot=N_BOOT, seed=SEED)))
    for th in NEAR_TIE_SENS:
        for tie, gb in s.groupby(s["rule_margin"] < th):
            bound.append(dict(scope=scope, split=f"near-tie (margin < {th})", level="near-tie" if tie else "clear",
                              mean_p_flip=gb["rule_p_flip"].mean(), **summarise(gb, n_boot=N_BOOT, seed=SEED)))
bound = pd.DataFrame(bound)
pt = (pr[pr["model"].isin(TOP3)].groupby("what_changed")
      .agg(pairs=("rule_p_flip", "size"), mean_p_flip=("rule_p_flip", "mean"), near_tie=("rule_margin", lambda s: (s < NEAR_TIE).mean()),
           flip_rate=("ref_flipped", "mean")).reset_index())

# %% Write
OUT.mkdir(parents=True, exist_ok=True)
coefs.to_csv(OUT / "mm18_rule_coefs.csv", index=False)
ceiling.to_csv(OUT / "mm18_rule_ceiling.csv", index=False)
bound.to_csv(OUT / "mm18_rule_boundary.csv", index=False)
pt.to_csv(OUT / "mm18_rule_pair_types.csv", index=False)
checks.to_csv(OUT / "mm18_rule_checks.csv", index=False)
keep = ["model", "group", "pair_id", "original_question_idx", "counterfactual_question_idx", "what_changed", "dimension",
        "o_answer_first", "c_answer_first", "ref_flipped", "c_w", "n_w", "c_wo", "n_wo", "rule_p_case1", "rule_p_flip",
        "rule_margin", "rule_right", "p_flip_bin"]
pr[keep].to_csv(OUT / "mm18_rule_pairs.csv", index=False)
cov_path = OUT / "mm18_coverage_sample.csv"
if cov_path.exists():
    cov = pd.read_csv(cov_path).merge(
        pr[keep[:1] + keep[3:5] + ["rule_p_case1", "rule_p_flip", "rule_margin", "p_flip_bin"]].rename(
            columns={"original_question_idx": "original_idx", "counterfactual_question_idx": "counterfactual_idx"}),
        on=["model", "original_idx", "counterfactual_idx"], how="left")
    cov["near_tie"] = cov["rule_margin"] < NEAR_TIE
    cov.to_csv(OUT / "mm18_coverage_sample_rule.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"answer position differs between original and counterfactual prompt in {mismatch:.1%} of pairs\n")
    print(checks.to_string(index=False), "\n")
    print(ceiling[["model", "pairs", "acc_all_decisions_rule", "acc_all_decisions_position_only", "cf_acc_copy_original",
                   "cf_acc_predictors_without", "cf_acc_predictors_with", "cf_acc_rule", "nsg_explanation", "nsg_rule"]]
          .round(3).to_string(index=False), "\n")
    print(coefs[coefs["model"].isin(TOP3 + [gm])].pivot(index="feature", columns="model", values="coef").round(2), "\n")
    print(bound[(bound["scope"] == "top 3 pooled")][["split", "level", "rows", "mean_p_flip", "flip_rate", "acc_with",
                                                      "acc_without", "nsg", "nsg_lo", "nsg_hi"]].round(3).to_string(index=False), "\n")
    print(pt.round(3).to_string(index=False))
