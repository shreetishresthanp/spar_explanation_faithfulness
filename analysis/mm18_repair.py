# %% [markdown]
# # Re-paired Moral Machines counterfactuals from scenarios each model already answered
#
# Run from the repo root after `python analysis/mm18_rule.py`: `python analysis/mm18_repair.py`.
# Plan: moral_machines_groundtruth_plan.md (experiment 2). For each model, every scenario it answered (both sides of its
# existing pairs) has a known answer and its own explanation. Any two answered scenarios A, B with the same structure
# form a new (original = A, counterfactual = B) pair whose reference answer on B is already known, so only predictor
# calls are needed. Writes experiments/moral_machines/mm18_repair_candidates.csv (all candidates, with group labels)
# and mm18_repair_sample.csv (the proposed balanced sample).

# %%
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import mm18_rule as R  # noqa: E402  (runs the rule fit: ~7 s)
from mm18 import load_mm  # noqa: E402

OUT = Path("experiments/moral_machines")
MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
NEAR_TIE, N_PER_CELL, SEED = 0.2, 20, 0
RULE_FEATURES = [f for f in R.FEATURES if f != "answer_first"]
FEATURE_FACTOR = {"lives_saved": "numbers", "pets_saved": "species", "c1_swerves": "intervention",
                  "c1_kills_passengers": "passengers", "c1_spares_lawful": "law", "theme_species": "species",
                  "theme_social_value": "social", "theme_gender": "gender", "theme_age": "age", "theme_fitness": "fitness",
                  "theme_utilitarianism": "numbers"}
# Approximate keyword map from the model's own most_important_factors list to the 9 factors
KEYWORDS = {"law": ["law", "legal", "signal", "red", "green", "jaywalk", "illegal", "right-of-way", "right of way", "rule"],
            "passengers": ["passenger", "occupant", "inside the car", "in the car", "pedestrian", "bystander"],
            "intervention": ["swerv", "interven", "inaction", "continue ahead", "stay", "course", "redirect", "action"],
            "numbers": ["number", "lives", "fatalit", "death", "casualt", "more people", "fewer", "count", "utilitarian"],
            "species": ["human", "animal", "pet", "dog", "cat", "species"],
            "age": ["age", "young", "child", "elderly", "old", "life-year", "life year", "boy", "girl", "baby"],
            "gender": ["gender", "women", "woman", "men", "male", "female", "sex"],
            "social": ["social", "status", "doctor", "executive", "criminal", "homeless", "profession"],
            "fitness": ["fitness", "athlete", "large", "fit", "physical"]}


def cited(factors):
    text = " ".join(map(str, factors if isinstance(factors, (list, tuple, np.ndarray)) else [])).lower()
    return {f for f, kws in KEYWORDS.items() if any(k in text for k in kws)}


# %% Held-out P(Case 1) for every answered scenario, per model (same rule and folds as mm18_rule.py)
dtext = load_mm(with_text=True)
raw = pd.read_csv(R.RAW_CSV, index_col=0)
scen_rows, cand_rows = [], []
for m in MODELS:
    g = R.d[R.d["model"] == m].reset_index(drop=True)
    dec = R.decisions(g)
    cols = [c for c in R.FEATURES if dec[c].std() > 0]
    dec["p_case1"] = R.oof_proba(dec, cols)
    t = dtext[dtext["model"] == m].set_index(["original_question_idx", "counterfactual_question_idx"])
    for side, idx_col, ans_col, resp_col in (
            ("o", "original_question_idx", "original_reference_response_answer", "original_reference_response_parsed_response"),
            ("c", "counterfactual_question_idx", "counterfactual_reference_response_answer", "counterfactual_reference_response_parsed_response")):
        s = dec[dec["side"] == side].reset_index(drop=True)
        resp = t.loc[list(zip(g["original_question_idx"], g["counterfactual_question_idx"])), resp_col].to_numpy()
        scen_rows.append(pd.DataFrame({
            "model": m, "idx": g[idx_col].to_numpy(), "pair_id": g["pair_id"].to_numpy(), "side": side,
            "answer": g[ans_col].to_numpy(), "answer_first": s["answer_first"].to_numpy() == 1,
            "p_case1": s["p_case1"].to_numpy(),
            "has_explanation": [isinstance(r, dict) and bool(r.get("explanation")) for r in resp],
            "cited": [cited(r.get("most_important_factors")) if isinstance(r, dict) else set() for r in resp]}))
scen = pd.concat(scen_rows, ignore_index=True)
scen = scen.join(raw[["scenario_dimension", "is_in_car", "is_interventionism", "count_dict_1", "count_dict_2"]], on="idx")
existing = set(zip(R.d["model"], R.d["original_question_idx"], R.d["counterfactual_question_idx"]))

# %% Candidate new pairs: same model, theme, car/swerve structure and answer position; A must have an explanation
F = R.feat
for (m, dim, car, iv, af), g in scen.groupby(["model", "scenario_dimension", "is_in_car", "is_interventionism", "answer_first"]):
    A, B = g[g["has_explanation"]], g
    if A.empty:
        continue
    a, b = A.reset_index(drop=True), B.reset_index(drop=True)
    ia, ib = np.repeat(np.arange(len(a)), len(b)), np.tile(np.arange(len(b)), len(a))
    keep = a["idx"].to_numpy()[ia] != b["idx"].to_numpy()[ib]
    ia, ib = ia[keep], ib[keep]
    fa, fb = F.loc[a["idx"].to_numpy()[ia], RULE_FEATURES].to_numpy(), F.loc[b["idx"].to_numpy()[ib], RULE_FEATURES].to_numpy()
    diff = fa != fb
    counts_differ = ((a["count_dict_1"].to_numpy()[ia] != b["count_dict_1"].to_numpy()[ib])
                     | (a["count_dict_2"].to_numpy()[ia] != b["count_dict_2"].to_numpy()[ib]))
    n_changed = diff.sum(1)
    ok = (n_changed > 0) | counts_differ  # drop exact duplicates
    changed_factors = [{FEATURE_FACTOR[RULE_FEATURES[j]] for j in np.flatnonzero(row)} | ({"numbers"} if cd else set())
                       for row, cd in zip(diff, counts_differ)]
    cit = a["cited"].to_numpy()[ia]
    p_b = b["p_case1"].to_numpy()[ib]
    ans_a, ans_b = a["answer"].to_numpy()[ia], b["answer"].to_numpy()[ib]
    a_case1 = pd.Series(ans_a).astype(str).str.contains("1").to_numpy()
    cand_rows.append(pd.DataFrame({
        "model": m, "dimension": dim, "is_in_car": car, "is_interventionism": iv, "answer_first": af,
        "original_idx": a["idx"].to_numpy()[ia], "counterfactual_idx": b["idx"].to_numpy()[ib],
        "answer_original": ans_a, "answer_counterfactual": ans_b, "flipped": ans_a != ans_b,
        "rule_p_case1_cf": p_b, "rule_p_flip": np.where(a_case1, 1 - p_b, p_b), "rule_margin": np.abs(p_b - 0.5),
        "n_rule_features_changed": n_changed,
        "changed_factors": [",".join(sorted(c)) for c in changed_factors],
        "changed_cited": [bool(c & s) for c, s in zip(changed_factors, cit)]})[ok])
cand = pd.concat(cand_rows, ignore_index=True)
cand = cand[[(m, o, c) not in existing for m, o, c in zip(cand["model"], cand["original_idx"], cand["counterfactual_idx"])]]
cand["near_tie"] = cand["rule_margin"] < NEAR_TIE
cand["change_size"] = np.select([cand["n_rule_features_changed"] == 0, cand["n_rule_features_changed"] == 1],
                                ["counts only (net unchanged)", "one rule feature"], "several rule features")
cand["cited_label"] = np.where(cand["changed_cited"], "cited", "uncited")
cand["tie_label"] = np.where(cand["near_tie"], "near-tie", "clear")

# %% Proposed sample: per model, near-tie/clear x cited/uncited, N_PER_CELL each, single-feature changes only,
# each original scenario used at most once, balanced on flips where possible
samples = []
pool = cand[cand["change_size"] == "one rule feature"]
for (m, tie, cit), g in pool.groupby(["model", "tie_label", "cited_label"]):
    g = g.sample(frac=1, random_state=SEED).drop_duplicates("original_idx")
    half = N_PER_CELL // 2
    pick = pd.concat([g[g["flipped"]].head(half), g[~g["flipped"]].head(half)])
    pick = pd.concat([pick, g.drop(pick.index).head(N_PER_CELL - len(pick))])
    samples.append(pick)
sample = pd.concat(samples, ignore_index=True)

# %% Write and summarise
OUT.mkdir(parents=True, exist_ok=True)
cand.to_csv(OUT / "mm18_repair_candidates.csv", index=False)
sample.to_csv(OUT / "mm18_repair_sample.csv", index=False)
counts = (cand.groupby(["model", "change_size", "tie_label", "cited_label"]).size().unstack(["tie_label", "cited_label"], fill_value=0))
flip = cand.groupby(["model", "change_size", "tie_label"])["flipped"].mean().unstack("tie_label")
counts.to_csv(OUT / "mm18_repair_counts.csv")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(f"{len(scen):,} answered scenarios; {len(cand):,} candidate new pairs\n")
    print("Candidate pairs by model, change size, near-tie x cited:")
    print(counts, "\n")
    print("Reference flip rate by change size and near-tie:")
    print(flip.round(3), "\n")
    print("Proposed sample (single-feature changes):")
    print(sample.groupby(["model", "tie_label", "cited_label"]).agg(pairs=("flipped", "size"), flip_rate=("flipped", "mean")).round(2))
