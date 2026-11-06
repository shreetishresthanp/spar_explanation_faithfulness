# %% [markdown]
# # Coverage sample: hand labels -> scope, literal-prediction accuracy, failure modes
#
# Run from the repo root after `python analysis/mm18_rule.py`: `python analysis/mm18_coverage.py`.
# Input: experiments/moral_machines/mm18_coverage_labels.csv (downloaded from the report's Coverage sample).
# Scope is derived, not hand-labelled: a pair is in scope when a factor it changes (from the generator's record) is
# tagged "reason" or "irrelevant" in the explanation.

# %%
import os
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18_rule import feat  # noqa: E402  (scenario features from the raw generator CSV)

OUT = Path("experiments/moral_machines")
# rule feature -> label-grid factor
FEATURE_FACTOR = {"lives_saved": "numbers", "pets_saved": "species", "c1_swerves": "intervention",
                  "c1_kills_passengers": "passengers", "c1_spares_lawful": "law", "theme_species": "species",
                  "theme_social_value": "social", "theme_gender": "gender", "theme_age": "age", "theme_fitness": "fitness",
                  "theme_utilitarianism": "numbers"}

lab = pd.read_csv(OUT / "mm18_coverage_labels.csv")
lab = lab[lab["predict"].notna()].copy()
rule = pd.read_csv(OUT / "mm18_coverage_sample_rule.csv")[["model", "original_idx", "counterfactual_idx", "rule_p_flip",
                                                            "rule_margin", "near_tie"]]
lab = lab.merge(rule, on=["model", "original_idx", "counterfactual_idx"], how="left", validate="one_to_one")
rule_right = pd.read_csv(OUT / "mm18_rule_pairs.csv")[["model", "original_question_idx", "counterfactual_question_idx", "rule_right"]]
lab = lab.merge(rule_right.rename(columns={"original_question_idx": "original_idx", "counterfactual_question_idx": "counterfactual_idx"}),
                on=["model", "original_idx", "counterfactual_idx"], how="left", validate="one_to_one")


raw = pd.read_csv("data/raw/moral_machines_raw.csv", index_col=0)


def changed_factors(o, c):
    """Factors that differ between the two scenarios. Head counts count as a change to "numbers" even when the net
    difference between the sides (the rule's lives_saved) stays the same, e.g. 2 vs 3 -> 3 vs 4."""
    fo, fc = feat.loc[o], feat.loc[c]
    out = {FEATURE_FACTOR[f] for f in FEATURE_FACTOR if fo[f] != fc[f]}
    if (raw.at[o, "count_dict_1"], raw.at[o, "count_dict_2"]) != (raw.at[c, "count_dict_1"], raw.at[c, "count_dict_2"]):
        out.add("numbers")
    return sorted(out)


lab["changed"] = [changed_factors(o, c) for o, c in zip(lab["original_idx"], lab["counterfactual_idx"])]
lab["changed_cited"] = [[f for f in ch if r[f"cites_{f}"] in ("reason", "irrelevant")] for ch, (_, r) in zip(lab["changed"], lab.iterrows())]
lab["in_scope"] = lab["changed_cited"].str.len() > 0
lab["flipped"] = lab["answer_original"] != lab["answer_counterfactual"]
lab["literal"] = lab["predict"].map(lambda p: "can't tell" if p == "can't tell" else None)
lab.loc[lab["literal"].isna(), "literal"] = (lab["predict"] == lab["answer_counterfactual"]).map({True: "right", False: "wrong"})


def mode(r):
    """Failure mode from the labels (failures only)."""
    if r.outcome != "failure":
        return "n/a (success)"
    if r.literal == "right":
        return "explanation settles it correctly: predictors misread"
    if r.literal == "can't tell":
        return "underdetermined: explanation doesn't settle the counterfactual"
    return "explanation points the wrong way: stated rule overridden" if r.in_scope else "explanation points the wrong way: change out of scope"


lab["mode"] = lab.apply(mode, axis=1)
lab.drop(columns=["literal"]).assign(literal_reading=lab["literal"]).to_csv(OUT / "mm18_coverage_labels_scored.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    for m, g in lab.groupby("model"):
        print(f"=== {m}: {len(g)} labelled rows ({(g.outcome == 'failure').sum()} failures, {(g.outcome == 'success').sum()} successes)\n")
        print("Your literal reading of the explanation, by outcome:")
        print(pd.crosstab(g["outcome"], g["literal"], margins=True), "\n")
        print("In scope (a changed factor is cited), by outcome:")
        print(pd.crosstab(g["outcome"], g["in_scope"], margins=True), "\n")
        print("Rule near-tie, by outcome:")
        print(pd.crosstab(g["outcome"], g["near_tie"], margins=True), "\n")
        print("Literal reading x near-tie (failures only):")
        f = g[g["outcome"] == "failure"]
        print(pd.crosstab(f["literal"], f["near_tie"], margins=True), "\n")
        print("Literal reading x fitted rule right (failures only):")
        print(pd.crosstab(f["literal"], f["rule_right"], margins=True), "\n")
        print("Failure modes:")
        print(f["mode"].value_counts().to_string(), "\n")
        print("Literal reading by pair type:")
        print(pd.crosstab(g["what_changed"], [g["outcome"], g["literal"]]), "\n")
        print("Rows:")
        print(g[["n", "what_changed", "changed", "changed_cited", "flipped", "outcome", "predict", "answer_counterfactual",
                 "right_with", "right_without", "rule_p_flip", "near_tie"]].sort_values(["outcome", "what_changed"]).to_string(index=False))
