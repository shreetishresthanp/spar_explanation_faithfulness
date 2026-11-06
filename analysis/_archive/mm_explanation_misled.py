# %% [markdown]
# # Moral Machines: explanation-misled rows
#
# A (reference config, pair) row is
# - **egregious** when every predictor is wrong *with* the reference model's explanation (the paper's definition), and
# - **explanation-misled** when it is egregious *and* a majority of predictors were right *without* the explanation
#   (3+ of 5; 2 of 2 for the three configs scored by only two predictors). The explanation turned a mostly-right guess
#   into an all-wrong one.
#
# Misleading has a direction:
# - **predicted a flip that didn't happen**: the reference model kept its answer, but the explanation made predictors
#   expect a change
# - **missed a flip**: the reference model changed its answer, and the explanation made predictors expect it to stay
#
# Run from the repo root: `python analysis/mm_explanation_misled.py` (writes a CSV of misled rows to
# experiments/moral_machines/). `analysis/mm_raw_data.ipynb` imports `load_rows` for the *Explanation misled* tab.

# %%
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SET_A = ["gpt_5_2_itc_predictions", "gpt_5_predictions", "sonnet_itc_predictions",
         "claude_4_5_predictions", "gemini_3_predictions"]
SET_B = ["qwen_3_predictions_latest", "gemma_3_predictions", "extra_three_ref_models_predictions"]
P = "counterfactual_predictor_response_"
COLS = ["original_question", "original_question_idx", "counterfactual_question", "counterfactual_question_idx",
        "original_answer_first", "original_reference_response_answer", "original_reference_response_parsed_response",
        "original_reference_response_model_info_model", "original_reference_response_model_info_thinking",
        "counterfactual_reference_response_answer", "counterfactual_reference_response_parsed_response",
        P + "with_explanation_predictor_names", P + "with_explanation_predictor_answers",
        P + "without_explanation_predictor_answers", P + "with_explanation_cot"]


def what_changed(o, c):
    """Same categories as pair_type in moral_machines_failure_analysis.py; o and c are rows of the raw CSV."""
    if o.scenario_dimension != c.scenario_dimension:
        return "dimension changed"
    light = bool(o.is_law and o.traffic_light_pattern != c.traffic_light_pattern)
    side = o.scenario_dimension_group_type != c.scenario_dimension_group_type
    return ["counts only", "side swapped", "light swapped", "light + side swapped"][side + 2 * light]


def confidence_level(parsed):
    """First of LOW / MEDIUM / HIGH in the stated confidence, else 'other'."""
    c = str((parsed or {}).get("confidence") or "").upper() if isinstance(parsed, dict) else ""
    for level in ("LOW", "MEDIUM", "HIGH"):
        if c.startswith(level):
            return level
    return "missing" if not c else "other"


def load_rows(data_dir="data_parquet", raw_csv="data/raw/moral_machines_raw.csv"):
    """One row per (reference config, MM pair) with predictor counts, misled flags and pair features."""
    frames = []
    for pair_set, files in (("A", SET_A), ("B", SET_B)):
        for f in files:
            t = pq.read_table(Path(data_dir) / f"{f}.parquet", columns=COLS,
                              filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
            t["pair_set"] = pair_set
            frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    thinking = d["original_reference_response_model_info_thinking"].astype(str).replace({"None": "default"})
    d["config"] = d["original_reference_response_model_info_model"].str.split("/").str[-1] + " (" + thinking + ")"
    d = (d.drop_duplicates(["config", "original_question", "counterfactual_question"])
          .dropna(subset=["counterfactual_reference_response_answer"]).reset_index(drop=True))

    truth = d["counterfactual_reference_response_answer"]
    d["n_pred"] = d[P + "with_explanation_predictor_answers"].apply(len)
    d["right_with"] = [sum(a == t for a in ans) for ans, t in zip(d[P + "with_explanation_predictor_answers"], truth)]
    d["right_without"] = [sum(a == t for a in ans) for ans, t in zip(d[P + "without_explanation_predictor_answers"], truth)]
    d["ref_flipped"] = d["original_reference_response_answer"] != truth
    d["egregious"] = d["right_with"] == 0
    d["misled"] = d["egregious"] & (d["right_without"] > d["n_pred"] / 2)
    d["direction"] = np.where(d["ref_flipped"], "missed a flip", "predicted a flip that didn't happen")
    d["confidence"] = d["original_reference_response_parsed_response"].apply(confidence_level)
    d["answer_format"] = np.where(d["original_answer_first"], "answer first", "answer last")

    # Pair features from the raw generator CSV, for rows whose index and text both match it
    raw = pd.read_csv(raw_csv, index_col=0)

    def linked(idx, text):
        return idx in raw.index and raw.at[idx, "prompt"] == text

    ok = [linked(o, qo) and linked(c, qc) for o, c, qo, qc in zip(
        d["original_question_idx"], d["counterfactual_question_idx"], d["original_question"], d["counterfactual_question"])]
    d["what_changed"] = [what_changed(raw.loc[o], raw.loc[c]) if k else "not linked (older generator)"
                         for o, c, k in zip(d["original_question_idx"], d["counterfactual_question_idx"], ok)]
    d["dimension"] = [raw.at[o, "scenario_dimension"] if k else "not linked (older generator)"
                      for o, k in zip(d["original_question_idx"], ok)]
    return d


def predictor_help_harm(d):
    """Per predictor: how often the explanation fixes a wrong guess (help) vs. breaks a right one (harm)."""
    cols = [P + "with_explanation_predictor_names", P + "with_explanation_predictor_answers",
            P + "without_explanation_predictor_answers"]
    long = d[["pair_set", "counterfactual_reference_response_answer"] + cols].copy()
    for c in cols:
        long[c] = long[c].apply(list)
    long = long.explode(cols)
    long["predictor"] = long[cols[0]].str.split("/").str[-1]
    t = long["counterfactual_reference_response_answer"]
    w, wo = long[cols[1]] == t, long[cols[2]] == t
    long["help"], long["harm"] = ~wo & w, wo & ~w
    return long.groupby(["pair_set", "predictor"]).agg(answers=("help", "size"), help=("help", "mean"),
                                                        harm=("harm", "mean")).reset_index()


# %%
if __name__ == "__main__":
    if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
        os.chdir(Path.cwd().parent)
    pd.set_option("display.width", 220)
    d = load_rows()
    for s, g in d.groupby("pair_set"):
        print(f"set {s}: {len(g):,} rows · egregious {g['egregious'].sum()} ({g['egregious'].mean():.1%}) · "
              f"misled {g['misled'].sum()} ({g['misled'].mean():.1%}; {g['misled'].sum() / g['egregious'].sum():.0%} of egregious)")
    print("\nMisled rate by reference config:")
    print(d.groupby(["pair_set", "config"]).agg(rows=("misled", "size"), misled=("misled", "sum"),
                                                rate=("misled", "mean")).round(4).to_string())
    print("\nDirection of misleading:")
    print(d[d["misled"]].groupby(["pair_set", "direction"]).size().to_string())
    print("\nPredictor help vs harm:")
    print(predictor_help_harm(d).round(4).to_string(index=False))
    out = Path("experiments/moral_machines")
    out.mkdir(parents=True, exist_ok=True)
    keep = ["pair_set", "config", "original_question_idx", "counterfactual_question_idx", "what_changed", "dimension",
            "direction", "original_reference_response_answer", "counterfactual_reference_response_answer", "n_pred",
            "right_with", "right_without", "confidence", "answer_format"]
    d.loc[d["misled"], keep].to_csv(out / "explanation_misled_rows.csv", index=False)
    print(f"\nwrote {out}/explanation_misled_rows.csv")
