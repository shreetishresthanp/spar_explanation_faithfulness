# %% [markdown]
# # Moral Machines coverage: how many MM data points each reference and predictor model saw
#
# Counts, for every reference-model config in `data_parquet/` (model + reasoning setting, 26 in all):
# - Moral Machines rows in the files, and how many remain after de-duplicating configs that appear in two files
# - rows with a usable counterfactual reference answer (the label predictors are scored against)
# - unique original questions and unique (original, counterfactual) pairs
# and, for every (reference config, predictor) combination, how many predictor answers exist and how many parsed,
# with and without the explanation.
#
# Run from the repo root: `python analysis/mm_coverage.py` (writes CSVs to experiments/moral_machines/).
# `analysis/mm_raw_data.ipynb` imports `coverage_tables` to draw the Coverage tab of the HTML report.

# %%
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SET_A = ["gpt_5_2_itc_predictions", "gpt_5_predictions", "sonnet_itc_predictions",
         "claude_4_5_predictions", "gemini_3_predictions"]
SET_B = ["qwen_3_predictions_latest", "gemma_3_predictions", "extra_three_ref_models_predictions"]
P = "counterfactual_predictor_response_"
COLS = ["original_dataset", "original_question", "counterfactual_question",
        "original_reference_response_model_info_model", "original_reference_response_model_info_thinking",
        "original_reference_response_answer", "counterfactual_reference_response_answer",
        P + "with_explanation_predictor_names", P + "with_explanation_predictor_answers",
        P + "without_explanation_predictor_answers"]


def config_label(model, thinking):
    """'gpt-5.2 (medium)'; models with no reasoning setting are labelled '(default)', as in the notebook."""
    t = thinking.astype(str).replace({"None": "default"})
    return model.str.split("/").str[-1] + " (" + t + ")"


def coverage_tables(data_dir="data_parquet"):
    """Return (ref, pred, files): per-reference-config counts, per (config, predictor) counts, per-file row counts."""
    frames, file_rows = [], []
    for pair_set, files in (("A", SET_A), ("B", SET_B)):
        for f in files:
            path = Path(data_dir) / f"{f}.parquet"
            total = pq.ParquetFile(path).metadata.num_rows
            t = pq.read_table(path, columns=COLS, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
            t["file"], t["pair_set"] = f, pair_set
            frames.append(t)
            file_rows.append(dict(file=f, pair_set=pair_set, rows_all_datasets=total, mm_rows=len(t)))
    d = pd.concat(frames, ignore_index=True)
    d["config"] = config_label(d["original_reference_response_model_info_model"],
                               d["original_reference_response_model_info_thinking"])

    raw = d.groupby("config").agg(pair_set=("pair_set", "first"), files=("file", lambda s: ", ".join(sorted(set(s)))),
                                  mm_rows_in_files=("file", "size"))
    d = d.drop_duplicates(["config", "original_question", "counterfactual_question"])
    d["usable"] = d["counterfactual_reference_response_answer"].notna()
    ref = raw.join(d.groupby("config").agg(
        mm_rows_deduplicated=("usable", "size"),
        mm_rows_usable=("usable", "sum"),
        unique_original_questions=("original_question", "nunique"),
        unique_pairs=("counterfactual_question", "size")))
    ref["dropped_no_cf_answer"] = ref["mm_rows_deduplicated"] - ref["mm_rows_usable"]
    pairs_per_set = d.groupby("pair_set").apply(
        lambda g: g[["original_question", "counterfactual_question"]].drop_duplicates().shape[0], include_groups=False)
    ref["pairs_in_its_pair_set"] = ref["pair_set"].map(pairs_per_set)
    ref["coverage_of_pair_set"] = ref["mm_rows_usable"] / ref["pairs_in_its_pair_set"]
    ref = ref.reset_index().sort_values(["pair_set", "config"])

    # One row per (reference config, pair, predictor slot), usable rows only
    u = d[d["usable"]]
    long = u[["config", "pair_set", P + "with_explanation_predictor_names", P + "with_explanation_predictor_answers",
              P + "without_explanation_predictor_answers"]].copy()
    for c in long.columns[2:]:
        long[c] = long[c].apply(list)
    long = long.explode([P + "with_explanation_predictor_names", P + "with_explanation_predictor_answers",
                         P + "without_explanation_predictor_answers"])
    long["predictor"] = long[P + "with_explanation_predictor_names"].str.split("/").str[-1]
    pred = long.groupby(["config", "predictor"]).agg(
        pair_set=("pair_set", "first"),
        answers=("predictor", "size"),
        parsed_with=(P + "with_explanation_predictor_answers", lambda s: s.notna().sum()),
        parsed_without=(P + "without_explanation_predictor_answers", lambda s: s.notna().sum())).reset_index()
    return ref, pred, pd.DataFrame(file_rows)


# %%
if __name__ == "__main__":
    if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
        os.chdir(Path.cwd().parent)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    ref, pred, files = coverage_tables()
    print(files.to_string(index=False))
    print(f"\n{ref['config'].nunique()} reference configs; with at least one usable MM row: "
          f"{(ref['mm_rows_usable'] > 0).sum()}")
    print(ref.drop(columns="files").to_string(index=False))
    print(f"\nPredictors: {sorted(pred['predictor'].unique())}")
    print(pred.groupby("predictor")[["answers", "parsed_with", "parsed_without"]].sum().to_string())
    out = Path("experiments/moral_machines")
    out.mkdir(parents=True, exist_ok=True)
    ref.to_csv(out / "mm_coverage_reference.csv", index=False)
    pred.to_csv(out / "mm_coverage_predictor.csv", index=False)
    print(f"\nwrote {out}/mm_coverage_reference.csv and mm_coverage_predictor.csv")
