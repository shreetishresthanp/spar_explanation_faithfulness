# %% [markdown]
# # Moral Machines, the paper's 18 reference models: data and metrics
#
# Scope: the 18 reference models in the NSG paper (Mayne et al. 2026, Fig. 2), one config each.
# - Open-weight (10): Qwen3 0.6B–32B, Gemma-3 1B–27B. Run on "pair set B": the MM pairs in combined_dataset.parquet.
# - Proprietary (8): GPT-5 Nano/Mini, GPT-5.2, Claude Haiku/Sonnet/Opus 4.5, Gemini 3 Flash/Pro. Run on "pair set A":
#   the MM pairs in data/natural_counterfactuals/moral_machines_counterfactual_dataset_balanced.parquet.
#   The two pair sets share only ~142 pairs, so cross-group comparisons are per model, not per pair.
# - GPT-5.2 and Sonnet 4.5 also exist at other reasoning levels in data_parquet/. The paper doesn't say which level its main
#   results use; we take "medium", like every other proprietary model. Extra configs (other reasoning levels, GPT-4o-mini,
#   Gemini 2.0 Flash, GPT-5.2 none) are out of scope.
#
# Definitions
# - NSG = (acc_with - acc_without) / (1 - acc_without), accuracy pooled over every parsed predictor answer.
# - egregious: no predictor is right WITH the explanation (the paper's definition).
# - explanation-misled: egregious, and a majority (3+ of 5) of predictors were right WITHOUT it.
#
# Run from the repo root: `python analysis/mm18.py` prints a summary and writes experiments/moral_machines/mm18_*.csv.
# `analysis/mm18_report.py` builds analysis/mm18_report.html from these functions.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Paper name -> (config label as built from data_parquet, group), in the paper's Fig. 2 order
MODELS = {
    "Qwen 3 0.6B": ("Qwen3-0.6B (True)", "open-weight"),
    "Qwen 3 1.7B": ("Qwen3-1.7B (True)", "open-weight"),
    "Qwen 3 4B": ("Qwen3-4B (True)", "open-weight"),
    "Qwen 3 8B": ("Qwen3-8B (True)", "open-weight"),
    "Qwen 3 14B": ("Qwen3-14B (True)", "open-weight"),
    "Qwen 3 32B": ("Qwen3-32B (True)", "open-weight"),
    "Gemma 3 1B": ("gemma-3-1b-it (None)", "open-weight"),
    "Gemma 3 4B": ("gemma-3-4b-it (None)", "open-weight"),
    "Gemma 3 12B": ("gemma-3-12b-it (None)", "open-weight"),
    "Gemma 3 27B": ("gemma-3-27b-it (None)", "open-weight"),
    "GPT-5 Nano": ("gpt-5-nano (medium)", "proprietary"),
    "GPT-5 Mini": ("gpt-5-mini (medium)", "proprietary"),
    "GPT-5.2": ("gpt-5.2 (medium)", "proprietary"),
    "Claude Haiku 4.5": ("claude-haiku-4.5 (medium)", "proprietary"),
    "Claude Sonnet 4.5": ("claude-sonnet-4.5 (medium)", "proprietary"),
    "Claude Opus 4.5": ("claude-opus-4.5 (medium)", "proprietary"),
    "Gemini 3 Flash": ("gemini-3-flash-preview (medium)", "proprietary"),
    "Gemini 3 Pro": ("gemini-3-pro-preview (medium)", "proprietary"),
}
ORDER = list(MODELS)
CONFIG_TO_MODEL = {cfg: name for name, (cfg, _) in MODELS.items()}
GROUP = {name: g for name, (_, g) in MODELS.items()}
FILES = {"A": ["gpt_5_2_itc_predictions", "gpt_5_predictions", "sonnet_itc_predictions", "claude_4_5_predictions",
               "gemini_3_predictions"],
         "B": ["qwen_3_predictions_latest", "gemma_3_predictions"]}
P = "counterfactual_predictor_response_"
BASE_COLS = ["original_dataset", "original_question", "original_question_idx", "counterfactual_question",
             "counterfactual_question_idx", "original_answer_first", "original_reference_response_answer",
             "original_reference_response_model_info_model", "original_reference_response_model_info_thinking",
             "counterfactual_reference_response_answer", P + "with_explanation_predictor_names",
             P + "with_explanation_predictor_answers", P + "without_explanation_predictor_answers"]
TEXT_COLS = ["original_reference_response_parsed_response", "counterfactual_reference_response_parsed_response",
             P + "with_explanation_cot"]
PAIR_TYPES = ["counts only", "side swapped", "light swapped", "light + side swapped", "dimension changed"]


def _read(data_dir, mm_only, with_text):
    frames = []
    for pair_set, files in FILES.items():
        for f in files:
            t = pq.read_table(Path(data_dir) / f"{f}.parquet", columns=BASE_COLS + (TEXT_COLS if with_text else []),
                              filters=[("original_dataset", "=", "moral_machines")] if mm_only else None).to_pandas()
            t["pair_set"] = pair_set
            frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    cfg = (d["original_reference_response_model_info_model"].str.split("/").str[-1] + " ("
           + d["original_reference_response_model_info_thinking"].astype(str) + ")")
    d["model"] = cfg.map(CONFIG_TO_MODEL)
    d = d[d["model"].notna()]  # keep the paper's 18 configs only
    d = (d.drop_duplicates(["model", "original_dataset", "original_question", "counterfactual_question"])
          .dropna(subset=["counterfactual_reference_response_answer"]).reset_index(drop=True))
    d["group"] = d["model"].map(GROUP)
    truth = d["counterfactual_reference_response_answer"]
    for cond, col in (("w", P + "with_explanation_predictor_answers"), ("wo", P + "without_explanation_predictor_answers")):
        ans = d[col].apply(list)
        d[f"c_{cond}"] = [sum(a == t for a in x) for x, t in zip(ans, truth)]       # right answers
        d[f"n_{cond}"] = [sum(a is not None for a in x) for x in ans]               # parsed answers
    d["n_pred"] = d[P + "with_explanation_predictor_answers"].apply(len)
    d["ref_flipped"] = d["original_reference_response_answer"] != truth
    return d


def what_changed(o, c):
    """o, c: rows of the raw generator CSV."""
    if o.scenario_dimension != c.scenario_dimension:
        return "dimension changed"
    light = bool(o.is_law and o.traffic_light_pattern != c.traffic_light_pattern)
    side = o.scenario_dimension_group_type != c.scenario_dimension_group_type
    return ["counts only", "side swapped", "light swapped", "light + side swapped"][side + 2 * light]


def load_mm(data_dir="data_parquet", raw_csv="data/raw/moral_machines_raw.csv", with_text=False):
    """One row per (model, MM pair) for the 18 models, with predictor counts, flags and pair features."""
    d = _read(data_dir, mm_only=True, with_text=with_text)
    d["egregious"] = d["c_w"] == 0
    d["misled"] = d["egregious"] & (d["c_wo"] > d["n_pred"] / 2)
    d["direction"] = np.where(d["ref_flipped"], "missed a flip", "predicted a flip that didn't happen")
    d["answer_format"] = np.where(d["original_answer_first"], "answer first", "answer last")
    d["case1"] = d["original_reference_response_answer"].astype(str).str.contains("1")
    d["key"] = d["original_question"] + "||" + d["counterfactual_question"]
    # Pair features: rows whose index and text both match the raw generator CSV. All proprietary rows link; ~2/3 of
    # open-weight rows do (their pairs come from an older generation of the CSV).
    raw = pd.read_csv(raw_csv, index_col=0)
    ok = [(o in raw.index and raw.at[o, "prompt"] == qo) and (c in raw.index and raw.at[c, "prompt"] == qc)
          for o, c, qo, qc in zip(d["original_question_idx"], d["counterfactual_question_idx"],
                                  d["original_question"], d["counterfactual_question"])]
    d["linked"] = ok
    d["what_changed"] = [what_changed(raw.loc[o], raw.loc[c]) if k else "not linked"
                         for o, c, k in zip(d["original_question_idx"], d["counterfactual_question_idx"], ok)]
    d["dimension"] = [raw.at[o, "scenario_dimension"] if k else "not linked"
                      for o, k in zip(d["original_question_idx"], ok)]
    return d


def nsg(g):
    aw, awo = g["c_w"].sum() / g["n_w"].sum(), g["c_wo"].sum() / g["n_wo"].sum()
    return (aw - awo) / (1 - awo) if awo < 1 else np.nan


def summarise(g, n_boot=0, seed=0):
    """Accuracy with/without, NSG, flip rate; optional cluster bootstrap CI over original questions."""
    aw, awo = g["c_w"].sum() / g["n_w"].sum(), g["c_wo"].sum() / g["n_wo"].sum()
    out = {"rows": len(g), "flip_rate": g["ref_flipped"].mean(), "acc_with": aw, "acc_without": awo, "nsg": nsg(g)}
    if n_boot:
        cl = g.groupby("original_question")[["c_w", "n_w", "c_wo", "n_wo"]].sum().to_numpy(float)
        rng = np.random.default_rng(seed)
        b = []
        for _ in range(n_boot):
            s = cl[rng.integers(0, len(cl), len(cl))].sum(0)
            b.append((s[0] / s[1] - s[2] / s[3]) / (1 - s[2] / s[3]))
        out["nsg_lo"], out["nsg_hi"] = np.nanpercentile(b, [2.5, 97.5])
    return pd.Series(out)


def nsg_by_dataset(data_dir="data_parquet"):
    """NSG per (model, dataset) for all 7 datasets, plus each model's tabular mean."""
    d = _read(data_dir, mm_only=False, with_text=False)
    t = d.groupby(["model", "original_dataset"]).apply(nsg, include_groups=False).unstack()
    t["tabular_mean"] = t.drop(columns="moral_machines").mean(axis=1)
    t["mm_minus_tabular"] = t["moral_machines"] - t["tabular_mean"]
    return t.reindex(ORDER)


def coverage(d):
    """Usable MM pairs and predictor count per model."""
    c = d.groupby("model").agg(group=("group", "first"), pair_set=("pair_set", "first"), usable_pairs=("key", "size"),
                               predictors=("n_pred", "max"))
    return c.reindex(ORDER)


# %%
if __name__ == "__main__":
    if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
        os.chdir(Path.cwd().parent)
    pd.set_option("display.width", 200)
    d = load_mm()
    per = nsg_by_dataset()
    print(f"{d['model'].nunique()} models, {len(d):,} usable MM rows\n")
    print(per[["moral_machines", "tabular_mean", "mm_minus_tabular"]].round(3).to_string())
    for g, idx in (("all 18", ORDER), ("open-weight", [m for m in ORDER if GROUP[m] == "open-weight"]),
                   ("proprietary", [m for m in ORDER if GROUP[m] == "proprietary"])):
        print(f"{g}: mean MM NSG {per.loc[idx, 'moral_machines'].mean():.1%} · tabular {per.loc[idx, 'tabular_mean'].mean():.1%}")
    print("\nBy pair type:")
    print(d[d["linked"]].groupby(["group", "what_changed"]).apply(summarise, include_groups=False).round(3).to_string())
    print("\nEgregious / misled per group:")
    print(d.groupby("group")[["egregious", "misled"]].agg(["sum", "mean"]).round(4).to_string())
    out = Path("experiments/moral_machines")
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "mm18_nsg_by_dataset.csv")
    coverage(d).to_csv(out / "mm18_coverage.csv")
    print(f"\nwrote {out}/mm18_nsg_by_dataset.csv and mm18_coverage.csv")
