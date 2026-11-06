# %% [markdown]
# # Moral Machines vs. tabular NSG, per reference model
#
# Regenerates the numbers in `moral_machines_README.md`:
# 1. NSG per (reference model, dataset), for both pair sets in `data_parquet/`
# 2. the paper's 18-model Moral Machines average (6.0%), under different
#    choices of config for GPT-5.2 and Sonnet 4.5
# 3. NSG conditional on whether the reference model flipped, MM vs. tabular
# 4. MM NSG and Case 1 rate by answer position (answer-first / answer-last)
# 5. how often explanations mention option position or order
# 6. NSG on the 142 MM pairs shared by pair sets A and B
#
# Run from the repo root: `python analysis/mm_vs_tabular_nsg.py`

# %%
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

SET_A = ["gpt_5_2_itc_predictions", "gpt_5_predictions", "sonnet_itc_predictions",
         "claude_4_5_predictions", "gemini_3_predictions"]
SET_B = ["qwen_3_predictions_latest", "gemma_3_predictions", "extra_three_ref_models_predictions"]
P = "counterfactual_predictor_response_"
COLS = ["original_dataset", "original_question", "counterfactual_question", "original_answer_first",
        "original_reference_response_answer", "original_reference_response_parsed_response",
        "original_reference_response_model_info_model", "original_reference_response_model_info_thinking",
        "counterfactual_reference_response_answer", "counterfactual_hamming_distance",
        P + "with_explanation_predictor_answers", P + "without_explanation_predictor_answers"]


def load(files, pair_set):
    d = pd.concat([pq.read_table(f"data_parquet/{f}.parquet", columns=COLS).to_pandas() for f in files])
    d["model"] = (d["original_reference_response_model_info_model"].str.split("/").str[-1] + " ("
                  + d["original_reference_response_model_info_thinking"].astype(str) + ")")
    d = d.drop_duplicates(["model", "original_question", "counterfactual_question"])
    d = d.dropna(subset=["counterfactual_reference_response_answer"]).reset_index(drop=True)
    ref = d["counterfactual_reference_response_answer"]
    for cond, col in (("w", P + "with_explanation_predictor_answers"), ("wo", P + "without_explanation_predictor_answers")):
        answers = d[col].apply(lambda a: [x for x in a if x is not None])
        d[f"c_{cond}"] = [sum(x == r for x in a) for a, r in zip(answers, ref)]
        d[f"n_{cond}"] = answers.apply(len)
    d["flip"] = d["original_reference_response_answer"] != ref
    d["pair_set"] = pair_set
    return d


def nsg(g):
    aw, awo = g.c_w.sum() / g.n_w.sum(), g.c_wo.sum() / g.n_wo.sum()
    return pd.Series({"rows": len(g), "flip": g["flip"].mean() if "flip" in g else np.nan, "acc_w": aw, "acc_wo": awo,
                      "nsg": (aw - awo) / (1 - awo) if awo < 1 else np.nan})


d = pd.concat([load(SET_A, "A"), load(SET_B, "B")], ignore_index=True)
d["is_mm"] = d["original_dataset"] == "moral_machines"

# %% [markdown]
# ## 1. NSG per (reference model, dataset)

# %%
per = d.groupby(["pair_set", "model", "original_dataset"]).apply(nsg, include_groups=False)["nsg"].unstack()
per["tabular_mean"] = per.drop(columns="moral_machines").mean(axis=1)
per["mm_minus_tabular"] = per["moral_machines"] - per["tabular_mean"]
print(per.round(3).sort_values("mm_minus_tabular").to_string())

# %% [markdown]
# ## 2. Reproduce the paper's 6.0% (18 models, one config each, unweighted mean)
#
# The paper's 18 models are Qwen3 x6, Gemma-3 x4, GPT-5 nano/mini/5.2,
# Claude 4.5 Haiku/Sonnet/Opus and Gemini 3 Flash/Pro. GPT-5.2 and Sonnet 4.5
# each have several reasoning configs here, so try each choice.

# %%
mm = per["moral_machines"].droplevel("pair_set")
FIXED = [m for m in mm.index if m.startswith(("Qwen3", "gemma-3", "gpt-5-nano", "gpt-5-mini", "claude-haiku",
                                              "claude-opus", "gemini-3"))]
assert len(FIXED) == 16, FIXED  # 18 paper models minus GPT-5.2 and Sonnet 4.5
rows = []
for gpt in [m for m in mm.index if m.startswith("gpt-5.2")]:
    for sonnet in [m for m in mm.index if m.startswith("claude-sonnet")]:
        rows.append({"gpt-5.2": gpt, "sonnet": sonnet, "mean_mm_nsg": mm[FIXED + [gpt, sonnet]].mean()})
repro = pd.DataFrame(rows).sort_values("mean_mm_nsg")
print(repro.round(4).to_string(index=False))

FRONTIER = ["gpt-5-nano (medium)", "gpt-5-mini (medium)", "gpt-5.2 (medium)", "claude-haiku-4.5 (medium)",
            "claude-sonnet-4.5 (medium)", "claude-opus-4.5 (medium)", "gemini-3-flash-preview (medium)",
            "gemini-3-pro-preview (medium)"]
SMALL_OPEN = [m for m in mm.index if m.startswith(("Qwen3", "gemma-3"))]
print(f"\nFrontier, 8 models one config each: MM NSG mean {mm[FRONTIER].mean():.3f}; "
      f"tabular mean {per['tabular_mean'].droplevel('pair_set')[FRONTIER].mean():.3f}")
print(f"Qwen3 + Gemma-3, 10 models: MM NSG mean {mm[SMALL_OPEN].mean():.3f}; "
      f"tabular mean {per['tabular_mean'].droplevel('pair_set')[SMALL_OPEN].mean():.3f}")

# %% [markdown]
# ## 3. NSG conditional on whether the reference model flipped
#
# When the reference keeps its answer, "copy the original answer" is already
# right, so pooled NSG depends on the flip rate. Conditioning removes that.

# %%
cond = d.groupby(["pair_set", "is_mm", "flip"]).apply(nsg, include_groups=False)
print(cond.round(3).to_string())

# %% [markdown]
# ## 4. Moral Machines by answer position
#
# `answer_first` is randomised per pair, independently of the scenario content.
# A content-driven model should pick Case 1 at the same rate in both formats.

# %%
mmd = d[d.is_mm].copy()
mmd["case1"] = mmd["original_reference_response_answer"].astype(str).str.contains("1")
pos = mmd.groupby(["model", "original_answer_first"]).agg(case1=("case1", "mean"), n=("case1", "size")).unstack()
pos.columns = [f"{a}_{'ans_first' if b else 'ans_last'}" for a, b in pos.columns]
pos["case1_gap"] = pos["case1_ans_last"] - pos["case1_ans_first"]
pos = pos.join(mmd.groupby(["model", "original_answer_first"]).apply(nsg, include_groups=False)["nsg"].unstack()
               .rename(columns={True: "nsg_ans_first", False: "nsg_ans_last"}))
print(pos.round(3).sort_values("case1_gap").to_string())

# %% [markdown]
# ## 5. Do explanations ever mention option position or order?

# %%
POSITION = r"\b(first|second) (option|case|choice)|\border\b|position|listed (first|second)|arbitrar|coin|random(ly)?\b"


def expl_text(p):
    if not isinstance(p, dict):
        return ""
    parts = []
    for k in ("explanation", "most_important_factors"):
        v = p.get(k)
        # parsed fields are strings, lists or numpy arrays depending on the model
        parts.append(" ".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else str(v or ""))
    return " ".join(parts).lower()


mmd["mentions_position"] = mmd["original_reference_response_parsed_response"].apply(expl_text).str.contains(POSITION)
print(mmd.groupby("model")["mentions_position"].mean().round(3).sort_values().to_string())

# %% [markdown]
# ## 6. The 142 Moral Machines pairs shared by pair sets A and B

# %%
mmd["pair_key"] = mmd["original_question"] + "||" + mmd["counterfactual_question"]
shared = set(mmd.loc[mmd.pair_set == "A", "pair_key"]) & set(mmd.loc[mmd.pair_set == "B", "pair_key"])
print(f"{len(shared)} shared pairs")
print(mmd[mmd.pair_key.isin(shared)].groupby("model").apply(nsg, include_groups=False).round(3).sort_values("nsg").to_string())
