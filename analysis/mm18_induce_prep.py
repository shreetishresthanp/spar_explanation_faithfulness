# %% [markdown]
# # Induced ground truth (exploratory, extends Exp 1): splits, ceilings and writer inputs for 9 models
#
# Run from the repo root: `python analysis/mm18_induce_prep.py`. No API calls. Doesn't touch any mm18_gt_* file.
# Writes to experiments/moral_machines/:
# - mm18_induce_split.csv: (model, pair_id, split). GPT-5.2, Gemini 3 Flash and Gemma 3 27B reuse Exp 1's held-out
#   pairs exactly (asserted); the other 6 models get new ones drawn the same way (20 per pair type, seed 0; GPT-5 Nano
#   is capped at half of each pair type because it has only 172 linked pairs).
# - mm18_induce_heldout_rule.csv: the training-only rule's prediction on each held-out counterfactual (the LR ceiling)
# - mm18_induce_writer_input/<model>.txt: up to 150 training pairs (30 per pair type), each as the verbatim original
#   scenario text → answer, counterfactual text → answer, with answer position as a plain field on every scenario. The only input to the
#   free-form writer, whose prompt (analysis/prompts/mm18_induce_writer_prompt.txt) is identical for every model and
#   doesn't mention answer position; whether a draft finds it is measured as a result.
# - mm18_induce_coef_material.md: training coefficients, saved training statistics and 40 training decisions per model,
#   the only input for the coefficient-informed texts. Unlike mm18_gt_train_material.md it gives no held-out accuracy,
#   and every percentage a text may cite is computed here and saved.

# %%
import ast
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
import mm18_rule as R  # noqa: E402  (fits the full rule on import; we only reuse its data and helpers)
from mm18 import PAIR_TYPES  # noqa: E402

OUT = Path("experiments/moral_machines")
MODELS = R.MODELS  # 8 proprietary + Gemma 3 27B
EXP1_MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
N_HELD_PER_TYPE, N_SHOW_PER_TYPE, N_WRITER_PER_TYPE, N_BOOT, SEED = 20, 8, 30, 200, 0
raw = pd.read_csv(R.RAW_CSV, index_col=0)


def describe(idx):
    """Compact one-line rendering of a scenario (same as mm18_gt_prep.describe)."""
    r = raw.loc[idx]
    c1, c2 = ast.literal_eval(r["count_dict_1"]), ast.literal_eval(r["count_dict_2"])
    light = ast.literal_eval(r["traffic_light_pattern"])
    fmt = lambda c: ", ".join(f"{v} {k}" for k, v in c.items())
    lt = lambda l: {"green": " (crossing legally)", "red": " (crossing illegally)", "NA": ""}[l]
    act1, act2 = ("swerve", "stay") if r["is_interventionism"] else ("stay", "swerve")
    where1 = "passengers" if r["is_in_car"] else "pedestrians"
    return (f"Case 1 = {act1}, kills {fmt(c1)} [{where1}{lt(light[0])}] | "
            f"Case 2 = {act2}, kills {fmt(c2)} [pedestrians{lt(light[1])}]")


def rate(mask, did):
    """'x% of n' for a subset of training decisions; '—' if empty."""
    n = int(mask.sum())
    return f"{did[mask].mean():.0%} of {n}" if n else "— (none)"


def train_stats(x):
    """Saved training statistics a coefficient-informed text may cite. x = training decisions; y = 1 if Case 1."""
    car, ped = x["c1_kills_passengers"] == 1, x["c1_kills_passengers"] == 0
    c1 = x["y"] == 1
    spares_lawful = ((x["c1_spares_lawful"] > 0) & c1) | ((x["c1_spares_lawful"] < 0) & ~c1)
    stays = ((x["c1_swerves"] == -1) & c1) | ((x["c1_swerves"] == 1) & ~c1)
    spares_more = ((x["lives_saved"] > 0) & c1) | ((x["lives_saved"] < 0) & ~c1)
    spares_humans = ((x["theme_species"] < 0) & c1) | ((x["theme_species"] > 0) & ~c1)
    ped_light = lambda v: car & (x["c1_spares_lawful"] == v)  # in car scenarios Case 2 holds the pedestrians
    rows = [
        ("Car vs pedestrians: sacrifices the passengers (all)", rate(car, c1)),
        ("  … pedestrians crossing legally", rate(ped_light(1), c1)),
        ("  … pedestrians crossing illegally", rate(ped_light(-1), c1)),
        ("  … no traffic light", rate(ped_light(0), c1)),
        ("  … passengers outnumber pedestrians", rate(car & (x["lives_saved"] < 0), c1)),
        ("  … pedestrians outnumber passengers", rate(car & (x["lives_saved"] > 0), c1)),
        ("  … equal numbers", rate(car & (x["lives_saved"] == 0), c1)),
        ("Pedestrians only, one group legal: spares the legal group", rate(ped & (x["c1_spares_lawful"] != 0), spares_lawful)),
        ("Pedestrians only, no light, equal numbers: stays in lane", rate(ped & (x["c1_spares_lawful"] == 0) & (x["lives_saved"] == 0), stays)),
        ("Pedestrians only, no light: stays in lane (all)", rate(ped & (x["c1_spares_lawful"] == 0), stays)),
        ("Numbers differ: spares the larger group (all)", rate(x["lives_saved"] != 0, spares_more)),
        ("  … pedestrians only", rate(ped & (x["lives_saved"] != 0), spares_more)),
        ("Humans vs animals: spares the humans", rate(x["theme_species"] != 0, spares_humans)),
        ("Answer asked first: chooses Case 1", rate(x["answer_first"] == 1, c1)),
        ("Answer asked last: chooses Case 1", rate(x["answer_first"] == -1, c1)),
    ]
    for t, lab in [("gender", "the women"), ("age", "the younger group"), ("social_value", "the higher-status group"),
                   ("fitness", "the fitter group"), ("utilitarianism", "the larger group")]:
        # theme_t = +1 means Case 1 kills the theme's first group (women / younger / lower-status / less fit / smaller)
        m = x[f"theme_{t}"] != 0
        spares_first = ((x[f"theme_{t}"] < 0) & c1) | ((x[f"theme_{t}"] > 0) & ~c1)
        did = ~spares_first if t in {"social_value", "fitness", "utilitarianism"} else spares_first
        rows.append((f"{t} scenarios: spares {lab} (raw rate; confounded by layout, prefer the coefficient)", rate(m, did)))
    return rows


# %% Splits, training-only rule and held-out ceiling
exp1 = pd.read_csv(OUT / "mm18_gt_split.csv")
split_rows, held_rows, material = [], [], [
    "# Training material for the coefficient-informed ground-truth texts (induced-ground-truth experiment)\n",
    "Everything here comes from TRAINING pairs only. Held-out pairs, and any accuracy measured on them, never appear.",
    "Coefficients: rule fitted on training pairs (standardised; positive = pushes towards Case 1).",
    "Training statistics: raw rates on training decisions (both sides of each pair). A text may cite only these numbers.\n"]
writer_dir = OUT / "mm18_induce_writer_input"
writer_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
for m in MODELS:
    g = R.d[R.d["model"] == m].reset_index(drop=True)
    if m in EXP1_MODELS:  # exactly Exp 1's held-out pairs
        held_ids = set(exp1[(exp1["model"] == m) & (exp1["split"] == "heldout")]["pair_id"])
    else:  # same draw as Exp 1; the half-of-type cap only binds for GPT-5 Nano
        held_ids = set(pd.concat([g[g["what_changed"] == pt].sample(min(N_HELD_PER_TYPE, (g["what_changed"] == pt).sum() // 2),
                                                                    random_state=SEED) for pt in PAIR_TYPES])["pair_id"])
    g["split"] = np.where(g["pair_id"].isin(held_ids), "heldout", "train")
    split_rows.append(g[["model", "pair_id", "what_changed", "split"]])

    dec = R.decisions(g)
    dec["split"] = dec["pair_id"].map(g.set_index("pair_id")["split"])
    tr, te = dec[dec["split"] == "train"], dec[dec["split"] == "heldout"]
    cols = [c for c in R.FEATURES if tr[c].std() > 0]
    full = R.fit(tr[cols], tr["y"])
    pairs, by_pair = tr["pair_id"].unique(), tr.groupby("pair_id").indices
    boots = []
    for _ in range(N_BOOT):
        ix = np.concatenate([by_pair[p] for p in rng.choice(pairs, len(pairs))])
        boots.append(R.fit(tr.iloc[ix][cols], tr["y"].iloc[ix])[-1].coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ct = pd.DataFrame(dict(feature=cols, label=[R.FEATURE_LABEL[c] for c in cols], coef=full[-1].coef_[0], lo=lo, hi=hi))

    cf = te[te["side"] == "c"].copy()
    cf["rule_p_case1"] = full.predict_proba(cf[cols])[:, 1]
    gh = g[g["split"] == "heldout"].set_index("pair_id")
    cf["reference_answer"] = gh.loc[cf["pair_id"], "counterfactual_reference_response_answer"].to_numpy()
    cf["rule_answer"] = np.where(cf["rule_p_case1"] > 0.5, "CASE 1", "CASE 2")
    held_rows.append(cf.assign(model=m)[["model", "pair_id", "rule_p_case1", "rule_answer", "reference_answer"]])

    gt = g[g["split"] == "train"]
    # Free-form writer input: 30 training pairs per pair type (all of them if fewer), shuffled so types are mixed
    wsel = pd.concat([gt[gt["what_changed"] == pt].sample(min(N_WRITER_PER_TYPE, (gt["what_changed"] == pt).sum()),
                                                          random_state=SEED) for pt in PAIR_TYPES]).sample(frac=1, random_state=SEED)
    # Verbatim scenario text (what the reference model read), so the writer isn't limited to the rule's features.
    # Answer position is a plain field on every scenario; the writer prompt says nothing about it.
    pos = lambda f: "answer position: before reasoning" if f else "answer position: after reasoning"
    lines = []
    for k, r in enumerate(wsel.itertuples(), 1):
        lines += [f"=== Pair {k} ===",
                  f"[A] {pos(r.o_answer_first)}", raw.at[r.original_question_idx, "prompt"].strip(),
                  f"Evaluator's answer: {r.original_reference_response_answer}", "",
                  f"[B] {pos(r.c_answer_first)}", raw.at[r.counterfactual_question_idx, "prompt"].strip(),
                  f"Evaluator's answer: {r.counterfactual_reference_response_answer}", ""]
    (writer_dir / f"{m.replace(' ', '_')}.txt").write_text("\n".join(lines) + "\n")
    split_rows[-1] = split_rows[-1].assign(writer_input=g["pair_id"].isin(wsel["pair_id"]))

    # Coefficient-informed material: coefficients, saved statistics, 40 training decisions (as in Exp 1)
    show = pd.concat([gt[gt["what_changed"] == pt].sample(min(N_SHOW_PER_TYPE, (gt["what_changed"] == pt).sum()),
                                                          random_state=SEED) for pt in PAIR_TYPES])
    material += [f"\n## {m}\n", f"Training pairs: {len(gt)} ({len(tr)} decisions).\n",
                 "| Factor | coef | 95% CI |", "|---|---|---|"]
    material += [f"| {r.label} | {r.coef:+.2f} | [{r.lo:+.2f}, {r.hi:+.2f}] |"
                 for r in ct.reindex(ct["coef"].abs().sort_values(ascending=False).index).itertuples()]
    material += ["", f"Intercept (not a clean Case 1 bias): {full[-1].intercept_[0]:+.2f}", "",
                 "### Training statistics\n", "| Situation | all decisions | answer asked first | answer asked last |",
                 "|---|---|---|---|"]
    by_pos = [train_stats(tr), train_stats(tr[tr["answer_first"] == 1]), train_stats(tr[tr["answer_first"] == -1])]
    material += [f"| {a[0]} | {a[1]} | {b[1]} | {c[1]} |" for a, b, c in zip(*by_pos)]
    material += ["", "### 40 training decisions (theme · answer position · scenario → choice)\n"]
    for k, r in enumerate(show.itertuples(), 1):
        side_idx = r.original_question_idx if k % 2 else r.counterfactual_question_idx
        ans = r.original_reference_response_answer if k % 2 else r.counterfactual_reference_response_answer
        p = "answer first" if r.o_answer_first else "answer last"
        material.append(f"{k}. {r.dimension} · {p} · {describe(side_idx)} → **{ans}**")

split = pd.concat(split_rows)
split.to_csv(OUT / "mm18_induce_split.csv", index=False)
pd.concat(held_rows).to_csv(OUT / "mm18_induce_heldout_rule.csv", index=False)
(OUT / "mm18_induce_coef_material.md").write_text("\n".join(material) + "\n")

if __name__ == "__main__":
    print(split.groupby(["model", "split"]).size().unstack())
    print("\nwriter-input pairs per model:", split.groupby("model")["writer_input"].sum().to_dict())
    h = pd.concat(held_rows)
    print("\ntraining-only rule accuracy on held-out counterfactuals (ceiling; not shown to any writer):")
    print(h.assign(ok=h["rule_answer"] == h["reference_answer"]).groupby("model")["ok"].mean().round(3).to_string())
