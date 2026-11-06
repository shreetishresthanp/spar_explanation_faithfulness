# %% [markdown]
# # Induced ground truth: analysis (exploratory, extends Exp 1)
#
# Run from the repo root after `python analysis/mm18_induce_run.py`: `python analysis/mm18_induce.py`. No API calls.
# Levels compared on each model's held-out pairs (same simulators as Exp 1):
#   none · own explanation · free-form induced (5 drafts pooled, and per draft) · coefficient-informed text ·
#   LR ceiling (training-only rule, no simulator) · always the held-out majority answer
#   For GPT-5.2, Gemini 3 Flash and Gemma, none/own come from Exp 1's run (same pairs, prompts and simulators), and
#   Exp 1's written ground truth ("exp1_gt") is shown as a reference row.
# Writes to experiments/moral_machines/:
# - mm18_induce_results.csv: accuracy and NSG per model x level, pooled over simulators and per simulator, 95% CIs
# - mm18_induce_contrasts.csv: paired NSG differences (induced - own, coef - induced, rule - coef) with 95% CIs
# - mm18_induce_drafts_summary.csv: per draft: words, whether it mentions answer position (and where), accuracy, NSG
# NSG = (acc - acc_none) / (1 - acc_none). Intervals: bootstrap over held-out pairs (2,000 replicates).

# %%
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)

OUT = Path("experiments/moral_machines")
N_BOOT, SEED = 2000, 0
EXP1_MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
MODELS = ["GPT-5.2", "GPT-5 Mini", "GPT-5 Nano", "Claude Opus 4.5", "Claude Sonnet 4.5", "Claude Haiku 4.5",
          "Gemini 3 Pro", "Gemini 3 Flash", "Gemma 3 27B"]
LEVELS = ["none", "own", "induced", "coef", "exp1_gt"]
LABEL = {"none": "no explanation", "own": "model's own explanation", "induced": "free-form induced (5 drafts)",
         "coef": "coefficient-informed text", "exp1_gt": "Exp 1 written ground truth (reference)",
         "rule": "LR ceiling (no simulator)", "majority": "always the majority answer"}
SIM_SHORT = {"openrouter/openai/gpt-oss-20b": "gpt-oss-20b", "openrouter/google/gemini-3-flash-preview": "gemini-3-flash"}
# Answer position / prompt format: any mention in the draft's text or factors
POSITION = re.compile(r"answer (position|first|last|order|before|after|is (requested|asked|stated|given))|"
                      r"(before|after) (its |the )?reasoning|reason(ing|s)? (first|before|after)|"
                      r"position of the answer|order of (the )?(answer|response)|\bformat\b|\bprompt\b", re.I)

ind = pd.read_csv(OUT / "mm18_induce_predictions.csv")
e1 = pd.read_csv(OUT / "mm18_gt_predictions.csv").query("run == 'exp1' and condition in ['own', 'none', 'gt']")
e1 = e1.assign(condition=e1["condition"].replace({"gt": "exp1_gt"}))
pr = pd.concat([ind, e1], ignore_index=True)
pr["correct"] = (pr["answer"] == pr["reference_answer"]).astype(float)
pr.loc[pr["answer"].isna(), "correct"] = np.nan  # unparsed answers are dropped, as in Exp 1
pr["sim"] = pr["simulator"].map(SIM_SHORT)
pr["level"] = pr["condition"].str.replace(r"^induced_\d$", "induced", regex=True)
rule = pd.read_csv(OUT / "mm18_induce_heldout_rule.csv")
rule["rule_right"] = (rule["rule_answer"] == rule["reference_answer"]).astype(float)


def matrix(g, conds, col):
    """pair x condition: (correct count, parsed count) summed over the rows in g."""
    c = g.pivot_table(index="pair_id", columns=col, values="correct", aggfunc="sum").reindex(columns=conds)
    n = g.assign(ok=g["correct"].notna()).pivot_table(index="pair_id", columns=col, values="ok", aggfunc="sum").reindex(columns=conds)
    return c.fillna(0), n.fillna(0)


def acc_nsg(c, n):
    acc = c.sum(0) / n.sum(0)
    return acc, (acc - acc[0]) / (1 - acc[0])


rows, contrasts, drafts_rows = [], [], []
rng = np.random.default_rng(SEED)
for m in MODELS:
    gm = pr[pr["model"] == m]
    rm = rule[rule["model"] == m].set_index("pair_id")
    for scope, g in [("both simulators", gm)] + list(gm.groupby("sim")):
        c, n = matrix(g, LEVELS, "level")
        # extra columns: LR ceiling and majority answer, one "answer" per pair (no simulator)
        ref = g.drop_duplicates("pair_id").set_index("pair_id")["reference_answer"].reindex(c.index)
        maj = ref.value_counts().idxmax()
        c["rule"], n["rule"] = rm.loc[c.index, "rule_right"].to_numpy(), 1.0
        c["majority"], n["majority"] = (ref == maj).astype(float).to_numpy(), 1.0
        cols = list(c.columns)
        C, N = c.to_numpy(), n.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            acc, nsg = acc_nsg(C, N)
            boots = [acc_nsg(C[ix], N[ix]) for ix in (rng.integers(0, len(C), len(C)) for _ in range(N_BOOT))]
        ba, bn = np.array([b[0] for b in boots]), np.array([b[1] for b in boots])
        for k, lev in enumerate(cols):
            if N[:, k].sum() == 0:
                continue
            rows.append(dict(model=m, simulator=scope, level=lev, label=LABEL[lev], pairs=len(C), answers=int(N[:, k].sum()),
                             acc=acc[k], acc_lo=np.nanpercentile(ba[:, k], 2.5), acc_hi=np.nanpercentile(ba[:, k], 97.5),
                             nsg=nsg[k] if k else np.nan, nsg_lo=np.nanpercentile(bn[:, k], 2.5) if k else np.nan,
                             nsg_hi=np.nanpercentile(bn[:, k], 97.5) if k else np.nan))
        for a, b in [("induced", "own"), ("coef", "induced"), ("coef", "own"), ("rule", "coef"), ("induced", "exp1_gt")]:
            ia, ib = cols.index(a), cols.index(b)
            if N[:, ia].sum() == 0 or N[:, ib].sum() == 0:
                continue
            dif = bn[:, ia] - bn[:, ib]
            contrasts.append(dict(model=m, simulator=scope, contrast=f"{a} - {b}", nsg_diff=nsg[ia] - nsg[ib],
                                  lo=np.nanpercentile(dif, 2.5), hi=np.nanpercentile(dif, 97.5)))
    # per draft (pooled over simulators)
    dc, dn = matrix(gm[gm["condition"].str.startswith("induced_") | (gm["condition"] == "none")],
                    ["none"] + [f"induced_{k}" for k in range(1, 6)], "condition")
    a, s = acc_nsg(dc.to_numpy(), dn.to_numpy())
    for k in range(1, 6):
        drafts_rows.append(dict(model=m, draft=k, acc=a[k], nsg=s[k]))

res, con = pd.DataFrame(rows), pd.DataFrame(contrasts)

# Draft text checks: words and answer-position mentions (regex; snippets saved for a hand check)
dj = {m: v for m, v in json.loads((OUT / "mm18_induce_drafts.json").read_text()).items() if not m.startswith("_")}
meta = []
for m, ds in dj.items():
    for x in ds:
        blob = x["text"] + " || " + " | ".join(x["factors"])
        hits = [blob[max(0, h.start() - 60):h.end() + 60].replace("\n", " ") for h in POSITION.finditer(blob)]
        meta.append(dict(model=m, draft=x["draft"], words=x["words"], attempts=x.get("attempts"),
                         mentions_position=bool(hits), in_factors=bool(POSITION.search(" | ".join(x["factors"]))),
                         snippets=" … ".join(hits), text=x["text"], factors=" | ".join(x["factors"])))
summary = pd.DataFrame(meta).merge(pd.DataFrame(drafts_rows), on=["model", "draft"])

res.to_csv(OUT / "mm18_induce_results.csv", index=False)
con.to_csv(OUT / "mm18_induce_contrasts.csv", index=False)
summary.to_csv(OUT / "mm18_induce_drafts_summary.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 200)
    print(f"{len(ind)} new predictions, parsed {ind['answer'].notna().mean():.1%}\n")
    t = res[res["simulator"] == "both simulators"]
    print("Accuracy, pooled over both simulators:")
    print(t.pivot(index="level", columns="model", values="acc").reindex(LEVELS + ["rule", "majority"])[MODELS].map(
        lambda v: f"{v:.0%}" if pd.notna(v) else "").to_string(), "\n")
    print("NSG, pooled:")
    print(t.pivot(index="level", columns="model", values="nsg").reindex(LEVELS[1:] + ["rule", "majority"])[MODELS].map(
        lambda v: f"{v:+.0%}" if pd.notna(v) else "").to_string(), "\n")
    c = con[con["simulator"] == "both simulators"].copy()
    c["diff"] = c.apply(lambda r: f"{r.nsg_diff:+.0%} [{r.lo:+.0%}, {r.hi:+.0%}]", axis=1)
    print("Paired NSG differences (pooled):")
    print(c.pivot(index="contrast", columns="model", values="diff")[MODELS].fillna("").to_string(), "\n")
    print("Per draft:")
    s = summary.groupby("model").agg(drafts=("draft", "size"), mention_position=("mentions_position", "sum"),
                                     words_min=("words", "min"), words_max=("words", "max"), acc_min=("acc", "min"),
                                     acc_max=("acc", "max"), nsg_min=("nsg", "min"), nsg_max=("nsg", "max"),
                                     nsg_sd=("nsg", "std")).reindex(MODELS)
    print(s.round(3).to_string())
