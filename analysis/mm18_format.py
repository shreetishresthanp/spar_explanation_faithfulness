# %% [markdown]
# # Gemma answer-format test: analysis
#
# Run from the repo root after `python analysis/mm18_format_run.py --full`: `python analysis/mm18_format.py`.
# Writes experiments/moral_machines/mm18_format_results.csv (one row per metric) and mm18_format_flips.csv
# (scenarios whose majority answer changes with the format, with Gemma's explanations, for spot-checking).
# Intervals: bootstrap over scenarios (2,000 replicates).

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)

OUT = Path("experiments/moral_machines")
N_BOOT, SEED = 2000, 0
p = pd.read_csv(OUT / "mm18_format_predictions.csv")
import argparse
import re
ap = argparse.ArgumentParser()
ap.add_argument("--strict", action="store_true", help="use only answers with an [ANSWER] tag (the repo's parser)")
args, _ = ap.parse_known_args()
p = p[p["set"] != "smoke"].copy()
# Gemma often answers answer-first prompts with a bare "CASE 2" first line and no [ANSWER] tag; the repo parser drops
# those (120 of 360 answer-first calls, 92 of them Case 2), which inflates the answer-first Case 1 rate. Lenient
# parsing (default) takes that leading answer; --strict reproduces the tag-only view.
p["answer_strict"] = p["answer"]
if not args.strict:
    lead = p["content"].fillna("").str.extract(r"^\s*\**\s*(CASE [12])\b", flags=re.I)[0].str.upper()
    p["answer"] = p["answer"].fillna(lead)
p["case1"] = np.where(p["answer"].isna(), np.nan, (p["answer"] == "CASE 1").astype(float))
rng = np.random.default_rng(SEED)
rows = []


def boot(per_scenario, stat):
    """per_scenario: DataFrame indexed by scenario; stat: function of it -> float. Returns (value, lo, hi)."""
    v = stat(per_scenario)
    idx = per_scenario.index.to_numpy()
    b = [stat(per_scenario.loc[rng.choice(idx, len(idx))]) for _ in range(N_BOOT)]
    return v, *np.nanpercentile(b, [2.5, 97.5])


def add(group, metric, val, n, note=""):
    v, lo, hi = val if isinstance(val, tuple) else (val, np.nan, np.nan)
    rows.append(dict(group=group, metric=metric, value=v, lo=lo, hi=hi, n=n, note=note))


for set_, label in (("main", "Gemma 3 27B (fresh)"), ("control", "Gemini 3 Flash control (fresh)")):
    g = p[p["set"] == set_]
    if g.empty:
        continue
    w = g.pivot_table(index="idx", columns="fmt", values="case1", aggfunc="mean")  # share Case 1 per scenario x format
    w = w.dropna()
    for fmt in ("answer_first", "answer_last"):
        add(label, f"Case 1 rate, {fmt.replace('_', ' ')}", boot(w, lambda x, f=fmt: x[f].mean()), len(w))
    add(label, "difference (last - first)", boot(w, lambda x: (x["answer_last"] - x["answer_first"]).mean()), len(w))
    maj = (w > 0.5).astype(int).where(w != 0.5)  # majority answer per format (ties dropped)
    m = maj.dropna()
    add(label, "scenarios whose majority answer flips with the format", boot(m, lambda x: (x["answer_first"] != x["answer_last"]).mean()), len(m))
    flips = m[m["answer_first"] != m["answer_last"]]
    if len(flips):
        add(label, "of those flips, share going to Case 1 when answering last", (flips["answer_last"] == 1).mean(), len(flips))
    if set_ == "main":
        # format effect beyond sampling noise: disagreement across formats vs within a format (over all sample pairs)
        def disagree(gg):
            a = gg.groupby(["idx", "fmt"])["case1"].apply(list).unstack()
            within, across = [], []
            for _, r in a.iterrows():
                for f in ("answer_first", "answer_last"):
                    xs = [x for x in r[f] if not np.isnan(x)]
                    within += [xs[i] != xs[j] for i in range(len(xs)) for j in range(i + 1, len(xs))]
                f1 = [x for x in r["answer_first"] if not np.isnan(x)]; f2 = [x for x in r["answer_last"] if not np.isnan(x)]
                across += [x != y for x in f1 for y in f2]
            return np.mean(within), np.mean(across)
        wi, ac = disagree(g)
        add(label, "two samples disagree: same format", wi, len(w), "sampling noise")
        add(label, "two samples disagree: different formats", ac, len(w), "noise + format effect")
        cons = g.groupby(["idx", "fmt"])["case1"].agg(lambda s: s.dropna().nunique() == 1).unstack()
        for fmt in ("answer_first", "answer_last"):
            add(label, f"all 3 samples agree, {fmt.replace('_', ' ')}", cons[fmt].mean(), len(cons))
        ex = g[g["idx"].isin(flips.index)].sort_values(["idx", "fmt", "sample"])
        ex[["idx", "theme", "in_car", "fmt", "sample", "answer", "explanation"]].to_csv(OUT / "mm18_format_flips.csv", index=False)

c = p[p["set"] == "calibration"].copy()
if len(c):
    c["agree_stored"] = c["answer"] == c["stored_answer"]
    add("calibration (20 paper prompts, unchanged)", "fresh answer = paper's stored answer", c["agree_stored"].mean(), len(c))
    pairs = c.groupby("idx")["answer"].apply(lambda s: np.mean([a == b for i, a in enumerate(s) for b in list(s)[i + 1:]]))
    add("calibration (20 paper prompts, unchanged)", "two fresh samples agree", pairs.mean(), len(pairs))
    for fmt, gg in c.groupby("fmt"):
        add("calibration (20 paper prompts, unchanged)", f"Case 1 rate fresh vs stored, {fmt.replace('_', ' ')}",
            (gg["answer"] == "CASE 1").mean(), len(gg), f"stored: {(gg['stored_answer'] == 'CASE 1').mean():.2f}")

res = pd.DataFrame(rows)
res["parsing"] = "strict" if args.strict else "lenient"
res.to_csv(OUT / ("mm18_format_results_strict.csv" if args.strict else "mm18_format_results.csv"), index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(f"{len(p)} fresh calls, parsed {p['answer'].notna().mean():.1%}\n")
    print(res.round(3).to_string(index=False))
