# %% [markdown]
# # Automatic ground-truth explanations: a depth-3 decision tree per model, rendered as if-then sentences
#
# Run from the repo root after `python analysis/mm18_gt_prep.py`: `python analysis/mm18_gt_tree.py`.
# No judgment involved: the tree is fitted on TRAINING decisions only (depth fixed at 3 in advance, for readability),
# and each leaf becomes one sentence with its training rate. Adds "tree_text"/"tree_factors" to
# experiments/moral_machines/mm18_gt_explanations.json and writes mm18_gt_tree_summary.csv.

# %%
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import mm18_rule as R  # noqa: E402

OUT = Path("experiments/moral_machines")
DEPTH, MIN_LEAF, SEED = 3, 20, 0
split = pd.read_csv(OUT / "mm18_gt_split.csv")
GROUPS = {"species": ("humans", "animals"), "social_value": ("lower-status people", "higher-status people"),
          "gender": ("women", "men"), "age": ("younger people", "older people"), "fitness": ("less fit people", "fitter people"),
          "utilitarianism": ("smaller group", "larger group")}


def phrase(f, vals, allvals):
    """Short plain-English condition for a feature restricted to the value set vals (allvals: all training values)."""
    pos, neg = all(v > 0 for v in vals), all(v < 0 for v in vals)
    nonneg, nonpos = all(v >= 0 for v in vals), all(v <= 0 for v in vals)
    if f == "c1_kills_passengers":
        return "Case 1 kills the passengers" if pos else "both groups are pedestrians"
    if f == "c1_swerves":
        return "Case 1 swerves" if pos else "Case 1 stays in lane"
    if f == "answer_first":
        return "answer asked first" if pos else "answer asked last"
    if f == "c1_spares_lawful":  # = light(Case 2's victims) - light(Case 1's victims); green +1, red -1, none 0
        desc = {-2: "Case 1 hits lawful crossers to spare jaywalkers", -1: "Case 1 kills the passengers to spare jaywalkers",
                0: "no traffic signals", 1: "Case 1 kills the passengers to spare lawful crossers",
                2: "Case 1 hits jaywalkers to spare lawful crossers"}
        if len(vals) == 1:
            return desc[vals[0]]
        if set(vals) == {1, 2}:
            return "Case 1 spares lawful crossers"
        if set(vals) == {-2, -1}:
            return "Case 1 spares jaywalkers"
        if nonneg:
            return "Case 1 doesn't spare jaywalkers"
        if nonpos:
            return "Case 1 doesn't spare lawful crossers"
        missing = [v for v in allvals if v not in vals]
        if len(missing) == 1:
            return f"not ({desc[missing[0]]})"
        return " or ".join(desc[v] for v in vals)
    if f in ("lives_saved", "pets_saved"):
        what = "people" if f == "lives_saved" else "animals"
        if vals == [0]:
            return f"equal numbers of {what} die"
        if pos or neg or nonneg or nonpos:
            return ("Case 1 kills fewer " if pos else "Case 1 kills more " if neg else
                                                "Case 1 kills no more " if nonneg else "Case 1 kills no fewer ") + what
        # mixed range: one side is open (all training values), so state the other bound
        k_lo, k_hi = -max(vals), -min(vals)  # how many more Case 1 kills
        return (f"Case 1 kills at most {-k_lo} fewer {what}" if min(vals) == min(allvals)
                else f"Case 1 kills at most {k_hi} more {what}")
    if f.startswith("theme_"):
        a, b = GROUPS[f[6:]]
        return (f"Case 1 kills the {a}" if vals == [1] else f"Case 1 kills the {b}" if vals == [-1] else
                f"not a {a}-vs-{b} scenario" if vals == [0] else
                f"Case 1 doesn't kill the {b} to spare the {a}" if nonneg else f"Case 1 doesn't kill the {a} to spare the {b}")
    return f"{f} in {vals}"


def leaves(tree, cols):
    """(conditions, n, share choosing Case 1) for each leaf, as feature ranges along the path."""
    t, out = tree.tree_, []

    def walk(node, ranges):
        if t.children_left[node] == -1:
            n = int(t.n_node_samples[node])
            v = t.value[node][0]
            out.append((ranges, n, v[1] / v.sum()))
            return
        f, th = cols[t.feature[node]], t.threshold[node]
        lo, hi = ranges.get(f, (-np.inf, np.inf))
        walk(t.children_left[node], {**ranges, f: (lo, min(hi, th))})
        walk(t.children_right[node], {**ranges, f: (max(lo, th), hi)})

    walk(0, {})
    return out


expl = json.loads((OUT / "mm18_gt_explanations.json").read_text())
summary = []
for m in [k for k in expl if not k.startswith("_")]:
    g = R.d[R.d["model"] == m].reset_index(drop=True)
    sp = split[split["model"] == m].set_index("pair_id")["split"]
    dec = R.decisions(g)
    dec["split"] = dec["pair_id"].map(sp)
    tr = dec[dec["split"] == "train"]
    cols = [c for c in R.FEATURES if tr[c].std() > 0]
    values = {c: sorted(tr[c].unique()) for c in cols}
    tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=SEED).fit(tr[cols], tr["y"])
    rules = []
    for ranges, n, p1 in sorted(leaves(tree, cols), key=lambda l: -l[1]):
        cond = "; ".join(phrase(f, [v for v in values[f] if lo < v <= hi], values[f]) for f, (lo, hi) in ranges.items())
        choice, rate = ("Case 1", p1) if p1 >= 0.5 else ("Case 2", 1 - p1)
        rules.append(f"- {cond} → {choice} ({rate:.0%} of {n})")
    text = ("Rules from this evaluator's past decisions (situation → usual choice, how often):\n" + "\n".join(rules))
    imp = pd.Series(tree.feature_importances_, index=cols).sort_values(ascending=False)
    factors = [R.FEATURE_LABEL[f] for f in imp[imp > 0].index[:4]]
    expl[m]["tree_text"], expl[m]["tree_factors"] = text, factors
    te = dec[(dec["split"] == "heldout") & (dec["side"] == "c")]
    summary.append(dict(model=m, leaves=tree.get_n_leaves(), words=len(text.split()),
                        train_acc=tree.score(tr[cols], tr["y"]), heldout_cf_acc=tree.score(te[cols], te["y"])))
expl["_note"] = (expl["_note"].split(" Tree texts:")[0] + " Tree texts: generated automatically by analysis/mm18_gt_tree.py "
                 f"(depth-{DEPTH} decision tree on training decisions; no judgment).")
(OUT / "mm18_gt_explanations.json").write_text(json.dumps(expl, indent=2, ensure_ascii=False))
pd.DataFrame(summary).to_csv(OUT / "mm18_gt_tree_summary.csv", index=False)

if __name__ == "__main__":
    print(pd.DataFrame(summary).round(3).to_string(index=False), "\n")
    for m in summary:
        print(f"=== {m['model']}\n{expl[m['model']]['tree_text']}\nfactors: {expl[m['model']]['tree_factors']}\n")
