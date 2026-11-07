"""Coverage audit of the Moral Machines pairs: which factors change, what Case 1 kills, what NSG can't evaluate.

Reads pairs_features.parquet (one row per unique pair, from build_tables.py) and writes, next to it:
    coverage_changes.csv     Table 1, one row per factor (n_alone, n_with_others, n_total, pct_total, status)
    coverage_combos.csv      top 10 combinations of changed factors
    coverage_case1.csv       Table 2, layout x theme: share where Case 1 kills the passengers / the theme's first group
    coverage_blindspots.csv  Table 3, factors with status never/only_mixed + always=true rows of Table 2 (fill in `why`)
    coverage_summary.md      one-page summary

No model calls and no text parsing: everything comes from existing columns. The five demographic factors are read
from the *_chars JSON columns (the parsed character lists). A demographic factor changed when, on either side, the
share of humans holding that attribute changed (e.g. gender: share female / share male), so adding one boy and one
girl to a 50/50 side is a size change, not a gender change. Pregnancy is its own factor and not part of status.

Usage (from the repo root):
    python -m moral_machines.coverage_audit [--input ...] [--out-dir ...] [--by-pair-set]
"""
import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pandas as pd

OUT = Path("moral_machines/output")
FACTORS = ["law", "species", "margin", "size", "gender", "age", "fitness", "status", "pregnancy",
           "swerve side", "passenger side", "case order"]
STRUCTURAL = ["swerve side", "passenger side", "case order"]
ANIMALS = {"dog", "cat"}
FEMALE = {"woman", "elderly woman", "girl", "large woman", "pregnant woman",
          "female executive", "female athlete", "female doctor"}
MALE = {"man", "elderly man", "boy", "large man", "male executive", "male athlete", "male doctor"}
# theme -> what its "first" group is (scenario_features.infer_theme)
FIRST_GROUP = {"species": "human", "social_value": "lower social value", "gender": "female", "age": "younger",
               "fitness": "lower fitness", "utilitarianism": "fewer"}


def attribute(char, attr):
    """The marked value of one attribute for one character type, or None if neutral."""
    if attr == "gender":
        return "female" if char in FEMALE else "male" if char in MALE else None
    if attr == "age":
        return "child" if char in {"baby", "boy", "girl"} else "elderly" if char.startswith("elderly") else None
    if attr == "fitness":
        return "large" if char.startswith("large") else "athlete" if char.endswith("athlete") else None
    if attr == "status":
        return ("high" if char.endswith(("executive", "doctor"))
                else "low" if char in {"homeless person", "criminal"} else None)
    return "pregnant" if char == "pregnant woman" else None  # pregnancy


def shares(chars_json, attr):
    """Share of the side's humans holding each marked value of `attr`."""
    humans = [(c, n) for c, n in json.loads(chars_json) if c not in ANIMALS]
    total = sum(n for _, n in humans)
    out = Counter()
    for c, n in humans:
        if attribute(c, attr):
            out[attribute(c, attr)] += Fraction(n, total)
    return out


def factor_matrix(df):
    """One boolean column per factor: did it change between original and counterfactual?"""
    m = pd.DataFrame(index=df.index)
    for f in ["law", "species", "margin", "size"]:
        m[f] = df[f"{f}_changed"].astype(bool)
    for attr in ["gender", "age", "fitness", "status", "pregnancy"]:
        m[attr] = [any(shares(r[f"orig_{s}_chars"], attr) != shares(r[f"cf_{s}_chars"], attr) for s in "AB")
                   for _, r in df.iterrows()]
    m["swerve side"] = df["case1_action_changed"].astype(bool)
    m["passenger side"] = df["layout_changed"].astype(bool)
    m["case order"] = df["orig_case1_kills"] != df["cf_case1_kills"]
    return m[FACTORS]


def status(n_alone, n_total):
    if n_total == 0:
        return "never"
    if n_alone == 0:
        return "only_mixed"
    return "alone" if n_alone >= 20 else "rarely_alone"


def changes_table(m):
    alone = m.sum(axis=1) == 1
    rows = []
    for f in FACTORS:
        n_alone, n_total = int((m[f] & alone).sum()), int(m[f].sum())
        rows.append({"factor": f, "n_alone": n_alone, "n_with_others": n_total - n_alone, "n_total": n_total,
                     "pct_total": round(100 * n_total / max(len(m), 1), 1), "status": status(n_alone, n_total)})
    return pd.DataFrame(rows)


def combos_table(m, top=10):
    labels = [" + ".join(f for f in FACTORS if r[f]) or "(none)" for _, r in m.iterrows()]
    counts = Counter(labels).most_common(top)
    return pd.DataFrame(counts, columns=["combination", "n"])


def case1_table(df):
    """Layout x theme: share of scenarios where Case 1 kills the passengers / the theme's first group."""
    rows = []
    for which in ["orig", "cf"]:
        for (layout, theme), g in df.groupby([f"{which}_layout", f"{which}_theme"]):
            kills = g[f"{which}_case1_kills"]
            pass_pct = round(100 * (kills == "A").mean(), 1) if layout == "car_vs_ped" else None
            first_pct = round(100 * (kills == g[f"{which}_theme_group_first_side"]).mean(), 1)
            always = any(p in (0.0, 100.0) for p in (pass_pct, first_pct) if p is not None)
            rows.append({"scenarios": "original" if which == "orig" else "counterfactual", "layout": layout,
                         "theme": theme, "first_group": FIRST_GROUP.get(theme, ""), "n": len(g),
                         "pct_case1_kills_passengers": pass_pct, "pct_case1_kills_first_group": first_pct,
                         "always": always})
    return pd.DataFrame(rows, columns=["scenarios", "layout", "theme", "first_group", "n", "pct_case1_kills_passengers",
                                       "pct_case1_kills_first_group", "always"])


def blindspots_table(changes, case1):
    rows = [{"kind": "factor", "item": r.factor, "detail": f"status={r.status}, n_total={r.n_total}", "why": ""}
            for r in changes.itertuples() if r.status in ("never", "only_mixed")]
    for r in case1[case1["always"]].itertuples():
        what = [f"passengers {r.pct_case1_kills_passengers:g}%"] if pd.notna(r.pct_case1_kills_passengers) else []
        what.append(f"first group ({r.first_group}) {r.pct_case1_kills_first_group:g}%")
        rows.append({"kind": f"case1 ({r.scenarios})", "item": f"{r.layout} x {r.theme}",
                     "detail": "Case 1 kills " + ", ".join(what), "why": ""})
    return pd.DataFrame(rows, columns=["kind", "item", "detail", "why"])


def headline(df, m):
    return {"pairs": len(df), "pct_law": round(100 * m["law"].mean(), 1),
            "pct_structural": round(100 * m[STRUCTURAL].any(axis=1).mean(), 1),
            "pct_single_factor": round(100 * (m.sum(axis=1) == 1).mean(), 1)}


def audit(df):
    m = factor_matrix(df)
    changes, case1 = changes_table(m), case1_table(df)
    return {"changes": changes, "combos": combos_table(m), "case1": case1,
            "blindspots": blindspots_table(changes, case1), "headline": headline(df, m)}


def checks(res):
    """Sanity checks on the overall audit; returns a list of warnings (empty = all passed)."""
    h, ch, c1 = res["headline"], res["changes"], res["case1"]
    car = c1[c1["layout"] == "car_vs_ped"]
    tests = [
        (h["pairs"] == 1858, f"pair count is {h['pairs']} (expected 1,858)"),
        (abs(h["pct_law"] - 26) <= 2, f"law changes in {h['pct_law']}% of pairs (expected about 26%)"),
        (h["pct_structural"] == 0, f"structural changes in {h['pct_structural']}% of pairs (expected 0)"),
        (abs(h["pct_single_factor"] - 6) <= 2, f"single-factor pairs at {h['pct_single_factor']}% (expected about 6%)"),
        ((ch["n_alone"] + ch["n_with_others"] == ch["n_total"]).all(), "Table 1: n_alone + n_with_others = n_total"),
        ((car["pct_case1_kills_passengers"] == 100).all(), "Table 2: Case 1 always kills the passengers in car rows"),
    ]
    for ok, msg in tests:
        print(("  ok    " if ok else "  WARN  ") + msg)
    return [msg for ok, msg in tests if not ok]


def md_table(df):
    df = df.fillna("")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summary_md(results, warnings):
    res = results["all"]
    h = pd.DataFrame([{"pair_set": k, **r["headline"]} for k, r in results.items()])
    case1 = res["case1"][res["case1"]["scenarios"] == "original"].drop(columns="scenarios")
    parts = [
        "# Moral Machines coverage audit", "",
        f"**{res['headline']['pairs']} pairs.** Law changes in {res['headline']['pct_law']}%, "
        f"structural changes (swerve side / passenger side / case order) in {res['headline']['pct_structural']}%, "
        f"single-factor pairs {res['headline']['pct_single_factor']}%.", "",
        md_table(h), "",
        "## 1. What changes", "",
        "alone = only factor changed (all 12 factors). Margin and size are exclusive by construction (edit_flags).", "",
        md_table(res["changes"]), "",
        "Top 5 combinations: " + "; ".join(f"{r.combination} ({r.n})" for r in res["combos"].head(5).itertuples()), "",
        "## 2. What Case 1 kills (original scenarios)", "",
        "always = a share is 0% or 100%: a preference for the Case 1 label can't be separated from that content. "
        "two_lanes 100% comes from the generator (tabular_utils: set_1, the theme's first group, is always Case 1's "
        "victims; only car_vs_ped swaps groups), and infer_theme assumes the same direction for two_lanes.", "",
        md_table(case1), "",
        "## 3. Factors NSG can't evaluate", "",
        md_table(res["blindspots"][res["blindspots"]["kind"] == "factor"].drop(columns=["kind", "why"])), "",
        f"Plus {int(case1['always'].sum())} of {len(case1)} layout x theme rows of Table 2 with always = true. "
        "All rows (including counterfactuals) and the `why` column are in coverage_blindspots.csv.", "",
    ]
    if warnings:
        parts += ["**Check warnings:** " + "; ".join(warnings), ""]
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=str(OUT / "pairs_features.parquet"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--by-pair-set", action="store_true", help="also repeat every table for pair set A and B")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    subsets = {"all": df}
    if args.by_pair_set:
        subsets.update(A=df[df["in_set_A"]], B=df[df["in_set_B"]])
    results = {name: audit(d) for name, d in subsets.items()}

    print("Checks (all pairs):")
    warnings = checks(results["all"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for table in ["changes", "combos", "case1", "blindspots"]:
        t = pd.concat([r[table].assign(pair_set=name) for name, r in results.items()], ignore_index=True)
        t = t[["pair_set"] + [c for c in t.columns if c != "pair_set"]]
        t.to_csv(out / f"coverage_{table}.csv", index=False)
    (out / "coverage_summary.md").write_text(summary_md(results, warnings))
    print(f"Wrote coverage_*.csv and coverage_summary.md to {out}")
    if warnings:
        print("WARNING: some checks failed (outputs were still written):\n  - " + "\n  - ".join(warnings))


if __name__ == "__main__":
    main()
