"""Step 1: discovery sample = the paper's original Moral Machines scenarios + a Case-order mirror of each.

    python v2/discovery/make_scenarios.py                    # -> data/scenarios.parquet + scenarios_report.html
    python v2/discovery/make_scenarios.py --train-rows 1000 --floor 37
    pytest v2/discovery/test_make_scenarios.py -q

Source: the original scenarios of the MM pairs that both the Claude 4.5 and the GPT-5.2 prediction runs used
(data_parquet/, 993 pairs). Prompts are verbatim from there (they must agree across the files). The same rows serve both
reference models. The paper's counterfactual scenarios are deliberately left out: the writer only needs scenario -> answer,
pair members are near-duplicates, and keeping them unseen leaves the paper's pairs clean for later evaluation.
Factors come from data/raw/moral_machines_raw.csv, joined on the exact scenario text.

Mirror: the same two outcomes with Case 1 and Case 2 swapped and nothing else changed (the paper ties the Case 1 label
to content: in car scenarios Case 1 is always the passengers; on foot it is the first group, e.g. the women). Its prompt is
built with the paper's template and answer position; for every unmirrored scenario that rebuild must equal the stored
prompt and re-rendering from factors must equal the stored text, or the build fails.

Sampling unit: an original scenario -> 2 rows (it and its mirror), in one split. Train scenarios per theme get --floor
each, the rest in proportion to availability; test scenarios per theme = round(train / 3), i.e. 25% of rows.
--train-rows must be even. Default floor 37 is the most species (50 originals) can supply with its test share.
"""
import argparse
import ast
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from factorial_scenarios import GROUP_LABELS, ROOT, THEMES, render
from src.templates.moral_machines import MoralMachines

HERE = Path(__file__).parent
SOURCES = [ROOT / "data_parquet/claude_4_5_predictions.parquet", ROOT / "data_parquet/gpt_5_2_itc_predictions.parquet"]
RAW = ROOT / "data/raw/moral_machines_raw.csv"
KEY = ["original_question", "counterfactual_question"]
COLS = KEY + ["original_dataset", "original_question_prompt", "counterfactual_question_prompt", "original_answer_first",
              "original_question_idx"]


def load_pairs(paths=SOURCES):
    """Pairs present in every source file, with prompts that must be identical across them."""
    frames = []
    for p in paths:
        d = pd.read_parquet(p, columns=COLS)
        d = d[d.original_dataset == "moral_machines"].drop_duplicates(KEY).set_index(KEY)
        frames.append(d)
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    check = ["original_question_prompt", "counterfactual_question_prompt", "original_answer_first"]
    for p, f in zip(paths[1:], frames[1:]):
        if not f.loc[common, check].equals(frames[0].loc[common, check]):
            raise ValueError(f"prompts or answer_first differ between {paths[0].name} and {p.name}")
    return frames[0].loc[common].reset_index().sort_values("original_question_idx", ignore_index=True)


def load_raw(path=RAW):
    raw = pd.read_csv(path).drop_duplicates("prompt").set_index("prompt")
    return {q: dict(theme=r.scenario_dimension, in_car=bool(r.is_in_car), interv=bool(r.is_interventionism),
                    law=bool(r.is_law), c1=Counter(ast.literal_eval(r.count_dict_1)),
                    c2=Counter(ast.literal_eval(r.count_dict_2)),
                    groups=ast.literal_eval(r.scenario_dimension_group_type),
                    lights=["none" if v == "NA" else v for v in ast.literal_eval(r.traffic_light_pattern)])
            for q, r in raw.iterrows()}


def scenario_row(q, stored_prompt, answer_first, f, mirrored):
    args = dict(x=f["c1"], y=f["c2"], x_swerve=f["interv"], in_car=f["in_car"], x_passengers=True,
                light_x=f["lights"][0], light_y=f["lights"][1])
    if render(**args, x_case1=True) != q:
        raise ValueError(f"re-rendered text differs from the paper's scenario:\n{q[:300]}")
    if MoralMachines.create_reference_prompt(q, answer_last=not answer_first) != stored_prompt:
        raise ValueError(f"template rebuild differs from the stored prompt:\n{q[:300]}")
    a, b = (1, 0) if mirrored else (0, 1)  # which of the paper's cases is shown as Case 1 / Case 2
    vic, grp, lit = [f["c1"], f["c2"]], f["groups"], f["lights"]
    question = render(**args, x_case1=not mirrored)
    prompt = MoralMachines.create_reference_prompt(question, answer_last=not answer_first) if mirrored else stored_prompt
    return dict(theme=f["theme"], in_car=f["in_car"], law=f["law"], mirrored=mirrored, answer_first=answer_first,
                case1_victims=json.dumps(dict(vic[a])), case2_victims=json.dumps(dict(vic[b])),
                case1_n=sum(vic[a].values()), case2_n=sum(vic[b].values()), case1_group=grp[a], case2_group=grp[b],
                case1_light=lit[a], case2_light=lit[b], case1_swerve=f["interv"] != mirrored,
                case1_passengers=f["in_car"] and not mirrored, question=question, prompt=prompt)


def allocate(available, n_train, floor, test_frac=0.25):
    """Train/test scenarios per theme: floor each, then proportional to availability (each next one to the theme
    with the highest target / (allocated + 1), so themes above the floor share one scaling factor)."""
    ratio = test_frac / (1 - test_frac)
    test_of = lambda t: int(round(t * ratio))  # noqa: E731
    cap = {k: max(t for t in range(n + 1) if t + test_of(t) <= n) for k, n in available.items()}
    if floor * len(available) > n_train:
        raise ValueError(f"floor {floor} x {len(available)} themes exceeds {n_train} train scenarios")
    if short := {k: c for k, c in cap.items() if c < floor}:
        raise ValueError(f"floor {floor} exceeds what these themes can supply: {short}")
    if sum(cap.values()) < n_train:
        raise ValueError(f"only {sum(cap.values())} train scenarios available, asked for {n_train}")
    train = {k: floor for k in available}
    target = {k: n_train * n / sum(available.values()) for k, n in available.items()}
    for _ in range(n_train - floor * len(available)):
        k = max((k for k in train if train[k] < cap[k]), key=lambda k: target[k] / (train[k] + 1))  # same scaling
        train[k] += 1
    return pd.DataFrame({"available": available, "proportional": pd.Series(target).round(1), "train": train,
                         "test": {k: test_of(t) for k, t in train.items()}})


def build(pairs, raw, train_rows=1000, floor=37, test_frac=0.25, seed=0):
    if train_rows <= 0 or train_rows % 2:
        raise ValueError(f"train_rows must be a positive even number (2 rows per scenario), got {train_rows}")
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    for q in pairs.original_question:
        if q not in raw:
            raise KeyError(f"no factors in the raw CSV for scenario: {q[:120]!r}")
    theme = pairs.original_question.map(lambda q: raw[q]["theme"])
    alloc = allocate(theme.value_counts().reindex(THEMES, fill_value=0).to_dict(), train_rows // 2, floor, test_frac)
    rng = np.random.default_rng(seed)
    rows = []
    for t, n_train, n_test in zip(alloc.index, alloc.train, alloc.test):
        idx = rng.permutation(np.flatnonzero(theme == t))
        for split, chosen in (("train", idx[:n_train]), ("test", idx[n_train: n_train + n_test])):
            for i in chosen:
                p = pairs.iloc[i]
                for mirrored in (False, True):
                    rows.append(dict(pair_id=int(p.original_question_idx), split=split, **scenario_row(
                        p.original_question, p.original_question_prompt, bool(p.original_answer_first),
                        raw[p.original_question], mirrored)))
    out = pd.DataFrame(rows).sort_values(["pair_id", "mirrored"], ignore_index=True)
    if out.question.duplicated().any():
        raise ValueError("duplicate scenario texts")
    out.insert(0, "scenario_id", np.arange(len(out)))
    out.insert(1, "base_id", out.scenario_id // 2)
    return out, alloc


# --- report --------------------------------------------------------------------------------------------------------
def table(df, caption, index=True):
    return f"<h3>{html.escape(caption)}</h3>" + df.to_html(border=0, index=index, float_format=lambda v: f"{v:.2f}")


def report(df, alloc, n_pairs, args):
    a = alloc.copy()
    a["train_rows"], a["test_rows"] = a.train * 2, a.test * 2
    a["share_of_train_rows"] = a.train_rows / a.train_rows.sum()
    a.loc["total"] = a.sum(numeric_only=True)
    a = a.astype({c: int for c in ["available", "train", "test", "train_rows", "test_rows"]})

    first = {t: g[0] for t, g in GROUP_LABELS.items()}
    df = df.assign(case1_first_group=[r.case1_group == first[r.theme] for r in df.itertuples()])
    mirror = df.pivot_table(index="theme", columns="mirrored", values="case1_first_group", aggfunc="mean")
    mirror["both"] = df.groupby("theme").case1_first_group.mean()
    mirror.columns = ["paper (mirrored=False)", "mirror (mirrored=True)", "both"]
    mirror.insert(0, "first group", pd.Series(first))
    car = df[df.in_car].groupby("mirrored").case1_passengers.mean().rename("Case 1 = passengers").to_frame()
    rows_split = pd.crosstab(df.split, df.mirrored, margins=True)
    af = pd.crosstab(df.mirrored, df.answer_first)
    themes = pd.crosstab(df.theme, [df.split, df.mirrored], margins=True)

    ex = df[df.split == "train"].iloc[:2]
    example = "".join(f"<div><h4>{'mirror' if r.mirrored else 'paper'} (scenario {r.scenario_id})</h4><pre>"
                      f"{html.escape(r.question)}</pre></div>" for r in ex.itertuples())
    css = ("body{background:#fcfcfb;color:#0b0b0b;font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0}"
           "main{max-width:1000px;margin:0 auto;padding:24px 16px}h2{margin-top:32px;border-top:1px solid #e4e3df;"
           "padding-top:12px}table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:6px 0 14px}"
           "th,td{padding:3px 10px;border-bottom:1px solid #e4e3df;text-align:right}.muted{color:#52514e}"
           ".ex{display:grid;grid-template-columns:1fr 1fr;gap:12px}pre{white-space:pre-wrap;background:#fff;"
           "border:1px solid #e4e3df;padding:8px;font-size:12px}@media(max-width:700px){.ex{grid-template-columns:1fr}}")
    body = f"""<h1>Discovery sample check</h1>
<p class="muted">{len(df):,} rows = {df.base_id.nunique()} original scenarios x 2 (the paper's and its Case-order
mirror). Drawn from the original scenarios of the {n_pairs} MM pairs used by both {', '.join(p.name for p in SOURCES)};
unmirrored prompts are verbatim from those files; the paper's counterfactual scenarios are not used. Train rows:
{args.train_rows}; floor: {args.floor} train scenarios per theme; seed {args.seed}.</p>
<h2>1. Theme breakdown</h2>
<p class="muted">Counts are original scenarios (2 rows each). "proportional" = train scenarios each theme would get with
no floor.</p>
{table(a, "Allocation per theme")}
{table(themes, "Rows per theme x split x mirrored")}
<h2>2. Mirror breakdown</h2>
{table(rows_split, "Rows per split x mirrored")}
{table(mirror, "Share of rows where Case 1 kills the theme's first group (paper: tied to the label, except car scenarios swap at random; with mirrors 'both' must be exactly 0.50)")}
{table(car, "Car scenarios: share where Case 1 is the passengers (paper: always 1.00)")}
{table(af, "answer_first x mirrored (should be identical rows)")}
<h2>3. Example: a paper scenario and its mirror</h2><div class="ex">{example}</div>"""
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Discovery Sample Check</title><style>{css}</style>"
            f"</head><body><main>{body}</main></body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-rows", type=int, default=1000)
    ap.add_argument("--floor", type=int, default=37, help="minimum train scenarios per theme")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "scenarios.parquet")
    ap.add_argument("--report", type=Path, default=HERE / "scenarios_report.html")
    a = ap.parse_args()
    pairs = load_pairs()
    df, alloc = build(pairs, load_raw(), a.train_rows, a.floor, a.test_frac, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    a.report.write_text(report(df, alloc, len(pairs), a))
    print(f"wrote {len(df)} rows to {a.out}\nwrote {a.report}")
    print(alloc.to_string())


if __name__ == "__main__":
    main()
