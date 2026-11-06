"""Per-predictor accuracy breakdowns on Moral Machines, one section per prediction parquet.

Run from the repo root:
    conda run -n faithfulness-env python v2/analysis/mm_predictor_breakdown.py                  # thinking in {medium, high}
    conda run -n faithfulness-env python v2/analysis/mm_predictor_breakdown.py --thinking all   # no thinking filter
    conda run -n faithfulness-env python v2/analysis/mm_predictor_breakdown.py --thinking high  # one level

For every parquet in data_parquet/ it keeps rows with original_dataset == "moral_machines" and
original_reference_response_model_info_thinking in --thinking (default: medium high; files with no such rows are skipped and
listed on the page). For each predictor in
the predictor_names / predictor_answers lists, computes:
  * accuracy with and without the reference model's explanation (target = reference model's counterfactual answer)
  * the "copy the original answer" baseline (always predict the reference model's answer on the original question)
  * how often the predictor itself copies the original answer
  * accuracy split by whether the reference model kept or flipped its answer on the counterfactual
  * accuracy split by answer format (original_answer_first True vs False)

Row filtering follows analysis_scripts/analyze_simulatability.py: a (row, predictor) pair is dropped when the reference
answer on the original or the counterfactual is missing/invalid, or when either predictor answer is missing/invalid.

Writes v2/analysis/mm_predictor_breakdown_thinking_<values>.html and .csv (no suffix for --thinking all).
"""
import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data_parquet"
OUT_DIR = Path(__file__).resolve().parent
THINKING_COL = "original_reference_response_model_info_thinking"

VALID = {"CASE 1", "CASE 2"}
P = "counterfactual_predictor_response_"
COLS = [
    "original_dataset", "original_answer_first", "original_question_idx", "original_reference_response_answer",
    "original_reference_response_model_info_model", THINKING_COL, "counterfactual_reference_response_answer",
    f"{P}with_explanation_predictor_names", f"{P}with_explanation_predictor_answers",
    f"{P}without_explanation_predictor_names", f"{P}without_explanation_predictor_answers",
]

# Validated categorical slots (dataviz reference palette); text stays in ink colours.
C_WITH, C_WITHOUT, C_AF, C_EF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_INK, C_MUTED, C_GRID, C_BASE = "#0b0b0b", "#52514e", "#e4e3df", "#52514e"
LAYOUT = dict(template="plotly_white", font=dict(size=13, color=C_INK), margin=dict(l=10, r=10, t=60, b=10),
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), bargap=0.25, bargroupgap=0.08,
              paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")


def norm(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None
    a = str(a).strip().upper()
    return a if a in VALID else None


def short(name):
    return name.split("/")[-1]


def wilson(k, n, z=1.96):
    """95% Wilson interval half-widths (lower, upper) around k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p - (c - h), (c + h) - p


def long_format(df):
    """One row per (record, predictor) with normalised answers and correctness flags."""
    rows = []
    for r in df.itertuples(index=False):
        names_w = getattr(r, f"{P}with_explanation_predictor_names")
        ans_w = getattr(r, f"{P}with_explanation_predictor_answers")
        names_wo = getattr(r, f"{P}without_explanation_predictor_names")
        ans_wo = getattr(r, f"{P}without_explanation_predictor_answers")
        if names_w is None or ans_w is None or ans_wo is None:
            continue
        wo_map = dict(zip(names_wo if names_wo is not None else names_w, ans_wo))
        for name, a_w in zip(names_w, ans_w):
            rows.append(dict(
                predictor=name, ref_model=r.original_reference_response_model_info_model,
                answer_first=r.original_answer_first, orig=norm(r.original_reference_response_answer),
                cf=norm(r.counterfactual_reference_response_answer), pw=norm(a_w), pwo=norm(wo_map.get(name)),
            ))
    long = pd.DataFrame(rows)
    n_all = len(long)
    long = long.dropna(subset=["orig", "cf", "pw", "pwo"]).copy()
    long["with_ok"] = long.pw == long.cf
    long["without_ok"] = long.pwo == long.cf
    long["copy_ok"] = long.orig == long.cf          # copy-original baseline is correct iff the reference kept its answer
    long["with_copies"] = long.pw == long.orig
    long["without_copies"] = long.pwo == long.orig
    long["ref_flipped"] = long.cf != long.orig
    return long, n_all


def summarise(g):
    n = len(g)
    out = {"n": n}
    for col in ["with_ok", "without_ok", "copy_ok", "with_copies", "without_copies", "ref_flipped"]:
        k = int(g[col].sum())
        out[col] = k / n if n else np.nan
        out[f"{col}_lo"], out[f"{col}_hi"] = wilson(k, n)
    out["gain"] = out["with_ok"] - out["without_ok"]
    return pd.Series(out)


def breakdown(long):
    """Tidy table: predictor x slice, where slice is overall / reference kept-flipped / answer format."""
    parts = []
    for label, key in [("all", None), ("ref kept answer", False), ("ref flipped answer", True)]:
        sub = long if key is None else long[long.ref_flipped == key]
        t = sub.groupby("predictor").apply(summarise, include_groups=False).reset_index()
        t.insert(1, "slice", label)
        parts.append(t)
    for af in [True, False]:
        t = long[long.answer_first == af].groupby("predictor").apply(summarise, include_groups=False).reset_index()
        t.insert(1, "slice", f"answer_first={af}")
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def bar(t, col, name, color, show_legend=True, legendgroup=None):
    return go.Bar(
        x=[short(p) for p in t.predictor], y=t[col], name=name, marker=dict(color=color, line=dict(width=0)),
        legendgroup=legendgroup or name, showlegend=show_legend,
        error_y=dict(type="data", symmetric=False, array=t[f"{col}_hi"], arrayminus=t[f"{col}_lo"],
                     color=C_MUTED, thickness=1, width=3),
        customdata=np.stack([t.n, t[f"{col}_lo"], t[f"{col}_hi"]], axis=-1),
        hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1%}}<br>n = %{{customdata[0]:,}}<extra></extra>",
    )


def copy_line(t, fig, row=None, col=None, show_legend=True):
    """Copy-original baseline as a dashed marker line per predictor (same value across predictors up to row drops)."""
    fig.add_trace(go.Scatter(
        x=[short(p) for p in t.predictor], y=t.copy_ok, mode="markers+lines", name="copy-original baseline",
        legendgroup="copy", showlegend=show_legend, line=dict(color=C_BASE, dash="dash", width=2),
        marker=dict(size=9, color=C_BASE, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>copy-original baseline: %{y:.1%}<extra></extra>",
    ), row=row, col=col)


def style(fig, title, ytitle="accuracy", height=440):
    fig.update_layout(**LAYOUT, title=dict(text=title, x=0, font=dict(size=15)), barmode="group", height=height)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor=C_GRID, zeroline=False, title_text=ytitle)
    fig.update_xaxes(showgrid=False, linecolor=C_GRID)
    return fig


def fig_overall(t):
    t = t[t.slice == "all"]
    fig = go.Figure([bar(t, "with_ok", "with explanation", C_WITH), bar(t, "without_ok", "without explanation", C_WITHOUT)])
    copy_line(t, fig)
    return style(fig, "Accuracy with vs without explanation (dashed: always copy the original answer)")


def fig_copy_rate(t):
    t = t[t.slice == "all"]
    fig = go.Figure([bar(t, "with_copies", "with explanation", C_WITH),
                     bar(t, "without_copies", "without explanation", C_WITHOUT)])
    fig.add_trace(go.Scatter(
        x=[short(p) for p in t.predictor], y=1 - t.ref_flipped, mode="markers+lines",
        name="reference model actually kept its answer", line=dict(color=C_BASE, dash="dash", width=2),
        marker=dict(size=9, color=C_BASE, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>reference kept answer: %{y:.1%}<extra></extra>"))
    return style(fig, "How often the predictor predicts the original answer (copy rate)", ytitle="share of predictions")


def fig_facets(t, slices, titles, title, with_copy=True):
    fig = make_subplots(rows=1, cols=len(slices), subplot_titles=titles, shared_yaxes=True, horizontal_spacing=0.04)
    for i, s in enumerate(slices, start=1):
        ts = t[t.slice == s]
        first = i == 1
        fig.add_trace(bar(ts, "with_ok", "with explanation", C_WITH, first), row=1, col=i)
        fig.add_trace(bar(ts, "without_ok", "without explanation", C_WITHOUT, first), row=1, col=i)
        if with_copy:
            copy_line(ts, fig, row=1, col=i, show_legend=first)
    style(fig, title, height=490)
    fig.update_layout(margin=dict(t=70, b=40), legend=dict(y=-0.32, yanchor="top"))
    fig.update_yaxes(title_text=None, col=2)
    for a in fig.layout.annotations:
        a.font = dict(size=13, color=C_MUTED)
    return fig


def fig_gain_format(t):
    fig = go.Figure()
    for s, name, color in [("answer_first=True", "answer first", C_AF), ("answer_first=False", "explanation first", C_EF)]:
        ts = t[t.slice == s]
        fig.add_trace(go.Bar(x=[short(p) for p in ts.predictor], y=ts.gain, name=name, marker=dict(color=color),
                             customdata=ts.n, hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:+.1%}}<br>n = %{{customdata:,}}<extra></extra>"))
    fig.update_layout(**LAYOUT, title=dict(text="Explanation gain (with − without) by answer format", x=0, font=dict(size=15)),
                      barmode="group", height=400)
    fig.update_yaxes(tickformat="+.0%", gridcolor=C_GRID, zeroline=True, zerolinecolor=C_MUTED, title_text="accuracy gain")
    fig.update_xaxes(showgrid=False)
    return fig


def fig_html(fig, first):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                       config={"responsive": True, "displaylogo": False})


def table_html(t):
    show = t[["predictor", "slice", "n", "with_ok", "without_ok", "gain", "copy_ok", "with_copies", "without_copies"]].copy()
    show.columns = ["predictor", "slice", "n", "acc with expl", "acc without expl", "gain", "copy-original acc",
                    "copy rate with", "copy rate without"]
    show["predictor"] = show.predictor.map(short)
    for c in show.columns[3:]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else (f"{v:+.1%}" if c == "gain" else f"{v:.1%}"))
    return ("<details><summary>Show the numbers as a table</summary><div class='table-scroll'>"
            + show.to_html(index=False, border=0) + "</div></details>")


def load(path, thinking):
    df = pd.read_parquet(path, columns=COLS)
    df = df[df.original_dataset == "moral_machines"]
    present = sorted(df[THINKING_COL].map(repr).unique())
    if thinking is not None:
        df = df[df[THINKING_COL].astype(str).str.lower().isin(thinking)]
    return df, present


def section(path, df, first, thinking):
    long, n_all = long_format(df)
    t = breakdown(long)
    t.insert(0, "file", path.name)

    refs = df.original_reference_response_model_info_model.value_counts()
    ref_txt = ", ".join(f"{short(m)} ({c:,})" for m, c in refs.items())
    af = df.original_answer_first.value_counts()
    tc = df[THINKING_COL].astype(str).value_counts()
    filt = "" if thinking is None else (" with reference thinking in {" + ", ".join(thinking) + "} ("
                                        + ", ".join(f"{k}: {v:,}" for k, v in tc.items()) + ")")
    meta = (f"{len(df):,} Moral Machines rows{filt} · reference models: {ref_txt} · answer_first True/False: "
            f"{af.get(True, 0):,}/{af.get(False, 0):,} · {len(long):,} of {n_all:,} (row, predictor) pairs kept after "
            f"dropping missing/invalid answers · reference flipped its answer on {long.ref_flipped.mean():.1%} of kept pairs")

    figs = [
        fig_overall(t),
        fig_copy_rate(t),
        fig_facets(t, ["ref kept answer", "ref flipped answer"],
                   ["Reference kept its original answer", "Reference flipped its answer"],
                   "Accuracy split by whether the reference model kept or flipped its answer", with_copy=False),
        fig_facets(t, ["answer_first=True", "answer_first=False"],
                   ["Answer first (answer_first = True)", "Explanation first (answer_first = False)"],
                   "Accuracy by answer format (dashed: copy-original baseline within that format)"),
        fig_gain_format(t),
    ]
    body = "".join(f"<div class='fig'>{fig_html(f, first and i == 0)}</div>" for i, f in enumerate(figs))
    anchor = path.stem
    out = (f"<section id='{anchor}'><h2>{html.escape(path.name)}</h2><p class='meta'>{html.escape(meta)}</p>"
           f"{body}{table_html(t)}</section>")
    return out, t


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Moral Machines Predictor Breakdown</title>
<style>
:root {{ --bg:#fcfcfb; --surface:#ffffff; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, system-ui, sans-serif; margin:0; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 8px; }} h2 {{ font-size:19px; margin:40px 0 4px; padding-top:16px; border-top:1px solid var(--line); }}
.meta, .explain {{ color:var(--muted); font-size:14px; }}
nav a {{ margin-right:14px; color:#2a78d6; text-decoration:none; font-size:14px; white-space:nowrap; }}
.fig {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:8px; margin:14px 0; }}
details {{ margin:10px 0; }} summary {{ cursor:pointer; color:var(--muted); }}
.table-scroll {{ overflow-x:auto; }} table {{ border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:4px 10px; border-bottom:1px solid var(--line); text-align:right; }} th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
</style></head><body><main>
<h1>Moral Machines: predictor accuracy breakdown{h1_suffix}</h1>
<p class="explain">Each section is one prediction parquet from <code>data_parquet/</code>, filtered to
<code>original_dataset == "moral_machines"</code>{filter_text}. A prediction is correct when it matches the reference model's answer on the
counterfactual. The <b>copy-original baseline</b> always predicts the reference model's answer on the original question; it is
right exactly when the reference model did not change its answer. The <b>copy rate</b> is how often a predictor's
answer equals that original answer. Error bars are 95% Wilson intervals (rows treated as independent; they are not clustered
by original question). A (row, predictor) pair is dropped if any reference or predictor answer is missing or not CASE 1/CASE 2.</p>
<nav>{nav}</nav>
{skipped}
{sections}
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking", nargs="+", default=["medium", "high"],
                    help='reference thinking levels to keep (case-insensitive); "all" disables the filter')
    args = ap.parse_args()
    levels = None if "all" in args.thinking else [v.lower() for v in args.thinking]
    th = "all" if levels is None else ", ".join(levels)
    suffix = "" if levels is None else "_thinking_" + "_".join(levels)
    out_html = OUT_DIR / f"mm_predictor_breakdown{suffix}.html"
    out_csv = OUT_DIR / f"mm_predictor_breakdown{suffix}.csv"

    sections, tables, kept, skipped = [], [], [], []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        df, present = load(f, levels)
        if df.empty:
            print(f"skipping {f.name}: no Moral Machines rows with thinking in {{{th}}} (has {', '.join(present)})")
            skipped.append((f, present))
            continue
        print(f"processing {f.name} ({len(df):,} rows)")
        s, t = section(f, df, first=not kept, thinking=levels)
        sections.append(s)
        tables.append(t)
        kept.append(f)
    if not kept:
        raise SystemExit(f"no file has Moral Machines rows with thinking in {{{th}}}")

    nav = "".join(f"<a href='#{f.stem}'>{html.escape(f.stem)}</a>" for f in kept)
    skipped_html = "" if not skipped else (
        f"<p class='meta'>Skipped (no rows with thinking in {{{html.escape(th)}}}): "
        + "; ".join(f"{html.escape(f.name)} (thinking values: {html.escape(', '.join(p))})" for f, p in skipped)
        + "</p>")
    page = PAGE.format(
        nav=nav, sections="".join(sections), skipped=skipped_html,
        h1_suffix="" if th == "all" else f" (reference thinking: {html.escape(th)})",
        filter_text="" if th == "all" else
        f" and <code>original_reference_response_model_info_thinking</code> in {{{html.escape(th)}}}")
    out_html.write_text(page)
    pd.concat(tables, ignore_index=True).to_csv(out_csv, index=False)
    print(f"wrote {out_html}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
