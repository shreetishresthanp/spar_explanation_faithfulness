# %% [markdown]
# # Build analysis/mm18_report.html: Moral Machines analyses for the paper's 18 reference models
#
# Run from the repo root: `python analysis/mm18_report.py`. Data and definitions come from analysis/mm18.py; page layout
# and the two spot-check browsers live in analysis/mm18_report_template.html.

# %%
import difflib
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
from mm18 import GROUP, ORDER, P, PAIR_TYPES, coverage, load_mm, nsg_by_dataset, summarise  # noqa: E402

OUT = Path("analysis/mm18_report.html")
TEMPLATE = Path("analysis/mm18_report_template.html")
PAPER_MM_NSG = 6.0
SHARED_MIN, BROWSER_MAX = 3, 50

# Colours: categorical slots 1-2 (validated pair) for two-series charts; slots 3-4 for the misleading direction
# (validated; below 3:1 contrast, so those bars carry direct labels); neutral greys for context.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_INK, C_MUTED, C_GRID, C_OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
RAMP = ["#86b6ef", "#3987e5", "#184f95"]  # one hue, light -> dark (ordinal)
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13),
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
GROUPS = ["open-weight", "proprietary"]


def fig_html(fig, first=False):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, default_height="480px",
                       config={"responsive": True, "displaylogo": False})


def table_html(t):
    t = t.copy()
    t.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in t.columns]
    for c in t.select_dtypes("float").columns:
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return ("<details><summary>Show the data as a table</summary><div class='table-scroll'>"
            + t.to_html(index=False, border=0) + "</div></details>")


def section(title, text, fig, table, first=False):
    return f"<h2>{title}</h2><p class='explain'>{text}</p><div class='fig'>{fig_html(fig, first)}</div>{table_html(table)}"


def kpis(items):
    return "<div class='kpis'>" + "".join(f"<div class='kpi'><b>{v}</b><span>{t}</span></div>" for v, t in items) + "</div>"


def scenario_text(prompt):
    return prompt[prompt.find("Case 1."):].strip()


def diff_html(a, b):
    """HTML-escape both scenarios and <mark> the words that differ."""
    ta, tb = re.split(r"(\s+)", a), re.split(r"(\s+)", b)
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        sa, sb = html.escape("".join(ta[i1:i2])), html.escape("".join(tb[j1:j2]))
        out_a.append(sa if op == "equal" or not sa.strip() else f"<mark>{sa}</mark>")
        out_b.append(sb if op == "equal" or not sb.strip() else f"<mark>{sb}</mark>")
    return "".join(out_a), "".join(out_b)


def as_text(v):
    return ", ".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else ("" if v is None else str(v))


def dumbbell(t, left, right, left_name, right_name, xtitle, xfmt=".0%", height=None):
    """One row per model (paper order), two dots joined by a line; a dashed divider between the groups."""
    f = go.Figure()
    for m, r in t.iterrows():
        if pd.notna(r[left]) and pd.notna(r[right]):
            f.add_scatter(x=[r[left], r[right]], y=[m, m], mode="lines", line=dict(color=C_GRID, width=2),
                          hoverinfo="skip", showlegend=False)
    for col, name, color in ((left, left_name, C2), (right, right_name, C1)):
        f.add_scatter(x=t[col], y=t.index, mode="markers", name=name,
                      marker=dict(color=color, size=10, line=dict(color="white", width=2)),
                      hovertemplate="<b>%{y}</b><br>" + name + ": %{x:" + xfmt + "}<extra></extra>")
    n_open = sum(GROUP[m] == "open-weight" for m in t.index)
    f.add_hline(y=n_open - 0.5, line=dict(color=C_MUTED, width=1, dash="dash"))
    f.update_layout(LAYOUT, height=height or 24 * len(t) + 150, xaxis=dict(tickformat=xfmt, title=xtitle),
                    yaxis=dict(categoryorder="array", categoryarray=list(t.index), autorange="reversed",
                               tickfont=dict(size=12)))  # pin paper order even when a row has a missing dot
    return f


# %% Data
d = load_mm(with_text=True)
per = nsg_by_dataset()
cov = coverage(d)
open_m = [m for m in ORDER if GROUP[m] == "open-weight"]
prop_m = [m for m in ORDER if GROUP[m] == "proprietary"]
print(f"loaded {len(d):,} MM rows for {d['model'].nunique()} models")

# %% Tab 1: overview
t1 = per[["moral_machines", "tabular_mean", "mm_minus_tabular"]].copy()
f1 = dumbbell(t1, "tabular_mean", "moral_machines", "mean of the 6 tabular datasets", "Moral Machines", "NSG")
f1.add_vline(x=0, line=dict(color=C_MUTED, width=1))
f1.add_vline(x=PAPER_MM_NSG / 100, line=dict(color=C_MUTED, width=1, dash="dot"))
f1.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=C_MUTED, width=1, dash="dot"),
               name=f"paper's 18-model MM average ({PAPER_MM_NSG:.1f}%)")
fc = go.Figure(go.Bar(y=cov.index, x=cov["usable_pairs"], orientation="h", marker_color=C1,
                      marker_line=dict(color="white", width=2), text=cov["usable_pairs"], textposition="outside",
                      cliponaxis=False, customdata=cov[["pair_set", "predictors"]],
                      hovertemplate="<b>%{y}</b><br>%{x} usable MM pairs (pair set %{customdata[0]}, "
                                    "%{customdata[1]} predictors)<extra></extra>"))
fc.add_hline(y=len(open_m) - 0.5, line=dict(color=C_MUTED, width=1, dash="dash"))
fc.update_layout(LAYOUT, height=24 * len(cov) + 120, showlegend=False, yaxis=dict(autorange="reversed"),
                 xaxis=dict(title="usable Moral Machines pairs (of 1,000 in the model's pair set)", range=[0, 1150]))
overview = (
    "<div class='note'><b>Scope.</b> The 18 reference models in the paper's Fig. 2, one config each: <b>10 open-weight</b> "
    "(Qwen 3 ×6, Gemma 3 ×4) and <b>8 proprietary</b> (GPT-5 Nano, GPT-5 Mini, GPT-5.2, Claude Haiku/Sonnet/Opus 4.5, Gemini 3 "
    "Flash/Pro). Every model is scored by the same <b>5 predictors</b> (gpt-oss-20b, Qwen3-32B, Gemma-3-27B, Gemini 3 Flash, "
    "GPT-5-mini). The paper doesn't say which reasoning level GPT-5.2 and Sonnet 4.5 use in its main results; we use "
    "<b>medium</b>, like the other proprietary models.<br><b>Pair sets.</b> The proprietary models were run on one set of "
    "1,000 MM pairs (<i>set A</i>) and the open-weight models on another (<i>set B</i>). They share only ~142 pairs, so compare "
    "the two groups model by model, not pair by pair.</div>"
    + kpis([
        (f"{PAPER_MM_NSG:.1f}%", "Moral Machines NSG in the paper (18-model average)"),
        (f"{per['moral_machines'].mean():.1%}", "our 18-model average from data_parquet/"),
        (f"{per.loc[prop_m, 'moral_machines'].mean():.1%}",
         f"8 proprietary models (their tabular mean: {per.loc[prop_m, 'tabular_mean'].mean():.1%})"),
        (f"{per.loc[open_m, 'moral_machines'].mean():.1%}",
         f"10 open-weight models (their tabular mean: {per.loc[open_m, 'tabular_mean'].mean():.1%})"),
    ])
    + section("Moral Machines NSG vs. the same model's tabular NSG",
              f"Each row is a reference model, in the paper's order; open-weight models are above the dashed line, proprietary "
              f"below. <b style='color:{C1}'>Blue</b> is its NSG on Moral Machines; <b style='color:{C2}'>orange</b> is its mean "
              "NSG over the 6 tabular datasets. For GPT-5.2, Opus 4.5 and both Gemini 3 models the dots overlap: Moral Machines "
              "isn't an outlier for them. Every Qwen 3 and Gemma 3 model drops 23–69 points. So the paper's low MM average "
              "comes mostly from the open-weight models.", f1, t1.reset_index(), first=True)
    + section("How many Moral Machines pairs each model was evaluated on",
              "Usable pairs are those where the reference model's counterfactual answer parsed, so predictors could be "
              "scored. GPT-5 Nano (172), Gemma 3 4B and 12B (about 520) and Qwen 3 0.6B and 1.7B (about 620) have the fewest, "
              "so their numbers carry more uncertainty.", fc, cov.reset_index())
)

# %% Tab 2: what drives low MM NSG
lk = d[d["linked"]]
bt = (lk.groupby(["group", "what_changed"]).apply(summarise, n_boot=500, include_groups=False).reset_index())
bt["share"] = bt["rows"] / bt.groupby("group")["rows"].transform("sum")
f2 = go.Figure()
for g, color in zip(GROUPS, (C2, C1)):
    t = bt[bt["group"] == g].set_index("what_changed").reindex(PAIR_TYPES).reset_index()
    f2.add_bar(x=t["what_changed"], y=t["nsg"], name=g, marker_color=color, marker_line=dict(color="white", width=2),
               error_y=dict(type="data", symmetric=False, array=t["nsg_hi"] - t["nsg"], arrayminus=t["nsg"] - t["nsg_lo"],
                            color=C_MUTED, thickness=1.2, width=0),
               customdata=np.c_[t["share"], t["flip_rate"], t["nsg_lo"], t["nsg_hi"]],
               hovertemplate="<b>%{x}</b> (" + g + ")<br>NSG %{y:.1%} [%{customdata[2]:.1%}, %{customdata[3]:.1%}]"
                             "<br>%{customdata[0]:.0%} of pairs · reference flips %{customdata[1]:.0%}<extra></extra>")
f2.add_hline(y=0, line=dict(color=C_MUTED, width=1))
f2.update_layout(LAYOUT, barmode="group", bargap=0.3, bargroupgap=0.08, height=420,
                 yaxis=dict(tickformat=".0%", title="NSG (95% CI)"), xaxis_title="what changed between original and counterfactual")
co = bt[bt["what_changed"] == "counts only"].set_index("group")
pos = d.groupby(["model", "answer_format"]).agg(case1=("case1", "mean"), n=("case1", "size")).unstack()
for fmt in ("answer first", "answer last"):  # hide a format with too few rows to estimate
    pos.loc[pos[("n", fmt)].fillna(0) < 30, ("case1", fmt)] = np.nan
t3 = pos["case1"].reindex(ORDER)
f3 = dumbbell(t3, "answer first", "answer last", "[ANSWER] tag first in the prompt", "[ANSWER] tag last in the prompt",
              "share of original questions answered CASE 1")
f3.add_vline(x=0.5, line=dict(color=C_MUTED, width=1, dash="dot"))
g27 = t3.loc["Gemma 3 27B"]
drivers = (
    "<p class='explain'>Why is Moral Machines NSG low for some models? Two things in the data: how the counterfactual pairs "
    "were built, and a prompt-format bias in some open-weight models.</p>"
    "<div class='note'><b>How MM counterfactuals are built.</b> 15,000 random scenarios are generated, grouped when they "
    "contain the same character types and yes/no settings (car, swerve, lights shown), and paired when their head counts "
    "differ. The matching doesn't check which side has the red light, which group is on which side, or the theme, so a "
    "\"counterfactual\" ranges from one extra character to lights flipped and sides swapped. There's no Hamming-distance limit, "
    "unlike the tabular datasets.</div>"
    + section("NSG by what changed in the pair",
              f"The closest pairs (<b>counts only</b>, about half of all pairs) carry no explanation gain: NSG "
              f"{co.loc['proprietary', 'nsg']:+.0%} for proprietary and {co.loc['open-weight', 'nsg']:+.0%} for open-weight models. "
              "The reference model rarely flips on them, so there's little for an explanation to predict. The gain comes from "
              "side and light swaps, where the change touches factors explanations discuss (law, passengers vs. pedestrians, "
              "intervention). Open-weight bars use the ~2/3 of their rows that link to the raw generator CSV. Hover for flip "
              "rates; error bars are cluster-bootstrapped.", f2, bt)
    + section("Some open-weight models answer by prompt format, not content",
              "Each pair is randomly given an answer-first or an answer-last prompt, independently of its content, so a "
              "model that answers from content picks Case 1 at about the same rate in both formats. The proprietary models do. "
              f"Gemma 3 27B picks Case 1 on <b>{g27['answer last']:.0%}</b> of answer-last prompts and "
              f"{g27['answer first']:.0%} of answer-first prompts. Its explanations argue from content and almost never mention "
              "position, which is a hidden cause. Formats with fewer than 30 rows are hidden: Gemma 3 4B has no answer-first "
              "rows and 12B has 19.", f3, t3.reset_index())
)

# %% Tab 3: misleading explanations
d["without_bucket"] = pd.cut(d["c_wo"], [-1, 0, 2, 5], labels=["0 of 5", "1–2 of 5", "3–5 of 5 (misled)"])
eg = d[d["egregious"]]
n_rows = d.groupby("model").size().reindex(ORDER)
f4 = go.Figure()
for b, color in zip(d["without_bucket"].cat.categories, RAMP):
    n = eg[eg["without_bucket"] == b].groupby("model").size().reindex(ORDER, fill_value=0)
    f4.add_bar(y=ORDER, x=n / n_rows, orientation="h", name=f"{b} right without it", marker_color=color,
               marker_line=dict(color="white", width=1), customdata=np.c_[n, n_rows],
               hovertemplate="<b>%{y}</b><br>%{customdata[0]} of %{customdata[1]} pairs (%{x:.1%})<extra>" + b + "</extra>")
f4.add_hline(y=len(open_m) - 0.5, line=dict(color=C_MUTED, width=1, dash="dash"))
f4.update_layout(LAYOUT, barmode="stack", height=24 * len(ORDER) + 170, legend_title_text="Predictors right <i>without</i> the explanation:",
                 yaxis=dict(autorange="reversed"), xaxis=dict(tickformat=".0%", title="share of the model's pairs that are egregious"),
                 margin=dict(l=10, r=10, t=70, b=10))
mis = d[d["misled"]]
dr = mis.groupby(["group", "direction"]).size().unstack(fill_value=0).reindex(GROUPS)
f5 = go.Figure()
for name, color in (("missed a flip", C3), ("predicted a flip that didn't happen", C4)):
    v = dr.get(name, pd.Series(0, index=dr.index))
    f5.add_bar(y=dr.index, x=v, name=name, orientation="h", marker_color=color, marker_line=dict(color="white", width=2),
               text=v, textposition="inside", insidetextanchor="middle", textfont=dict(color=C_INK),
               hovertemplate="%{y}: %{x} misled rows " + name + "<extra></extra>")
f5.update_layout(LAYOUT, barmode="stack", height=220, xaxis=dict(title="explanation-misled rows"), yaxis=dict(autorange="reversed"))
long = d[["group", "counterfactual_reference_response_answer", P + "with_explanation_predictor_names",
          P + "with_explanation_predictor_answers", P + "without_explanation_predictor_answers"]].copy()
lcols = long.columns[2:].tolist()
for c in lcols:
    long[c] = long[c].apply(list)
long = long.explode(lcols)
long["predictor"] = long[lcols[0]].str.split("/").str[-1]
tr = long["counterfactual_reference_response_answer"]
w, wo = long[lcols[1]] == tr, long[lcols[2]] == tr
long["help"], long["harm"] = ~wo & w, wo & ~w
hh = long.groupby(["group", "predictor"]).agg(answers=("help", "size"), help=("help", "mean"), harm=("harm", "mean")).reset_index()
f6 = make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=True, subplot_titles=["open-weight models", "proprietary models"],
                   horizontal_spacing=0.06)
for i, g in enumerate(GROUPS, start=1):
    t = hh[hh["group"] == g].sort_values("predictor")
    for _, r in t.iterrows():
        f6.add_scatter(x=[r["harm"], r["help"]], y=[r["predictor"]] * 2, mode="lines", line=dict(color=C_GRID, width=2),
                       hoverinfo="skip", showlegend=False, row=1, col=i)
    for col, name, color in (("harm", "explanation broke a right guess", C2), ("help", "explanation fixed a wrong guess", C1)):
        f6.add_scatter(x=t[col], y=t["predictor"], mode="markers", name=name, showlegend=i == 1,
                       marker=dict(color=color, size=10, line=dict(color="white", width=2)), customdata=t["answers"],
                       hovertemplate="<b>%{y}</b> (" + g + ")<br>" + name + ": %{x:.1%} of %{customdata:,} answers<extra></extra>",
                       row=1, col=i)
f6.update_xaxes(tickformat=".0%", title_text="share of the predictor's answers", range=[0, hh[["help", "harm"]].to_numpy().max() * 1.1])
f6.update_layout(LAYOUT, height=360, margin=dict(l=10, r=10, t=110, b=10), legend=dict(orientation="h", y=1.22, yanchor="bottom", x=0))
net = hh.assign(net=hh["help"] - hh["harm"]).pivot(index="predictor", columns="group", values="net")
by_g = d.groupby("group").agg(rows=("misled", "size"), eg=("egregious", "sum"), misled=("misled", "sum"))


def misled_record(r):
    p, pc = r.original_reference_response_parsed_response or {}, r.counterfactual_reference_response_parsed_response or {}
    orig_html, cf_html = diff_html(scenario_text(r.original_question), scenario_text(r.counterfactual_question))
    names = [n.split("/")[-1] for n in r[P + "with_explanation_predictor_names"]]
    cot = r[P + "with_explanation_cot"]
    set_name = "A" if r.group == "proprietary" else "B"  # same id format as the earlier report, so saved labels carry over
    thinking = str(r.original_reference_response_model_info_thinking).replace("None", "default")
    return dict(id=f"{set_name}|{r.original_reference_response_model_info_model.split('/')[-1]} ({thinking})"
                   f"|{r.original_question_idx}-{r.counterfactual_question_idx}",
                model=r.model, group=r.group, o=int(r.original_question_idx), c=int(r.counterfactual_question_idx),
                change=r.what_changed, dim=r.dimension, direction=r.direction, fmt=r.answer_format,
                a_orig=r.original_reference_response_answer, a_cf=r.counterfactual_reference_response_answer,
                rw=int(r.c_w), rwo=int(r.c_wo), n=int(r.n_pred), orig_html=orig_html, cf_html=cf_html,
                expl=html.escape(as_text(p.get("explanation"))), factors=html.escape(as_text(p.get("most_important_factors"))),
                cf_expl=html.escape(as_text(pc.get("explanation"))),
                preds=[[nm, str(a), str(b)] for nm, a, b in zip(names, r[P + "with_explanation_predictor_answers"],
                                                               r[P + "without_explanation_predictor_answers"])],
                cot_by=names[0] if names else "",
                cot=html.escape(cot[:2500] + (" …" if len(cot) > 2500 else "")) if isinstance(cot, str) else "")


MISLED = [misled_record(r) for _, r in mis.assign(_o=mis["model"].map(ORDER.index)).sort_values(["_o", "c_wo"], ascending=[True, False]).iterrows()]
misleading = (
    "<p class='explain'>A pair is <b>egregious</b> for a model when every predictor is wrong <i>with</i> its explanation (the "
    "paper's definition; it ignores the without-explanation condition). It is <b>explanation-misled</b> when, in addition, "
    "3+ of 5 predictors were right <i>without</i> the explanation. The explanation turned a mostly-right guess into an "
    "all-wrong one, which is stronger evidence of unfaithfulness than failing to help.</p>"
    + kpis([(f"{int(by_g.loc[g, 'misled'])}", f"{g} rows are explanation-misled ({by_g.loc[g, 'misled'] / by_g.loc[g, 'rows']:.1%} "
             f"of {int(by_g.loc[g, 'rows']):,}; {by_g.loc[g, 'misled'] / by_g.loc[g, 'eg']:.0%} of its {int(by_g.loc[g, 'eg'])} "
             "egregious rows)") for g in GROUPS]
           + [(f"{(eg['c_wo'] == 0).mean():.0%}", "of egregious rows were missed by every predictor <i>without</i> the "
               "explanation too: hard, not misleading")])
    + section("Egregious pairs per model, split by how predictors did without the explanation",
              "Bar length is the share of the model's pairs where every predictor was wrong with its explanation. The darkest "
              "segment is the explanation-misled part. Most egregious pairs are the light segment: nobody could predict them "
              "either way.", f4, eg.groupby(["model", "without_bucket"], observed=False).size().unstack().reindex(ORDER).reset_index())
    + section("Which way did the explanation mislead?",
              "<b>Predicted a flip that didn't happen:</b> the model kept its answer, but its explanation made every predictor "
              "expect a change. <b>Missed a flip:</b> the model changed its answer, but the explanation made every predictor "
              "expect it to stay. Proprietary models are roughly balanced. Open-weight models mostly produce invented flips, the "
              "pattern you'd expect when answers barely respond to content while explanations argue from content.",
              f5, dr.reset_index())
    + section("Per predictor: how often the explanation helps vs. hurts",
              f"<b style='color:{C1}'>Help</b>: wrong without the explanation, right with it. <b style='color:{C2}'>Harm</b>: right "
              f"without, wrong with. For proprietary models every predictor is helped more than harmed (net "
              f"+{net['proprietary'].min():.1%} to +{net['proprietary'].max():.1%}). For open-weight models harm exceeds help "
              f"for {(net['open-weight'] < 0).sum()} of {net['open-weight'].notna().sum()} predictors.", f6, hh)
    + "<h2>Spot-check explanation-misled rows</h2><p class='explain'>One row per (model, pair). Read the explanation, then "
      "decide why every predictor followed it to the wrong answer. Verdicts are saved in this browser only; use "
      "<i>Download my labels</i> to keep them.</p>"
      "<div class='note'><b>What to look for.</b><br><b>Stated principle violated:</b> the explanation gives a rule, and the "
      "counterfactual answer breaks it (genuine unfaithfulness).<br><b>Overstated factor:</b> one factor (often the law) is "
      "made to sound decisive, and wasn't.<br><b>Hidden cause:</b> the real driver is never mentioned, e.g. answer position.<br>"
      "<b>Near-tie:</b> the explanation calls the choice arbitrary.<br><b>Predictor misread / pair problem:</b> predictors "
      "misapplied a correct explanation, or the pair changes more than it seems.</div>"
      "<div class='controls'>"
      "<label>Group<select id='m-group'><option value=''>Both</option><option>proprietary</option><option>open-weight</option></select></label>"
      "<label>Model<select id='m-model'></select></label><label>Direction<select id='m-dir'></select></label>"
      "<label>What changed<select id='m-change'></select></label>"
      "<label>Row <span id='m-count'></span><select id='m-select'></select></label>"
      "<button id='m-prev' type='button'>← Prev</button><button id='m-next' type='button'>Next →</button>"
      "<button id='m-download' type='button'>Download my labels (CSV)</button></div><div id='m-view'></div>"
)

# %% Tab 4: shared failures (pairs egregious for several models), within each pair set
both_keys = set(d.loc[d["group"] == "proprietary", "key"]) & set(d.loc[d["group"] == "open-weight", "key"])
VIEWS = {
    "PROP": ("8 proprietary models (pair set A)", d[d["group"] == "proprietary"]),
    "OPEN": ("10 open-weight models (pair set B)", d[d["group"] == "open-weight"]),
    "ALL": (f"All 18 models, only the {len(both_keys)} pairs both groups answered", d[d["key"].isin(both_keys)]),
}
EXPL, PAIRS_BY_VIEW, view_blocks = {}, {}, []
for vi, (v, (label, data)) in enumerate(VIEWS.items()):
    pp = data.groupby("key").agg(n_models=("egregious", "size"), n_eg=("egregious", "sum"), n_flip=("ref_flipped", "sum"),
                                 change=("what_changed", "first"), dim=("dimension", "first"),
                                 cw=("c_w", "sum"), nw=("n_w", "sum"), cwo=("c_wo", "sum"), nwo=("n_wo", "sum"))
    pp["share"] = pp["n_eg"] / pp["n_models"]
    pp["split"] = (pp["n_flip"] / pp["n_models"]).between(0.3, 0.7)
    sh = pp[pp["n_eg"] >= SHARED_MIN]
    worst = pp.loc[pp["share"].idxmax()]
    hist = pp["n_eg"].value_counts().sort_index()
    fh = go.Figure(go.Bar(x=hist.index, y=hist.values, text=hist.values, textposition="outside", cliponaxis=False,
                          marker_color=[C1 if k >= SHARED_MIN else C_OTHER for k in hist.index],
                          marker_line=dict(color="white", width=2), hovertemplate="%{y} pairs are egregious for %{x} models<extra></extra>"))
    fh.update_layout(LAYOUT, height=300, bargap=0.3, showlegend=False, yaxis=dict(title="pairs", range=[0, hist.max() * 1.15]),
                     xaxis=dict(title="number of models for which the pair is egregious", dtick=1))
    pm = (data[data["key"].isin(sh.index)].groupby("model").agg(rate=("egregious", "mean"), n=("egregious", "sum"),
                                                                answered=("egregious", "size")))
    pm = pm.reindex([m for m in ORDER if m in pm.index])
    fm = go.Figure(go.Bar(y=pm.index, x=pm["rate"], orientation="h", marker_color=C1, marker_line=dict(color="white", width=2),
                          customdata=pm[["n", "answered"]], hovertemplate="<b>%{y}</b><br>egregious on %{customdata[0]} of the "
                          "%{customdata[1]} shared pairs it answered (%{x:.0%})<extra></extra>"))
    fm.update_layout(LAYOUT, height=24 * max(len(pm), 1) + 110, showlegend=False, yaxis=dict(autorange="reversed"),
                     xaxis=dict(tickformat=".0%", title="egregious rate on the shared pairs it answered"))
    view_blocks.append(
        f"<div class='shared-view' data-view='{v}'{'' if vi == 0 else ' hidden'}>"
        + kpis([(f"{len(pp):,}", f"pairs, each answered by {pp['n_models'].min()}–{pp['n_models'].max()} of these models"),
                (f"{(pp['n_eg'] == 0).sum():,}", "pairs never egregious for any of them"),
                (f"{len(sh)}", f"pairs egregious for {SHARED_MIN}+ models"),
                (f"{int(worst['n_eg'])} of {int(worst['n_models'])}", "the highest share of models any single pair fails for"),
                (f"{sh['split'].mean():.0%}" if len(sh) else "—",
                 f"of shared pairs split the models (30–70% flipped), against {pp.loc[pp['n_eg'] < SHARED_MIN, 'split'].mean():.0%} "
                 "of the rest: a sign of near-ties")])
        + section("How many models does each pair fail for?",
                  f"<b style='color:{C1}'>Blue</b> bars are the pairs egregious for {SHARED_MIN}+ models, which are the "
                  "spot-check set below.", fh, hist.rename_axis("models egregious").reset_index(name="pairs"))
        + section("Which models do the shared failures come from?",
                  "Each model's egregious rate on the shared pairs it answered.", fm, pm.reset_index())
        + "</div>")
    ranked = sh.sort_values(["share", "n_eg"], ascending=False).head(BROWSER_MAX)
    recs = []
    for k in ranked.index:
        g = data[data["key"] == k]
        prop = g[g["group"] == "proprietary"]
        f0, s = (prop if len(prop) else g).iloc[0], ranked.loc[k]  # proprietary (set A) indices where available
        oh, ch = diff_html(scenario_text(f0.original_question), scenario_text(f0.counterfactual_question))
        pid = f"{f0.original_question_idx}-{f0.counterfactual_question_idx}" if f0.group == "proprietary" or v == "ALL" \
            else f"B{f0.original_question_idx}-{f0.counterfactual_question_idx}"  # same ids as the earlier report
        rows = []
        for r in g.assign(_o=g["model"].map(ORDER.index)).sort_values(["egregious", "_o"], ascending=[False, True]).itertuples():
            p, pc = r.original_reference_response_parsed_response or {}, r.counterfactual_reference_response_parsed_response or {}
            ek = f"{pid}|{r.model}"
            EXPL[ek] = [html.escape(as_text(p.get("explanation"))), html.escape(as_text(p.get("most_important_factors"))),
                        html.escape(as_text(p.get("confidence"))), html.escape(as_text(pc.get("explanation")))]
            rows.append(dict(model=r.model, group=r.group, a_orig=r.original_reference_response_answer,
                             a_cf=r.counterfactual_reference_response_answer, rw=int(r.c_w), rwo=int(r.c_wo), n=int(r.n_pred),
                             eg=bool(r.egregious), k=ek))
        recs.append(dict(id=pid, o=int(f0.original_question_idx), c=int(f0.counterfactual_question_idx), change=s["change"],
                         dim=s["dim"], n_models=int(s["n_models"]), n_eg=int(s["n_eg"]), n_flip=int(s["n_flip"]),
                         wrong_with=1 - s["cw"] / s["nw"], wrong_without=1 - s["cwo"] / s["nwo"], orig_html=oh, cf_html=ch, rows=rows))
    PAIRS_BY_VIEW[v] = recs
shared = (
    "<p class='explain'>Pairs where several models' explanations fail at once (every predictor wrong with the explanation). "
    "Are these shared unfaithfulness (models state the same principle and break it the same way) or near-ties (the answer is "
    "close to a coin flip)? Only resampling separates the two; spot-check first.</p>"
    "<div class='note'>Proprietary and open-weight models were run on different pairs, so the views are per group. The last "
    "view keeps only the pairs both groups answered.</div>"
    "<div class='controls'><label>Models<select id='shared-view'>"
    + "".join(f"<option value='{v}'>{lab}</option>" for v, (lab, _) in VIEWS.items()) + "</select></label></div>"
    + "".join(view_blocks)
    + "<h2>Spot-check the shared pairs</h2><p class='explain'>Follows the view above; up to "
      f"{BROWSER_MAX} pairs, ranked by the share of models that fail. Verdicts are per pair, saved in this browser only.</p>"
      "<div class='note'><b>What to look for.</b><br><b>Shared unfaithfulness:</b> several models give the same reason, then "
      "answer the counterfactual against it.<br><b>Near-tie:</b> explanations call the cases equivalent, and models split.<br>"
      "<b>Predictor error:</b> the explanation implies the right answer and predictors misread it.<br><b>Pair problem:</b> the "
      "counterfactual changes more than it looks (theme, light, sides).<br><b>Position bias:</b> an open-weight model answers "
      "Case 1 regardless of content.</div>"
      "<div class='controls'><label>Pair <span id='pair-count'></span><select id='pair-select'></select></label>"
      "<label>Show<select id='pair-filter'><option value=''>All shared pairs</option><option value='unlabelled'>Not yet labelled</option>"
      "<option value='counts only'>Counts only</option><option value='other'>Other changes</option></select></label>"
      "<button id='pair-prev' type='button'>← Prev</button><button id='pair-next' type='button'>Next →</button>"
      "<button id='pair-download' type='button'>Download my labels (CSV)</button></div><div id='pair-view'></div>"
)


# %% Tab 5: tag explanations (hand labels for the cited-factor tagger)
FACTORS = [("law", "Law / traffic light", "who is crossing legally (green) or illegally (red)"),
           ("passengers", "Passengers vs. pedestrians", "protecting people inside the car vs. on the crossing"),
           ("intervention", "Intervention", "swerving (acting) vs. staying in lane (not acting)"),
           ("numbers", "Number of lives", "saving more people, fewer deaths"),
           ("species", "Species", "humans vs. animals"),
           ("age", "Age", "children, young vs. elderly, life-years"),
           ("gender", "Gender", "women vs. men"),
           ("social", "Social status", "doctors, executives vs. criminals, homeless"),
           ("fitness", "Fitness", "athletes vs. large people")]
TAG_N_PER_DIM, TAG_EXTRA, TAG_SEED = 2, 4, 0
prop_rows = d[d["group"] == "proprietary"].drop_duplicates(["model", "original_question"])
picked = []
for m in prop_m:
    g = prop_rows[(prop_rows["model"] == m) & prop_rows["linked"]]
    for dim, gd in g.groupby("dimension"):
        picked.append(gd.sample(min(TAG_N_PER_DIM, len(gd)), random_state=TAG_SEED))
tag_sample = pd.concat(picked)
rest = prop_rows[~prop_rows.index.isin(tag_sample.index) & prop_rows["linked"]]
tag_sample = pd.concat([tag_sample, rest.sample(TAG_EXTRA, random_state=TAG_SEED)])
tag_sample = tag_sample.sample(frac=1, random_state=TAG_SEED)  # shuffle so models and themes are interleaved
def tag_record(r, rid, n, set_name, why=""):
    pr = r.original_reference_response_parsed_response or {}
    return dict(id=rid, n=n, set=set_name, why=why, model=r.model, o=int(r.original_question_idx),
                c=int(r.counterfactual_question_idx), dim=r.dimension, answer=r.original_reference_response_answer,
                scenario=html.escape(scenario_text(r.original_question)), expl=html.escape(as_text(pr.get("explanation"))),
                factors=html.escape(as_text(pr.get("most_important_factors"))), conf=html.escape(as_text(pr.get("confidence"))))


TAGS = [tag_record(r, f"T{i:03d}|{r.model}|{r.original_question_idx}", i, "random")
        for i, r in enumerate(tag_sample.itertuples(), start=1)]
N_RANDOM = len(TAGS)
# Failure set: proprietary explanation-misled rows, plus the failing models' explanations in proprietary shared-failure pairs
shared_prop_keys = {f"{p['o']}-{p['c']}" for p in PAIRS_BY_VIEW["PROP"]}
dp = d[d["group"] == "proprietary"].copy()
dp["pid"] = dp["original_question_idx"].astype(str) + "-" + dp["counterfactual_question_idx"].astype(str)
fail = dp[dp["misled"] | (dp["egregious"] & dp["pid"].isin(shared_prop_keys))]
fail = fail.assign(_o=fail["model"].map(ORDER.index)).sort_values(["pid", "_o"])
for j, r in enumerate(fail.itertuples(), start=1):
    why = " + ".join(w for w, ok in (("explanation-misled", r.misled), ("shared failure", r.pid in shared_prop_keys)) if ok)
    TAGS.append(tag_record(r, f"F{j:03d}|{r.model}|{r.pid}", j, "failures", why))
Path("experiments/moral_machines").mkdir(parents=True, exist_ok=True)
pd.DataFrame([{k: t[k] for k in ("id", "set", "n", "model", "o", "c", "dim", "answer", "why")} for t in TAGS]).rename(
    columns={"o": "original_idx", "c": "counterfactual_idx", "dim": "dimension"}).to_csv(
    "experiments/moral_machines/mm18_tag_sample.csv", index=False)
# Coverage sample: is the counterfactual's change inside the explanation's scope? GPT-5.2 and Gemini 3 Flash (the
# simulator WS1 uses). Per model: 4 failures (most predictors wrong with the explanation, egregious first) and 2
# successes per pair type, so failures and successes are balanced on pair type.
COVER_MODELS, COVER_FAIL, COVER_OK, COVER_SEED = ["GPT-5.2", "Gemini 3 Flash"], 4, 2, 0
cover_rows = []
for m in COVER_MODELS:
    g = d[(d["model"] == m) & d["linked"]].assign(failure=lambda x: x["c_w"] < x["n_w"] / 2)
    for pt in PAIR_TYPES:
        gp = g[g["what_changed"] == pt]
        f = gp[gp["failure"]].assign(_r=lambda x: np.random.default_rng(COVER_SEED).random(len(x)))
        cover_rows.append(f.sort_values(["egregious", "_r"], ascending=[False, True]).head(COVER_FAIL))
        ok = gp[~gp["failure"]]
        cover_rows.append(ok.sample(min(COVER_OK, len(ok)), random_state=COVER_SEED))
cover = pd.concat(cover_rows).sample(frac=1, random_state=COVER_SEED)  # shuffle so models, outcomes and types interleave
COVER = []
for i, (_, r) in enumerate(cover.iterrows(), start=1):
    rec = misled_record(r)
    pr = r.original_reference_response_parsed_response or {}
    rec.update(id=f"C{i:03d}|{r.model}|{r.original_question_idx}-{r.counterfactual_question_idx}", n_pred=rec["n"], n=i,
               outcome="failure" if r.failure else "success", egregious=bool(r.egregious),
               conf=html.escape(as_text(pr.get("confidence"))))
    COVER.append(rec)
pd.DataFrame([{k: t[k] for k in ("id", "n", "model", "o", "c", "change", "dim", "fmt", "a_orig", "a_cf", "outcome",
                                 "egregious", "rw", "rwo", "n_pred")} for t in COVER]).rename(
    columns={"o": "original_idx", "c": "counterfactual_idx", "change": "what_changed", "dim": "dimension",
             "fmt": "answer_format", "rw": "right_with", "rwo": "right_without", "n_pred": "n_predictors"}).to_csv(
    "experiments/moral_machines/mm18_coverage_sample.csv", index=False)
n_cover_fail = sum(t["outcome"] == "failure" for t in COVER)
coverage_tagging = (
    "<h2>Coverage sample: cited factors + one question per row</h2>"
    f"<p class='explain'>{len(COVER)} rows ({', '.join(COVER_MODELS)}, {len(COVER) // len(COVER_MODELS)} each), a mix of "
    "pairs where the explanation helped the predictors and pairs where it didn't. You aren't told which until you've "
    "answered. Read the explanation, click the factors it cites, look at the <mark>highlighted</mark> words in the "
    "counterfactual, and answer one question. Scope, failure modes and near-ties are worked out from your tags, the generator's record of what changed, and the fitted decision rule.</p>"
    "<div class='note'><b>Keyboard:</b> <b>1 / 2 / 3</b> answers the question (Case 1 / Case 2 / Can't tell), "
    "<b>→</b> or <b>Enter</b> goes to the next row and <b>←</b> back. Labels save in this browser as you go.</div>"
    "<div class='controls'><label>Model<select id='cv-model'><option value=''>Both models</option>"
    + "".join(f"<option>{m}</option>" for m in COVER_MODELS) + "</select></label>"
    "<label>Row <span id='cv-count'></span><select id='cv-select'></select></label>"
    "<label>Show<select id='cv-filter'><option value=''>All</option><option value='todo'>Not yet labelled</option>"
    "<option value='done'>Labelled</option></select></label>"
    "<button id='cv-prev' type='button'>← Prev</button><button id='cv-next' type='button'>Next →</button>"
    "<button id='cv-download' type='button'>Download my labels (CSV)</button><span class='saved' id='cv-progress'></span></div>"
    "<div id='cv-view'></div>"
    "<h2>Original tagger sample</h2>"
)
# Gate 2a: hand-check the keyword-based "cited" label on 20 re-paired counterfactuals (experiment 2)
REPAIR = []
if Path("experiments/moral_machines/mm18_repair_check.csv").exists():
    rp = pd.read_csv("experiments/moral_machines/mm18_repair_check.csv")
    raw_prompts = pd.read_csv("data/raw/moral_machines_raw.csv", index_col=0)["prompt"]
    FACTOR_NAME = {k: n for k, n, _ in FACTORS} | {"numbers": "Number of lives"}
    for i, r in enumerate(rp.itertuples(), start=1):
        o_html, c_html = diff_html(scenario_text(raw_prompts[r.original_idx]), scenario_text(raw_prompts[r.counterfactual_idx]))
        REPAIR.append(dict(id=f"R{i:02d}|{r.model}|{r.original_idx}-{r.counterfactual_idx}", n=i, model=r.model,
                           o=int(r.original_idx), c=int(r.counterfactual_idx), factor=FACTOR_NAME.get(r.primary_factor, r.primary_factor),
                           factors=html.escape(str(r.original_factors)).replace(" | ", "<br>• "),
                           keywords=html.escape(str(r.cited_keywords)), orig_html=o_html, cf_html=c_html))
repair_check = (
    "<h2>Gate 2a: check the \"cited\" label on 20 re-paired counterfactuals</h2>"
    "<p class='explain'>Experiment 2 labels each new pair <i>cited</i> when the original scenario's own factor list mentions "
    "the one factor the counterfactual changes, by keyword match. These 20 pairs were all labelled <b>cited</b>. Is that "
    "right? Press <b>1</b> yes, <b>2</b> no, <b>3</b> unclear; <b>Enter</b> for the next pair. If more than 4 of 20 are wrong, "
    "I'll tighten the keyword map before running.</p>"
    "<div class='controls'><label>Pair <span id='rp-count'></span><select id='rp-select'></select></label>"
    "<button id='rp-prev' type='button'>← Prev</button><button id='rp-next' type='button'>Next →</button>"
    "<button id='rp-download' type='button'>Download my labels (CSV)</button><span class='saved' id='rp-progress'></span></div>"
    "<div id='rp-view'></div>"
) if REPAIR else ""
tagging = repair_check + coverage_tagging + (
    f"<p class='explain'>Hand-label which factors each explanation <b>gives as a reason</b> for its choice. The {N_RANDOM} random-sample "
    "labels are the answer key for an automatic cited-factor tagger. That tagger then labels every proprietary explanation, "
    "so we can test whether NSG is higher when a counterfactual changes a cited factor (H1), and build counterfactuals from "
    f"explanations (H2). The random sample: {N_RANDOM} original-scenario explanations from the 8 proprietary models, about 2 per moral "
    "theme per model, shuffled. It's saved as <code>experiments/moral_machines/mm18_tag_sample.csv</code>.</p>"
    "<div class='note'><b>How to label each factor</b><br>"
    "<b>Reason:</b> the explanation uses it to justify the choice, e.g. \"they were breaking the law, so…\" or \"saving more "
    "lives matters most\".<br>"
    "<b>Irrelevant:</b> it explicitly says the factor shouldn't count, e.g. \"gender is not a morally relevant factor\". "
    "These are the stated principles counterfactuals will test.<br>"
    "<b>Not used:</b> absent, or only <i>mentioned</i> while describing the scenario (\"the pedestrians were crossing on "
    "red\") without being used to decide.<br>"
    "Pick <i>reason</i> only when the explanation leans on it. If unsure, choose <i>unclear</i> and add a note.</div>"
    "<div class='note'><b>Two sets.</b> <b>Random sample</b>: representative, used to score the automatic tagger; label "
    "these first. <b>Failure cases</b>: the explanations behind proprietary explanation-misled rows and shared-failure pairs, "
    "for checking those failures against your own reading. Each one has buttons that open it in the other tabs. They're not "
    "representative, so they don't count towards the tagger score.</div>"
    "<div class='controls'><label>Set<select id='t-set'><option value='random'>Random sample (100)</option>"
    f"<option value='failures'>Failure cases ({sum(t['set'] == 'failures' for t in TAGS)})</option></select></label>"
    "<label>Explanation <span id='t-count'></span><select id='t-select'></select></label>"
    "<label>Show<select id='t-filter'><option value=''>All</option><option value='todo'>Not yet labelled</option>"
    "<option value='done'>Labelled</option></select></label>"
    "<button id='t-prev' type='button'>← Prev</button><button id='t-next' type='button'>Next →</button>"
    "<button id='t-download' type='button'>Download my labels (CSV)</button><span class='saved' id='t-progress'></span></div>"
    "<div id='t-view'></div>"
)

# %% Tab 6: decision rule (outputs of analysis/mm18_rule.py and analysis/mm18_coverage.py)
RULE_DIR = Path("experiments/moral_machines")
RULE_TOP3, RULE_CONTROL = ["GPT-5.2", "Gemini 3 Flash", "Gemini 3 Pro"], "Gemma 3 27B"
if (RULE_DIR / "mm18_rule_ceiling.csv").exists():
    ceil = pd.read_csv(RULE_DIR / "mm18_rule_ceiling.csv").set_index("model")
    rcoef = pd.read_csv(RULE_DIR / "mm18_rule_coefs.csv")
    rbound = pd.read_csv(RULE_DIR / "mm18_rule_boundary.csv")
    rtypes = pd.read_csv(RULE_DIR / "mm18_rule_pair_types.csv").set_index("what_changed").reindex(PAIR_TYPES).reset_index()
    rchecks = pd.read_csv(RULE_DIR / "mm18_rule_checks.csv")

    # Ceiling: per model, accuracy on the counterfactual answer
    fr1 = go.Figure()
    rows_m = list(ceil.index)
    for m in rows_m:
        r = ceil.loc[m]
        lo, hi = min(r["cf_acc_predictors_without"], r["cf_acc_rule"]), max(r["cf_acc_predictors_with"], r["cf_acc_rule"])
        fr1.add_scatter(x=[lo, hi], y=[m, m], mode="lines", line=dict(color=C_GRID, width=2), hoverinfo="skip", showlegend=False)
    for col, name, color in (("cf_acc_predictors_without", "predictors, no explanation", C2),
                             ("cf_acc_predictors_with", "predictors, with explanation", C1),
                             ("cf_acc_rule", "fitted rule (scenario factors only)", C3)):
        fr1.add_scatter(x=ceil[col], y=ceil.index, mode="markers", name=name,
                        marker=dict(color=color, size=10, line=dict(color="white", width=2)),
                        hovertemplate="<b>%{y}</b><br>" + name + ": %{x:.1%}<extra></extra>")
    fr1.add_hline(y=len(rows_m) - 1.5, line=dict(color=C_MUTED, width=1, dash="dash"))
    fr1.update_layout(LAYOUT, height=24 * len(rows_m) + 150, xaxis=dict(tickformat=".0%", title="accuracy on the counterfactual answer (held out)"),
                      yaxis=dict(categoryorder="array", categoryarray=rows_m, autorange="reversed"))
    ceil_tab = ceil.reset_index()[["model", "pairs", "cf_acc_copy_original", "cf_acc_predictors_without", "cf_acc_predictors_with",
                                   "cf_acc_rule", "nsg_explanation", "nsg_rule", "acc_all_decisions_rule", "acc_all_decisions_position_only"]]
    ceil_tab["explanation_share_of_rule_nsg"] = ceil_tab["nsg_explanation"] / ceil_tab["nsg_rule"]

    # Coefficients: top 3 + the control, dodged rows
    show = RULE_TOP3 + [RULE_CONTROL]
    feats = [f for f in rcoef[rcoef["model"] == RULE_TOP3[0]].sort_values("coef", key=abs, ascending=False)["feature"] if f != "intercept"]
    lab = rcoef.drop_duplicates("feature").set_index("feature")["label"]
    fr2 = go.Figure()
    for j, (m, color) in enumerate(zip(show, (C1, C2, C3, C4))):
        t = rcoef[rcoef["model"] == m].set_index("feature").reindex(feats)
        y = np.arange(len(feats)) + (j - 1.5) * 0.18
        fr2.add_scatter(x=t["coef"], y=y, mode="markers", name=m,
                        marker=dict(color=color, size=9, line=dict(color="white", width=2)),
                        error_x=dict(type="data", symmetric=False, array=t["hi"] - t["coef"], arrayminus=t["coef"] - t["lo"],
                                     color=color, thickness=1.5, width=0),
                        customdata=np.c_[t["lo"], t["hi"]], text=[lab[f] for f in feats],
                        hovertemplate="<b>" + m + "</b><br>%{text}<br>%{x:+.2f} [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]<extra></extra>")
    fr2.add_vline(x=0, line=dict(color=C_MUTED, width=1))
    fr2.update_layout(LAYOUT, height=34 * len(feats) + 150,
                      xaxis=dict(title="standardised coefficient (log-odds of choosing Case 1; 95% pair bootstrap)"),
                      yaxis=dict(tickvals=list(range(len(feats))), ticktext=[lab[f] for f in feats], autorange="reversed"))

    # Boundary: top 3 pooled, NSG and flip rate by P(flip) quintile (two panels, each with its own axis)
    bq = rbound[(rbound["scope"] == "top 3 pooled") & (rbound["split"] == "P(flip) quintile")].sort_values("level")
    xq = [f"Q{int(l)}<br>P(flip) ≈ {p:.0%}" for l, p in zip(bq["level"], bq["mean_p_flip"])]
    fr3 = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("NSG (95% CI)", "how often the reference model flipped"))
    fr3.add_bar(x=xq, y=bq["nsg"], marker_color=C1, marker_line=dict(color="white", width=2), showlegend=False,
                error_y=dict(type="data", symmetric=False, array=bq["nsg_hi"] - bq["nsg"], arrayminus=bq["nsg"] - bq["nsg_lo"],
                             color=C_MUTED, thickness=1.2, width=0),
                customdata=np.c_[bq["nsg_lo"], bq["nsg_hi"], bq["rows"]],
                hovertemplate="%{x}<br>NSG %{y:.1%} [%{customdata[0]:.1%}, %{customdata[1]:.1%}] · %{customdata[2]:.0f} pairs<extra></extra>",
                row=1, col=1)
    fr3.add_bar(x=xq, y=bq["flip_rate"], marker_color=C_OTHER, marker_line=dict(color="white", width=2), showlegend=False,
                hovertemplate="%{x}<br>flip rate %{y:.1%}<extra></extra>", row=1, col=2)
    fr3.add_hline(y=0, line=dict(color=C_MUTED, width=1), row=1, col=1)
    fr3.update_yaxes(tickformat=".0%")
    fr3.update_xaxes(tickangle=0, tickfont=dict(size=11))
    fr3.update_layout(LAYOUT, height=400, margin=dict(l=10, r=10, t=60, b=10))

    bt_tie = rbound[(rbound["scope"] == "top 3 pooled") & rbound["split"].str.startswith("near-tie")].copy()
    bt_tie["threshold"] = bt_tie["split"].str.extract(r"< ([\d.]+)")[0]
    fr4 = go.Figure()
    for level, color in (("clear", C1), ("near-tie", C2)):
        t = bt_tie[bt_tie["level"] == level]
        fr4.add_bar(x=["margin < " + th for th in t["threshold"]], y=t["nsg"], name=level, marker_color=color,
                    marker_line=dict(color="white", width=2),
                    error_y=dict(type="data", symmetric=False, array=t["nsg_hi"] - t["nsg"], arrayminus=t["nsg"] - t["nsg_lo"],
                                 color=C_MUTED, thickness=1.2, width=0),
                    customdata=np.c_[t["rows"], t["nsg_lo"], t["nsg_hi"]],
                    hovertemplate="<b>" + level + "</b> (%{x})<br>NSG %{y:.1%} [%{customdata[1]:.1%}, %{customdata[2]:.1%}]"
                                  " · %{customdata[0]:.0f} pairs<extra></extra>")
    fr4.add_hline(y=0, line=dict(color=C_MUTED, width=1))
    fr4.update_layout(LAYOUT, barmode="group", bargap=0.35, bargroupgap=0.08, height=380,
                      yaxis=dict(tickformat=".0%", title="NSG (95% CI)"),
                      xaxis_title="near-tie threshold: |P(Case 1 on the counterfactual) − 0.5|")
    tie = bt_tie[bt_tie["threshold"] == "0.2"].set_index("level")

    # Coverage labels (GPT-5.2)
    cov_block = ""
    if (RULE_DIR / "mm18_coverage_labels_scored.csv").exists():
        cl = pd.read_csv(RULE_DIR / "mm18_coverage_labels_scored.csv")
        cl = cl[cl["model"] == "GPT-5.2"]
        fl, sc = cl[cl["outcome"] == "failure"], cl[cl["outcome"] == "success"]
        sep = pd.DataFrame({
            "measure": ["changed factor is cited (any head-count change counts)", "near-tie under the fitted rule",
                        "your literal reading was wrong", "your literal reading was \"can't tell\"", "fitted rule got it right"],
            "failures": [fl["in_scope"].sum(), fl["near_tie"].sum(), (fl["literal_reading"] == "wrong").sum(),
                         (fl["literal_reading"] == "can't tell").sum(), fl["rule_right"].sum()],
            "successes": [sc["in_scope"].sum(), sc["near_tie"].sum(), (sc["literal_reading"] == "wrong").sum(),
                          (sc["literal_reading"] == "can't tell").sum(), sc["rule_right"].sum()]})
        sep["failures"] = sep["failures"].astype(int).astype(str) + f" / {len(fl)}"
        sep["successes"] = sep["successes"].astype(int).astype(str) + f" / {len(sc)}"
        CELLS = [("wrong", True, "The explanation misstates what the model consistently does: strongest unfaithfulness candidates."),
                 ("wrong", False, "Neither the explanation nor the rule predicts the choice: consistent with non-determinism, or with interactions the main-effects rule omits."),
                 ("can't tell", True, "The explanation leaves out the weighting that decides this pair: incomplete, not wrong."),
                 ("can't tell", False, "Hard for everything."),
                 ("right", None, "The explanation settles it correctly; the predictors erred.")]
        trs = []
        for lit, rr, meaning in CELLS:
            g = fl[(fl["literal_reading"] == lit) & ((fl["rule_right"] == rr) if rr is not None else True)]
            ex = g[g["notes"].notna()].sort_values("near_tie").head(1)  # prefer a clear (non-near-tie) example
            ex_html = (f"#{int(ex['n'].iloc[0])} ({ex['what_changed'].iloc[0]}): <i>“{html.escape(ex['notes'].iloc[0].strip())}”</i>"
                       if len(ex) else "—")
            trs.append(f"<tr><td>{lit}</td><td>{'—' if rr is None else ('right' if rr else 'wrong')}</td><td>{len(g)}</td>"
                       f"<td>{', '.join('#' + str(int(n)) for n in g['n'])}</td><td>{meaning}</td><td>{ex_html}</td></tr>")
        cov_block = (
            "<h2>Hand labels: why the GPT-5.2 failures fail</h2>"
            f"<p class='explain'>{len(cl)} GPT-5.2 rows from the Coverage sample ({len(fl)} failures, {len(sc)} successes; 4 failures "
            "and 2 successes per pair type, egregious failures first). It's a stratified sample, so read the counts as a "
            "description, not as rates. <b>Whether the counterfactual changes a cited factor doesn't separate failures from "
            "successes</b>: every generated pair changes head counts, and frontier explanations cite nearly every factor. Being a "
            "near-tie under the fitted rule does separate them. Your literal reading was wrong on 9 failures and no successes, "
            "but that is partly built in: you read the same text as the predictors.</p>"
            "<div class='table-scroll'><table><thead><tr><th>Measure</th><th>Failures</th><th>Successes</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{r.measure}</td><td>{r.failures}</td><td>{r.successes}</td></tr>" for r in sep.itertuples())
            + "</tbody></table></div>"
            "<h4>The failures, by your literal reading × the fitted rule</h4>"
            "<div class='table-scroll'><table><thead><tr><th>Literal reading</th><th>Fitted rule</th><th>n</th><th>Rows</th>"
            "<th>What it means</th><th>Example (your note)</th></tr></thead><tbody>" + "".join(trs) + "</tbody></table></div>"
            "<p class='explain'>Several notes describe a <b>conditional rule whose condition fails in the counterfactual</b>, "
            "e.g. \"protect law-abiding pedestrians as a tie-breaker when casualties are equal\", on a pair where casualties "
            "differ or every pedestrian is breaking the law. Citing a factor isn't the same as covering the counterfactual.</p>")

    g52 = ceil.loc["GPT-5.2"]
    rule_tab = (
        "<p class='explain'>For each reference model, an L2 logistic regression of \"chose Case 1\" on the scenario factors "
        "(net lives saved, net animals saved, swerve vs. stay, passengers vs. pedestrians, law/light, the theme's groups, "
        "answer position), cross-validated with pairs kept whole. It's an empirical \"known rule\" for each real model, and "
        "it gives a formal distance to the decision boundary. Built by <code>analysis/mm18_rule.py</code>; plan in "
        "<code>moral_machines_rule_fit_plan.md</code>.</p>"
        + kpis([(f"{ceil.loc[RULE_TOP3, 'acc_all_decisions_rule'].min():.0%}–{ceil.loc[RULE_TOP3, 'acc_all_decisions_rule'].max():.0%}",
                 "held-out accuracy of the fitted rule on all decisions, top 3 models"),
                (f"{(ceil.loc[RULE_TOP3, 'nsg_explanation'] / ceil.loc[RULE_TOP3, 'nsg_rule']).min():.0%}–"
                 f"{(ceil.loc[RULE_TOP3, 'nsg_explanation'] / ceil.loc[RULE_TOP3, 'nsg_rule']).max():.0%}",
                 "of the rule's gain the explanation recovers (explanation NSG ÷ rule NSG), top 3"),
                (f"{tie.loc['near-tie', 'nsg']:.0%} vs {tie.loc['clear', 'nsg']:.0%}",
                 "NSG on near-tie vs. clear counterfactuals (margin < 0.2), top 3 pooled; paper's pairs only, didn't "
                 "replicate on purpose-built pairs (Experiments tab)")])
        + section("Rule ceiling: how predictable each model is from scenario factors alone",
                  f"Accuracy on the counterfactual answer. <b style='color:{C2}'>Predictors without</b> and "
                  f"<b style='color:{C1}'>with</b> the explanation are the existing NSG inputs; "
                  f"<b style='color:{C3}'>the fitted rule</b> predicts each held-out pair from the scenario alone. It isn't a "
                  f"strict ceiling: the rule learns from about 1,300 of the model's own decisions, the predictors see one "
                  f"explanation. GPT-5.2: {g52['cf_acc_predictors_without']:.1%} without, {g52['cf_acc_predictors_with']:.1%} "
                  f"with, {g52['cf_acc_rule']:.1%} rule. The weaker proprietary models are about as predictable as the strong "
                  "ones; their explanations recover much less of it. Gemma 3 27B (below the line) is the known-unfaithful control.",
                  fr1, ceil_tab)
        + section("What drives each model's choices",
                  "Positive means the factor pushes towards Case 1. Features are standardised, so bars are comparable within a "
                  "model. The law is the largest effect for all three top models; gender is about zero, matching their \"gender "
                  "shouldn't matter\" explanations. Gemma 3 27B's largest effect is the answer position, a cause its explanations "
                  "never mention. The proprietary models have small but non-zero position effects too. Don't read the "
                  "intercept (in the table) as a Case 1 bias: the generator's case layout is asymmetric (Case 1 always kills "
                  "the passengers when there are any).",
                  fr2, rcoef[rcoef["model"].isin(show)])
        + section("Distance to the decision boundary (top 3 models pooled)",
                  "Each pair's <b>P(flip)</b> is the rule's held-out probability that the counterfactual answer differs from "
                  "the original one. Pairs are split into quintiles of P(flip) within each model. NSG is highest where the rule "
                  "confidently expects a flip; it isn't a smooth climb, and the lowest quintile's interval includes 0.",
                  fr3, bq)
        + section("Near-ties carry little explanation signal (paper's pairs only; didn't replicate)",
                  "A counterfactual is a <b>near-tie</b> when the rule's P(Case 1) on it is close to 0.5. On the paper's pairs "
                  "the gap holds at all three thresholds. <b>It didn't replicate on 200 purpose-built single-factor pairs</b> "
                  "(experiment 2, Experiments tab: 21% vs. 22%; difference −1 point [−24, +22]), so treat it as observational, "
                  "not causal.", fr4, bt_tie)
        + "<h2>Where each pair type sits (top 3 models)</h2><p class='explain'>Counts-only pairs are at the \"too close\" end: "
          "the rule rarely expects a flip, so there's little for an explanation to predict.</p>"
        + "<div class='table-scroll'>" + rtypes.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}") + "</div>"
        + cov_block
        + "<h2>Sanity checks</h2><p class='explain'>Fixed before fitting. The Gemma position-only check failed: position is "
          "Gemma's largest effect but it also responds to content (law, gender, passengers), so it is position-dominated "
          "rather than position-only. It still works as the unfaithful control.</p>"
        + "<div class='table-scroll'>" + rchecks.to_html(index=False, border=0) + "</div>")
else:
    rule_tab = "<p class='explain'>Run <code>python analysis/mm18_rule.py</code> first.</p>"

# %% Tab 7: experiments (API runs; outputs of analysis/mm18_gt.py)
if (RULE_DIR / "mm18_gt_results.csv").exists():
    gres = pd.read_csv(RULE_DIR / "mm18_gt_results.csv")
    gcon = pd.read_csv(RULE_DIR / "mm18_gt_contrasts.csv")
    gdrift = pd.read_csv(RULE_DIR / "mm18_gt_drift.csv")
    gceil = pd.read_csv(RULE_DIR / "mm18_gt_heldout_rule.csv")
    gceil = (gceil["rule_answer"] == gceil["reference_answer"]).groupby(gceil["model"]).mean()
    gtexts = json.loads((RULE_DIR / "mm18_gt_explanations.json").read_text())
    GT_MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemma 3 27B"]
    GT_CONDS = [("none", "no explanation", C_OTHER), ("own", "model's own explanation", C1),
                ("gt", "ground truth (written)", C3), ("tree", "ground truth (decision tree)", C4),
                ("gt+own", "written ground truth + own", C2)]
    pooled = gres[gres["simulator"] == "both simulators"]
    fe = go.Figure()
    for j, (cond, name, color) in enumerate(GT_CONDS):
        t = pooled[pooled["condition"] == cond].set_index("model").reindex(GT_MODELS)
        y = np.arange(len(GT_MODELS)) + (j - 2) * 0.14
        fe.add_scatter(x=t["acc"], y=y, mode="markers", name=name,
                       marker=dict(color=color, size=10, line=dict(color="white", width=2)),
                       error_x=dict(type="data", symmetric=False, array=t["acc_hi"] - t["acc"], arrayminus=t["acc"] - t["acc_lo"],
                                    color=color, thickness=1.5, width=0),
                       customdata=np.c_[t["acc_lo"], t["acc_hi"], t["nsg"]], text=GT_MODELS,
                       hovertemplate="<b>%{text}</b> · " + name + "<br>accuracy %{x:.1%} [%{customdata[0]:.1%}, "
                                     "%{customdata[1]:.1%}]<br>NSG %{customdata[2]:+.1%}<extra></extra>")
    fe.add_scatter(x=[gceil[m] for m in GT_MODELS], y=list(range(len(GT_MODELS))), mode="markers", name="fitted rule (no simulator)",
                   marker=dict(symbol="line-ns-open", color=C_INK, size=22, line=dict(width=2)),
                   hovertemplate="<b>%{y}</b><br>training-only fitted rule: %{x:.0%}<extra></extra>")
    fe.update_layout(LAYOUT, height=420, xaxis=dict(tickformat=".0%", title="accuracy on the counterfactual answer (100 held-out pairs per model; 95% pair bootstrap)"),
                     yaxis=dict(tickvals=list(range(len(GT_MODELS))), ticktext=GT_MODELS, autorange="reversed"),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    ct = gcon[gcon["simulator"] == "both simulators"].copy()
    ct["NSG difference [95% CI]"] = ct.apply(lambda r: f"{r.nsg_diff:+.0%} [{r.lo:+.0%}, {r.hi:+.0%}]", axis=1)
    ctab = ct.pivot(index="contrast", columns="model", values="NSG difference [95% CI]").reindex(
        ["gt - own", "tree - own", "gt+own - gt", "gt - tree"])[GT_MODELS].rename_axis(columns=None).reset_index()
    texts_html = "".join(
        f"<details><summary>{m}: the two ground-truth texts</summary><h4>Written (Claude, from training coefficients + "
        f"40 training decisions, checked claim by claim)</h4><pre>{html.escape(gtexts[m]['text'])}</pre>"
        f"<h4>Decision tree (automatic, depth 3, training decisions)</h4><pre>{html.escape(gtexts[m]['tree_text'])}</pre></details>"
        for m in GT_MODELS)
    g_acc = pooled.set_index(["model", "condition"])["acc"]
    # Gemma answer-format test (fresh reference calls, same scenario both ways)
    exp3_html = ""
    if (RULE_DIR / "mm18_format_results.csv").exists():
        fm = pd.read_csv(RULE_DIR / "mm18_format_results.csv").set_index(["group", "metric"])
        fs = pd.read_csv(RULE_DIR / "mm18_format_results_strict.csv").set_index(["group", "metric"])
        G, C, K = "Gemma 3 27B (fresh)", "Gemini 3 Flash control (fresh)", "calibration (20 paper prompts, unchanged)"
        v = lambda grp, met, t=fm: t.loc[(grp, met)]
        f3 = go.Figure()
        for fmt, color in (("answer first", C1), ("answer last", C2)):
            ys = [v(G, f"Case 1 rate, {fmt}"), v(C, f"Case 1 rate, {fmt}")]
            f3.add_bar(x=["Gemma 3 27B", "Gemini 3 Flash (control)"], y=[y["value"] for y in ys], name=f"[ANSWER] asked {fmt.split()[1]}",
                       marker_color=color, marker_line=dict(color="white", width=2),
                       error_y=dict(type="data", symmetric=False, array=[y["hi"] - y["value"] for y in ys],
                                    arrayminus=[y["value"] - y["lo"] for y in ys], color=C_MUTED, thickness=1.2, width=0),
                       hovertemplate="%{x}<br>Case 1 rate %{y:.0%}<extra>" + fmt + "</extra>")
        f3.update_layout(LAYOUT, barmode="group", bargap=0.35, bargroupgap=0.08, height=380,
                         yaxis=dict(tickformat=".0%", range=[0, 1.05], title="share of answers that are Case 1 (95% CI)"))
        tab = fm.reset_index()[["group", "metric", "value", "lo", "hi", "n", "note"]]
        calib_match = v(K, "fresh answer = paper's stored answer")["value"]
        exp3_html = (
            "<h2>Gemma answer-format test: the same scenario asked both ways</h2>"
            "<p class='explain'><b>Question:</b> does the answer format <i>cause</i> Gemma's choice, and does the effect hold in "
            "fresh runs? In the paper's data each scenario was asked in only one format. Here 100 held-out scenarios were each "
            "asked answer-first and answer-last, 3 samples each, with fresh Gemma 3 27B calls (the paper's sampling: "
            "temperature 1.0, top-p 0.95, top-k 64). Control: Gemini 3 Flash on the same prompts. Calibration: 20 of the paper's "
            "original prompts, unchanged. Prompts are the repo's reference prompts; the two formats differ only in where "
            "[ANSWER] is requested. 860 calls, $1.10. Code: <code>analysis/mm18_format_run.py</code>, <code>analysis/mm18_format.py</code>.</p>"
            + kpis([(f"{v(G, 'Case 1 rate, answer first')['value']:.0%} → {v(G, 'Case 1 rate, answer last')['value']:.0%}",
                     "Gemma: Case 1 rate, answer asked first → last (same scenarios)"),
                    (f"{v(G, 'scenarios whose majority answer flips with the format')['value']:.0%}",
                     "of scenarios change Gemma's answer when only the format changes; all of them towards Case 1 when answering last"),
                    (f"{v(C, 'scenarios whose majority answer flips with the format')['value']:.0%}", "for the Gemini 3 Flash control")])
            + section("Case 1 rate by answer format",
                      f"Two samples of the same prompt disagree {v(G, 'two samples disagree: same format')['value']:.0%} of the time; "
                      f"two samples of the same scenario in different formats disagree {v(G, 'two samples disagree: different formats')['value']:.0%}. "
                      f"On the 20 unchanged paper prompts, fresh answers match the stored ones {calib_match:.0%} "
                      "of the time, so the serving setup reproduces the paper's behaviour. Gemma's explanations still argue from "
                      "content when the format flips its answer, e.g. \"prioritizing the safety of those adhering to the law\" "
                      "attached to the answer that kills the lawful crossers (scenario 341, answer asked last).",
                      f3, tab)
            + "<p class='explain'><b>Parser note.</b> Asked for its answer first, Gemma often writes a bare \"CASE 2\" first "
              "line without the [ANSWER] tag. The repo parser drops these (120 of 360 answer-first calls, 92 of them Case 2), which "
              f"inflates the answer-first Case 1 rate. The numbers above take that leading answer; with the strict parser the "
              f"rates are {v(G, 'Case 1 rate, answer first', fs)['value']:.0%} vs. {v(G, 'Case 1 rate, answer last', fs)['value']:.0%}.</p>")
    # Experiment 2 (re-paired counterfactuals)
    exp2_html = "<h2>Experiment 2: purpose-built counterfactuals</h2><p class='explain'>Not run yet.</p>"
    if (RULE_DIR / "mm18_repair_results.csv").exists():
        r2 = pd.read_csv(RULE_DIR / "mm18_repair_results.csv")
        c2 = pd.read_csv(RULE_DIR / "mm18_repair_contrasts.csv")
        tie = r2.query("arm == 'main' and split == 'tie'")
        order = ["GPT-5.2", "Gemini 3 Flash", "GPT-5.2 + Gemini 3 Flash", "Gemma 3 27B"]
        f2e = go.Figure()
        for level, color in (("clear", C1), ("near-tie", C2)):
            t = tie[tie["level"] == level].set_index("scope").reindex(order)
            f2e.add_bar(x=order, y=t["nsg_own"], name=level, marker_color=color, marker_line=dict(color="white", width=2),
                        error_y=dict(type="data", symmetric=False, array=t["nsg_own_hi"] - t["nsg_own"],
                                     arrayminus=t["nsg_own"] - t["nsg_own_lo"], color=C_MUTED, thickness=1.2, width=0),
                        customdata=np.c_[t["acc_none"], t["acc_own"], t["pairs"], t["nsg_own_lo"], t["nsg_own_hi"]],
                        hovertemplate="<b>%{x}</b> · " + level + "<br>NSG %{y:.1%} [%{customdata[3]:.1%}, %{customdata[4]:.1%}]"
                                      "<br>accuracy %{customdata[0]:.1%} without → %{customdata[1]:.1%} with own explanation"
                                      "<br>%{customdata[2]:.0f} pairs<extra></extra>")
        f2e.add_hline(y=0, line=dict(color=C_MUTED, width=1))
        f2e.update_layout(LAYOUT, barmode="group", bargap=0.3, bargroupgap=0.08, height=400,
                          yaxis=dict(tickformat=".0%", title="NSG of the model's own explanation (95% CI)"))
        pc = c2.query("scope == 'GPT-5.2 + Gemini 3 Flash' and contrast == 'near-tie - clear'").iloc[0]
        fr = tie.query("scope == 'GPT-5.2 + Gemini 3 Flash'").set_index("level")
        gta = r2.query("arm == 'gemma ground-truth arm'").set_index("level")
        cit = r2.query("arm == 'main' and split == 'cited' and scope == 'GPT-5.2 + Gemini 3 Flash'").set_index("level")
        gtab = gta.reset_index()[["level", "pairs", "acc_none", "acc_own", "acc_gt", "nsg_own", "nsg_gt", "nsg_gt_lo", "nsg_gt_hi"]]
        exp2_html = (
            "<h2>Experiment 2: purpose-built counterfactuals</h2>"
            "<p class='explain'><b>Question:</b> on counterfactuals chosen on purpose rather than by the generator, is the "
            "explanation's gain lower near the model's decision boundary? New pairs were built from scenarios each model "
            "had already answered, so no new reference-model calls were needed. Per model: 50 near-tie + 50 clear pairs, "
            "each changing exactly one factor; traffic-light changes capped at half; flips balanced 50/50; no scenario used "
            "twice. Simulators: gpt-oss-20b (8,000-token cap) and Gemini 3 Flash, 2 repeats each; conditions: own "
            "explanation vs. none. Code: <code>analysis/mm18_repair_sample.py</code>, <code>analysis/mm18_repair_analysis.py</code>. "
            "Cost about $1.</p>"
            + kpis([(f"{fr.loc['near-tie', 'nsg_own']:.0%} vs {fr.loc['clear', 'nsg_own']:.0%}",
                     "GPT-5.2 + Gemini 3 Flash: NSG on near-tie vs. clear pairs"),
                    (f"{pc['nsg_diff'] * 100:+.0f} pts [{pc['lo'] * 100:+.0f}, {pc['hi'] * 100:+.0f}]",
                     "difference: the near-tie gap seen on the paper's pairs does not replicate"),
                    (f"{gta.loc['all', 'acc_own']:.1%} → {gta.loc['all', 'acc_gt']:.1%}",
                     "Gemma, new held-out-scenario pairs: own explanation → behaviour description")])
            + section("Near-tie vs. clear: explanation gain is about the same",
                      f"Near-ties are harder to predict with or without the explanation (pooled frontier accuracy "
                      f"{fr.loc['near-tie', 'acc_own']:.0%} vs. {fr.loc['clear', 'acc_own']:.0%} with it), but the explanation "
                      "closes about the same share of the gap. This isn't a flip-rate effect: on the paper's pairs the "
                      "near-tie and clear groups flipped almost equally often (31.5% vs. 30.8%). Untested candidate reasons "
                      "for the difference from the paper's pairs: single-factor changes (half of them the traffic light), "
                      "the 50% flip balance changing which clear pairs are included, 2 simulators vs. the paper's 5, and the "
                      "generator's mix of pair types. Gemma's own explanation lowers accuracy in both groups.",
                      f2e, tie[["scope", "level", "pairs", "flip_rate", "acc_none", "acc_own", "nsg_own", "nsg_own_lo", "nsg_own_hi"]])
            + "<h2>Cited vs. uncited: confounded</h2><p class='explain'>Flips were balanced across near-tie/clear, not across "
              f"this label, so cited pairs flip {cit.loc['cited', 'flip_rate']:.0%} of the time against "
              f"{cit.loc['uncited', 'flip_rate']:.0%} for uncited (frontier pooled). The NSG comparison can't be interpreted. "
              "It's consistent with the hand labels: no evidence that citing the changed factor matters. \"Cited\" = the "
              "original's factor list mentions the one factor the pair changes (keyword match, checked by hand on 20 pairs: "
              "agrees with all 20 after a fix for substring matches).</p>"
            + "<h2>Gemma: the behaviour description still helps on new counterfactuals</h2><p class='explain'>50 new pairs "
              "built only from experiment 1's held-out scenarios, so the description is never tested on scenarios it was "
              "written from. Third confirmation of the hidden cause. Only 4 of these pairs are near-ties: too few to read.</p>"
            + "<div class='table-scroll'>" + gtab.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}") + "</div>"
            + "<p class='explain'><b>Simulator consistency:</b> the same prompt sent twice gets the same answer 93–98% of the "
              "time from Gemini 3 Flash and 84–87% from gpt-oss-20b.</p>")
    exp_tab = (
        "<p class='explain'><b>Experiment 1: does a description of how the model actually decides beat its own "
        "explanation?</b> For GPT-5.2, Gemini 3 Flash and Gemma 3 27B, 100 held-out pairs per model (20 per pair type; the "
        "paper's own counterfactual pairs). Two simulators (gpt-oss-20b, Gemini 3 Flash) predict the counterfactual answer "
        "under five conditions. The ground-truth texts were written from training pairs only. Every prompt adds one line "
        "stating whether the evaluator was asked for its answer before or after its reasoning, so a simulator can use an "
        "answer-position rule. Plan: <code>moral_machines_experiments_README.md</code>; code: <code>analysis/mm18_gt_run.py</code>, "
        "<code>analysis/mm18_gt.py</code>. Cost about $1.12.</p>"
        + kpis([(f"{g_acc['Gemma 3 27B', 'own']:.0%} → {g_acc['Gemma 3 27B', 'gt']:.0%}",
                 "Gemma 3 27B: accuracy with its own explanation → with a description of its behaviour"),
                (f"{g_acc['GPT-5.2', 'own']:.1%} vs {g_acc['GPT-5.2', 'gt']:.1%}",
                 "GPT-5.2: own explanation vs. written ground truth (no evidence the description adds anything)"),
                (f"{g_acc['Gemini 3 Flash', 'own']:.1%} vs {g_acc['Gemini 3 Flash', 'gt']:.1%}",
                 "Gemini 3 Flash: own explanation vs. written ground truth")])
        + section("Accuracy by condition",
                  "Each dot is one condition, pooled over both simulators. The black tick is the training-only fitted rule's own "
                  "accuracy on the same pairs, with no simulator involved. <b>Gemma</b>: its own explanation is worse than "
                  "nothing, and either ground-truth text (written or automatic) recovers most of the gap: the explanation hides "
                  "the answer-position effect. <b>GPT-5.2 and Gemini 3 Flash</b>: their own explanations do as well as or "
                  "better than a behaviour description, and adding the description doesn't help. Simulators given the written "
                  "rule stay below the fitted rule's own accuracy, so the rule's accuracy isn't a ceiling a reader can reach.",
                  fe, pooled[["model", "condition", "label", "answers", "acc", "acc_lo", "acc_hi", "nsg", "nsg_lo", "nsg_hi"]])
        + "<h2>Paired differences in NSG (pooled over both simulators)</h2><p class='explain'>Same pairs, different "
          "explanation; 95% pair-bootstrap intervals. For the frontier models the no-explanation baseline is about 80%, so NSG "
          "differences are wide: lead with accuracy. The Gemini 3 Flash gt − own gap moved from −15 to −40 points between two "
          "runs of the same prompts, so simulator sampling noise is large too.</p>"
        + "<div class='table-scroll'>" + ctab.to_html(index=False, border=0) + "</div>"
        + "<h2>The ground-truth texts</h2>" + texts_html
        + "<h2>Caveats</h2><ul class='findings'>"
          "<li><b>Self-simulation:</b> in the Gemini 3 Flash column, one of the two simulators is Gemini 3 Flash itself.</li>"
          "<li><b>Not comparable with the paper's NSG:</b> held-out sets are stratified (20 per pair type), and gpt-oss-20b samples "
          "at temperature 0.6. Its new answers agree with the paper's stored ones 74–92% of the time (table below). Compare "
          "conditions within this experiment only.</li>"
          "<li><b>Token cap:</b> gpt-oss-20b hit the paper's 2,000-token cap on 36% of decision-tree prompts. All gpt-oss calls "
          "were re-run at 8,000 tokens so every condition uses the same cap.</li>"
          "<li><b>The written ground truth involves judgment.</b> The automatic tree text is the check: it reproduces the Gemma "
          "result without that judgment.</li></ul>"
        + "<h4>Drift check: new answers vs. the paper's stored answers (same simulator, pair and prompt)</h4>"
        + "<div class='table-scroll'>" + gdrift.to_html(index=False, border=0, float_format=lambda v: f"{v:.2f}") + "</div>"
        + exp2_html + exp3_html)
else:
    exp_tab = "<p class='explain'>Run <code>python analysis/mm18_gt_run.py --exp 1</code> and <code>python analysis/mm18_gt.py</code> first.</p>"

# %% Write the page
dump = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
page = (TEMPLATE.read_text()
        .replace("__OVERVIEW__", overview).replace("__DRIVERS__", drivers).replace("__MISLEADING__", misleading)
        .replace("__SHARED__", shared).replace("__TAGGING__", tagging).replace("__RULE__", rule_tab).replace("__EXP__", exp_tab).replace("__PAIRS__", dump(PAIRS_BY_VIEW)).replace("__EXPL__", dump(EXPL))
        .replace("__MISLED__", dump(MISLED)).replace("__TAGS__", dump(TAGS)).replace("__COVER__", dump(COVER)).replace("__REPAIR__", dump(REPAIR))
        .replace("__FACTORS__", dump(FACTORS)))
OUT.write_text(page)
print(f"wrote {OUT} ({len(page) / 1e6:.1f} MB): {N_RANDOM} random + {len(TAGS) - N_RANDOM} failure explanations to tag, {len(MISLED)} misled rows, "
      + ", ".join(f"{v} {len(p)} shared pairs" for v, p in PAIRS_BY_VIEW.items()))
