"""Plots for the Moral Machines egregious-unfaithfulness filter outputs.

Reads all_pairs.csv, egregious_candidates.csv, comparison_set.csv and summary.json (written by
moral_machines/filter_egregious.py) and writes one PNG per plot, all_plots.pdf, captions.md and index.html.

Usage (from the repo root):
    python -m moral_machines.plot_egregious
    python -m moral_machines.plot_egregious --thinking medium --configs 'claude'

"Model" in every plot means the reference config (model + thinking level, plus a [file] tag for pairs that
appear in two input files). Each config keeps the same colour in every plot and across filtered runs.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from moral_machines.filter_egregious import EDIT_TYPES, killed_location, parse_scenario  # noqa: E402

N_BOOT = 2000
SEED = 0
NCOLS = 4
EDIT_SHORT = {"law_moved": "law\nmoved", "species_changed": "species\nchanged",
              "margin_changed": "margin\nchanged", "size_changed": "size\nchanged",
              "demographic_reshuffle": "demographic\nreshuffle", "within_category_reshuffle": "within-cat.\nreshuffle"}
PASSENGER_RE = re.compile(r"passeng|occupant|inside the car|trusted|accepted (the )?risk", re.I)


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def to_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else
                 {"true": True, "false": False}.get(str(v).strip().lower(), np.nan))


def boot(values, stat="mean", n_boot=N_BOOT, seed=SEED):
    """(estimate, lo, hi) with a 95% percentile bootstrap over the entries of `values` (pairs)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    f = np.mean if stat == "mean" else np.sum
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    draws = f(v[idx], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(f(v)), float(lo), float(hi)


def config_colors(configs) -> dict:
    """Stable colour per config: sorted order over every config in all_pairs (not the filtered subset)."""
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(configs))}


def conf_level(x) -> str:
    """HIGH / MEDIUM / LOW from the stated confidence; 'MEDIUM-HIGH' and free text -> OTHER."""
    if not isinstance(x, str) or not x.strip():
        return "MISSING"
    m = re.match(r"\s*(HIGH|MEDIUM|LOW)\b(?!-)", x.upper())
    return m.group(1) if m else "OTHER"


def short(c: str, width=40) -> str:
    """Display name: drop the vendor prefix and '_predictions' from file tags, then truncate."""
    c = re.sub(r"^[^/ ]+/", "", str(c)).replace("_predictions]", "]")
    return c if len(c) <= width else c[: width - 1] + "…"


def grid(n, panel_w=4.2, panel_h=3.2, sharey=True):
    ncols = min(NCOLS, max(n, 1))
    nrows = int(np.ceil(max(n, 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows + 0.8),
                             sharey=sharey, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def err(est, lo, hi):
    """Asymmetric error for one bar as a plain list (a (2,1) ndarray trips a NumPy 2.x deprecation in mpl)."""
    return [[max(float(est) - float(lo), 0.0)], [max(float(hi) - float(est), 0.0)]]


def helped_mask(df: pd.DataFrame, args: dict) -> pd.Series:
    """Mirror of the egregious rule: all simulators right WITH the explanation, a majority wrong WITHOUT.

    Uses the thresholds stored in summary.json args; None -> per-row defaults (all / majority).
    """
    n = df["n_simulators"]
    right_with = n - df["n_wrong_with"]
    wrong_without = n - df["n_right_without"]
    mw = n if args.get("min_wrong_with") is None else args["min_wrong_with"]
    mr = n // 2 + 1 if args.get("min_right_without") is None else args["min_right_without"]
    return (right_with >= mw) & (wrong_without >= mr)


# ---------------------------------------------------------------------------------------------
# Pure computations (tested directly)
# ---------------------------------------------------------------------------------------------
def check_counts(all_pairs, egr, summary):
    """Egregious counts per config must agree across summary.json, egregious_candidates.csv, all_pairs.csv."""
    s = {c: v["egregious"] for c, v in summary["counts_per_ref_config"].items()}
    e = egr.groupby("ref_config").size().to_dict()
    a = all_pairs[all_pairs["egregious"]].groupby("ref_config").size().to_dict()
    bad = []
    for c in sorted(set(s) | set(e) | set(a)):
        trio = (s.get(c, 0), e.get(c, 0), a.get(c, 0))
        if len(set(trio)) > 1:
            bad.append(f"{c}: summary={trio[0]} candidates_csv={trio[1]} all_pairs={trio[2]}")
    if bad:
        raise ValueError("egregious counts disagree with summary.json:\n  " + "\n  ".join(bad))


def assert_matched(egr, comp, summary):
    """Comparison counts per (config, edit_type, answer_changed) = egregious counts minus logged shortfalls."""
    keys = ["ref_config", "edit_type", "answer_changed"]
    need = egr.groupby(keys).size()
    got = comp.groupby(keys).size().reindex(need.index, fill_value=0)
    expected = need.copy()
    for sf in summary.get("comparison_shortfalls", []):
        k = (sf["ref_config"], sf["edit_type"], bool(sf["answer_changed"]))
        if k in expected.index:
            expected[k] = sf["available"]
    extra = set(comp.groupby(keys).size().index) - set(need.index)
    assert not extra, f"comparison rows in groups with no egregious rows: {sorted(extra)[:5]}"
    diff = (got != expected)
    assert not diff.any(), f"comparison set not matched on {keys}: {got[diff].to_dict()} vs {expected[diff].to_dict()}"


def edit_shares(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """Share of each edit type per (ref_config, set) where set is 'all' or 'egregious'. Rows sum to 1."""
    parts = []
    for name, d in (("all", all_pairs), ("egregious", all_pairs[all_pairs["egregious"]])):
        t = pd.crosstab(d["ref_config"], d["edit_type"], normalize="index")
        t["set"] = name
        parts.append(t.set_index("set", append=True))
    out = pd.concat(parts).fillna(0.0)
    out = out.reindex(columns=[e for e in EDIT_TYPES if e in out.columns]
                      + sorted(c for c in out.columns if c not in EDIT_TYPES))
    sums = out.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), f"edit-type shares do not sum to 1: {sums[~np.isclose(sums, 1)]}"
    return out


def egregious_rate_by_edit(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """P(egregious | edit type) per config, with n, k and a 95% bootstrap CI over pairs."""
    rows = []
    for (c, et), g in all_pairs.groupby(["ref_config", "edit_type"]):
        est, lo, hi = boot(g["egregious"].astype(float))
        rows.append({"ref_config": c, "edit_type": et, "n": len(g), "k": int(g["egregious"].sum()),
                     "rate": est, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def accuracy_by_group(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean simulator accuracy with/without explanation per config x edit type x answer changed."""
    d = all_pairs.assign(acc_with=1 - all_pairs["n_wrong_with"] / all_pairs["n_simulators"],
                         acc_without=all_pairs["n_right_without"] / all_pairs["n_simulators"])
    d["gain"] = d["acc_with"] - d["acc_without"]
    rows = []
    for (c, et, ch), g in d.groupby(["ref_config", "edit_type", "answer_changed"]):
        r = {"ref_config": c, "edit_type": et, "answer_changed": bool(ch), "n": len(g)}
        for col in ("acc_with", "acc_without", "gain"):
            r[col], r[col + "_lo"], r[col + "_hi"] = boot(g[col])
        rows.append(r)
    return pd.DataFrame(rows)


def jaccard_matrix(all_pairs: pd.DataFrame):
    """Jaccard of egregious pair keys between configs, restricted to pairs eligible in BOTH configs."""
    d = all_pairs.assign(key=all_pairs["original_question_idx"].astype(str) + "|"
                         + all_pairs["counterfactual_question_idx"].astype(str))
    configs = sorted(d["ref_config"].unique())
    elig = d.groupby("ref_config")["key"].apply(set)
    egr = d[d["egregious"]].groupby("ref_config")["key"].apply(set).reindex(configs)
    egr = egr.apply(lambda v: v if isinstance(v, set) else set())
    J = np.full((len(configs), len(configs)), np.nan)
    shared_n = np.zeros_like(J)
    for i, a in enumerate(configs):
        for j, b in enumerate(configs):
            shared = elig[a] & elig[b]
            ea, eb = egr[a] & shared, egr[b] & shared
            shared_n[i, j] = len(shared)
            if ea | eb:
                J[i, j] = len(ea & eb) / len(ea | eb)
    k_counts = d[d["egregious"]].groupby("key")["ref_config"].nunique().value_counts().sort_index()
    return configs, J, shared_n, k_counts


LAW_STATUS = {"green": "lawful", None: "no signal", "red": "unlawful"}


def passenger_sacrifice_table(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """Per config x pedestrian law status: share of reference choices that kill the car's passengers.

    Uses both scenarios of each pair (original with the original reference answer, counterfactual with
    the counterfactual one), de-duplicated on (config, scenario text). Only in-car scenarios count.
    """
    recs = pd.concat([
        all_pairs[["ref_config", "original_description", "original_reference_answer"]]
        .set_axis(["ref_config", "text", "answer"], axis=1),
        all_pairs[["ref_config", "counterfactual_description", "counterfactual_reference_answer"]]
        .set_axis(["ref_config", "text", "answer"], axis=1),
    ]).drop_duplicates(["ref_config", "text"])
    out = []
    for r in recs.itertuples(index=False):
        try:
            groups = parse_scenario(r.text)
        except ValueError:
            continue
        if "car" not in groups:
            continue
        ped = next(l for l in groups if l != "car")
        killed = killed_location(r.text, r.answer)
        if killed is None:
            continue
        out.append({"ref_config": r.ref_config, "law_status": LAW_STATUS[groups[ped]["law"]],
                    "sacrifice_passengers": killed == "car"})
    d = pd.DataFrame(out, columns=["ref_config", "law_status", "sacrifice_passengers"])
    rows = []
    for (c, s), g in d.groupby(["ref_config", "law_status"]):
        est, lo, hi = boot(g["sacrifice_passengers"].astype(float))
        rows.append({"ref_config": c, "law_status": s, "n": len(g),
                     "k": int(g["sacrifice_passengers"].sum()), "share": est, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Plots: each returns a Figure
# ---------------------------------------------------------------------------------------------
def _hbar_pairs(ax, configs, values, colors, labels, light_first=True):
    """Horizontal grouped bars: for each config two bars (values[0], values[1]); values = [(est,lo,hi,n)...]."""
    y = np.arange(len(configs))
    h = 0.38
    for j, (vals, lab) in enumerate(zip(values, labels)):
        for i, c in enumerate(configs):
            est, lo, hi, n = vals[i]
            if np.isnan(est):
                continue
            off = (j - 0.5) * h
            col = colors[c]
            ax.barh(y[i] + off, est, h, xerr=err(est, lo, hi), color=col,
                    alpha=0.45 if (j == 0 and light_first) else 1.0,
                    hatch="//" if (j == 0 and light_first) else None, edgecolor="white", capsize=2,
                    error_kw={"lw": 0.8}, label=lab if i == 0 else None)
            ax.text(hi + 0.005 * (ax.get_xlim()[1] or 1), y[i] + off, f"n={n}", va="center", fontsize=6)
    ax.set_yticks(y, [short(c) for c in configs], fontsize=7)
    ax.invert_yaxis()


def plot_counts(ctx):
    ap, cfgs, col = ctx["all"], ctx["configs"], ctx["colors"]
    helped = helped_mask(ap, ctx["summary"].get("args", {}))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(cfgs) + 1.8), sharey=True)
    for ax, mask, label in ((axes[0], ap["egregious"], "Egregious: wrong WITH, right WITHOUT"),
                            (axes[1], helped, "Helped: right WITH, wrong WITHOUT")):
        y = np.arange(len(cfgs))
        for i, c in enumerate(cfgs):
            m = mask[ap["ref_config"] == c].astype(float)
            est, lo, hi = boot(m, stat="sum")
            ax.barh(i, est, xerr=err(est, lo, hi), color=col[c], capsize=2, error_kw={"lw": 0.8})
            ax.text(hi + 0.5, i, f"{int(est)} of {len(m)}", va="center", fontsize=7)
        ax.set_yticks(y, [short(c) for c in cfgs], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("pairs (95% bootstrap CI)")
        ax.margins(x=0.2)
    axes[0].invert_yaxis()
    return fig


def plot_edit_shares(ctx):
    ap, cfgs, col = ctx["all"], ctx["configs"], ctx["colors"]
    shares = edit_shares(ap)  # asserts rows sum to 1
    ets = list(shares.columns)
    fig, axes = grid(len(cfgs))
    x = np.arange(len(ets))
    for ax, c in zip(axes, cfgs):
        for j, (setname, d) in enumerate((("all", ap[ap.ref_config == c]),
                                          ("egregious", ap[(ap.ref_config == c) & ap.egregious]))):
            for k, et in enumerate(ets):
                ind = (d["edit_type"] == et).astype(float)
                est, lo, hi = boot(ind)
                if np.isnan(est):
                    continue
                xpos = x[k] + (j - 0.5) * 0.4
                ax.bar(xpos, est, 0.4, yerr=err(est, lo, hi), color=col[c], alpha=0.4 if j == 0 else 1,
                       hatch="//" if j == 0 else None, edgecolor="white", capsize=2, error_kw={"lw": 0.7},
                       label=("all pairs" if j == 0 else "egregious") if k == 0 else None)
                ax.text(xpos, -0.04, f"{int(ind.sum())}", ha="center", va="top", fontsize=6)
        ax.set_title(short(c, 40), fontsize=8)
        ax.set_xticks(x, [EDIT_SHORT.get(e, e) for e in ets], fontsize=6)
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of pairs")
    axes[0].legend(fontsize=7, loc="upper right")
    return fig


def plot_rates(ctx):
    rates = egregious_rate_by_edit(ctx["all"])
    cfgs, col = ctx["configs"], ctx["colors"]
    sel = rates[rates.ref_config.isin(cfgs)]
    big = sel[sel.n >= 10]
    ymax = max(0.02, 1.25 * float((big if len(big) else sel)["hi"].max()))
    fig, axes = grid(len(cfgs))
    ets = [e for e in EDIT_TYPES if e in set(rates.edit_type)] + sorted(set(rates.edit_type) - set(EDIT_TYPES))
    x = np.arange(len(ets))
    for ax, c in zip(axes, cfgs):
        r = rates[rates.ref_config == c].set_index("edit_type").reindex(ets)
        overall = ctx["all"].loc[ctx["all"].ref_config == c, "egregious"].mean()
        for k, et in enumerate(ets):
            if np.isnan(r.loc[et, "rate"]):
                continue
            ax.bar(x[k], r.loc[et, "rate"], 0.6, yerr=err(*r.loc[et, ["rate", "lo", "hi"]]),
                   color=col[c], capsize=2, error_kw={"lw": 0.7})
            clipped = r.loc[et, "hi"] > ymax
            ax.text(x[k], min(r.loc[et, "hi"], ymax * 0.9),
                    f"{int(r.loc[et, 'k'])}/{int(r.loc[et, 'n'])}" + (" ↑" if clipped else ""),
                    ha="center", va="bottom", fontsize=6, color="#b54708" if clipped else "k")
        ax.axhline(overall, ls="--", lw=0.8, color="k")
        ax.set_ylim(0, ymax)
        ax.set_title(short(c, 40), fontsize=8)
        ax.set_xticks(x, [EDIT_SHORT.get(e, e) for e in ets], fontsize=6)
    axes[0].set_ylabel("P(egregious | edit type)")
    return fig


def plot_accuracy(ctx):
    acc = accuracy_by_group(ctx["all"])
    cfgs, col = ctx["configs"], ctx["colors"]
    fig, axes = grid(len(cfgs), panel_w=5.2, panel_h=3.4, sharey=False)
    for ax, c in zip(axes, cfgs):
        a = acc[acc.ref_config == c].copy()
        a["order"] = a.edit_type.map({e: i for i, e in enumerate(EDIT_TYPES)})
        a = a.sort_values(["order", "answer_changed"]).reset_index(drop=True)
        x = np.arange(len(a))
        # with = solid model colour; without = white with hatched model-colour outline (grey would clash
        # with the grey model colours in the palette)
        for off, colname, solid, lab in ((-0.2, "acc_with", True, "with explanation (solid)"),
                                         (0.2, "acc_without", False, "without (hatched)")):
            for i, r in a.iterrows():
                ax.bar(x[i] + off, r[colname], 0.4, yerr=err(r[colname], r[colname + "_lo"], r[colname + "_hi"]),
                       facecolor=col[c] if solid else "white", edgecolor=col[c] if not solid else "none",
                       hatch=None if solid else "////", linewidth=0.8, capsize=1.5, error_kw={"lw": 0.6},
                       label=lab if i == 0 else None)
        ax.set_ylim(0, 1.05)
        ax2 = ax.twinx()
        ax2.errorbar(x, a["gain"], yerr=np.vstack([(a.gain - a.gain_lo).clip(0), (a.gain_hi - a.gain).clip(0)]),
                     fmt="o", color="k", ms=3, lw=0.8, capsize=1.5, label="gain (with − without)")
        ax2.axhline(0, ls="--", lw=0.8, color="k")
        ax2.set_ylim(-1, 1)
        ax2.tick_params(labelsize=6)
        ax.set_xticks(x, [f"{EDIT_SHORT.get(r.edit_type, r.edit_type).replace(chr(10), ' ')}\n"
                          f"{'changed' if r.answer_changed else 'kept'} n={r.n}" for r in a.itertuples()],
                      fontsize=5.5, rotation=60, ha="right")
        ax.set_title(short(c, 44), fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
    axes[0].set_ylabel("simulator accuracy (left) / gain (right)", fontsize=7)
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([], [], marker="o", color="k", ls="none")], l1 + ["gain (right axis)"],
               loc="lower center", ncol=3, fontsize=8)
    return fig


def plot_answer_changed(ctx):
    ap, cfgs, col = ctx["all"], ctx["configs"], ctx["colors"]
    vals = []
    for d in (ap, ap[ap.egregious]):
        row = []
        for c in cfgs:
            m = d.loc[d.ref_config == c, "answer_changed"].astype(float)
            row.append((*boot(m), len(m)))
        vals.append(row)
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(cfgs) + 1.6))
    ax.set_xlim(0, 1.15)
    _hbar_pairs(ax, cfgs, vals, col, ["all pairs", "egregious"])
    ax.set_xlabel("share of pairs where the reference answer changed (95% CI)")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def _ensure_matched(ctx):
    if not ctx.get("_matched_ok"):
        assert_matched(ctx["egr"], ctx["comp"], ctx["summary"])
        ctx["_matched_ok"] = True


def plot_egr_vs_comp(ctx):
    _ensure_matched(ctx)
    egr, comp = ctx["egr"], ctx["comp"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    sets = (("egregious", egr, "#c0392b"), ("matched comparison", comp, "#2f6db5"))
    # confidence
    levels = ["HIGH", "MEDIUM", "LOW", "OTHER", "MISSING"]
    present = [l for l in levels if any((d["confidence"].map(conf_level) == l).any() for _, d, _ in sets)]
    x = np.arange(len(present))
    for j, (name, d, colour) in enumerate(sets):
        lv = d["confidence"].map(conf_level)
        for k, l in enumerate(present):
            est, lo, hi = boot((lv == l).astype(float))
            axes[0].bar(x[k] + (j - 0.5) * 0.4, est, 0.4, yerr=err(est, lo, hi), color=colour, capsize=2,
                        label=name if k == 0 else None)
            axes[0].text(x[k] + (j - 0.5) * 0.4, hi + 0.01, f"{int((lv == l).sum())}", ha="center", fontsize=7)
    axes[0].set_xticks(x, present)
    axes[0].set_ylabel("share of pairs")
    axes[0].set_title("Reference confidence", fontsize=10)
    axes[0].legend(fontsize=8)
    # answer position
    for j, (name, d, colour) in enumerate(sets):
        af = to_bool(d["answer_first"]).dropna().astype(float)
        for k, (lab, v) in enumerate((("answer first", af), ("answer last", 1 - af))):
            est, lo, hi = boot(v)
            axes[1].bar(k + (j - 0.5) * 0.4, est, 0.4, yerr=err(est, lo, hi), color=colour, capsize=2)
            axes[1].text(k + (j - 0.5) * 0.4, hi + 0.01, f"{int(v.sum())}", ha="center", fontsize=7)
    axes[1].set_xticks([0, 1], ["answer first", "answer last"])
    axes[1].set_title("Answer position in the reference prompt", fontsize=10)
    # explanation length
    lens = [d["explanation"].dropna().astype(str).str.split().str.len() for _, d, _ in sets]
    bp = axes[2].boxplot(lens, patch_artist=True, widths=0.5)
    for patch, (_, _, colour) in zip(bp["boxes"], sets):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    axes[2].set_xticks([1, 2], [f"{n}\n(n={len(l)})" for (n, _, _), l in zip(sets, lens)])
    axes[2].set_ylabel("words")
    axes[2].set_title("Explanation length", fontsize=10)
    return fig


def plot_passenger_keyword(ctx):
    _ensure_matched(ctx)
    egr, comp, col = ctx["egr"], ctx["comp"], ctx["colors"]
    cfgs = [c for c in ctx["configs"] if (egr.ref_config == c).any()]
    vals = []
    for d in (comp, egr):
        row = []
        for c in cfgs:
            m = d.loc[d.ref_config == c, "explanation"].fillna("").astype(str).map(
                lambda s: bool(PASSENGER_RE.search(s))).astype(float)
            row.append((*boot(m), len(m)))
        vals.append(row)
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(cfgs) + 1.6))
    ax.set_xlim(0, 1.15)
    _hbar_pairs(ax, cfgs, vals, col, ["matched comparison", "egregious"])
    ax.set_xlabel("share of reference explanations matching the passenger keywords (95% CI)")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_overlap(ctx):
    configs, J, shared, k_counts = jaccard_matrix(ctx["all"])
    n = len(configs)
    side = max(6, 0.36 * n + 3)
    fig, (ax, tab_ax) = plt.subplots(1, 2, figsize=(side + 3.5, side),
                                     gridspec_kw={"width_ratios": [side, 3]})
    im = ax.imshow(np.ma.masked_invalid(J), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), [short(c, 44) for c in configs], rotation=90, fontsize=6)
    ax.set_yticks(range(n), [short(c, 44) for c in configs], fontsize=6)
    for i in range(n):
        for j in range(n):
            if not np.isnan(J[i, j]) and i != j:
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if J[i, j] < 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Jaccard over pairs eligible in both\n(blank = neither has egregious pairs there)", fontsize=8)
    tab_ax.axis("off")
    cells = [[str(k), str(v)] for k, v in k_counts.items()] or [["–", "0"]]
    t = tab_ax.table(cellText=cells, colLabels=["# models", "# pairs"], loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    tab_ax.set_title("Pairs egregious for\nk models", fontsize=9)
    return fig


def plot_passenger_law(ctx):
    tab = passenger_sacrifice_table(ctx["all"])
    cfgs, col = [c for c in ctx["configs"] if c in set(tab.ref_config)], ctx["colors"]
    order = ["lawful", "no signal", "unlawful"]
    fig, axes = grid(len(cfgs), panel_w=3.4, panel_h=2.8)
    for ax, c in zip(axes, cfgs):
        t = tab[tab.ref_config == c].set_index("law_status").reindex(order)
        for k, s in enumerate(order):
            if np.isnan(t.loc[s, "share"]):
                continue
            ax.bar(k, t.loc[s, "share"], 0.6, yerr=err(*t.loc[s, ["share", "lo", "hi"]]), color=col[c],
                   alpha=[1, 0.65, 0.35][k], capsize=2, error_kw={"lw": 0.7})
            ax.text(k, t.loc[s, "hi"] + 0.02, f"{int(t.loc[s, 'k'])}/{int(t.loc[s, 'n'])}", ha="center",
                    fontsize=6)
        ax.set_xticks(range(3), ["pedestrians\nlawful", "no\nsignal", "pedestrians\nunlawful"], fontsize=6)
        ax.set_ylim(0, 1.1)
        ax.set_title(short(c, 36), fontsize=8)
    if axes:
        axes[0].set_ylabel("share choosing to kill passengers")
    return fig


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------
BASE = ["ref_config", "egregious"]
PLOTS = [
    dict(key="01_egregious_counts", fn=plot_counts,
         title="How many pairs does each model's explanation mislead, and how many does it rescue?",
         needs={"all": BASE + ["n_simulators", "n_wrong_with", "n_right_without"]},
         caption="Left: number of egregious pairs per model (all simulators wrong with the explanation, a "
                 "majority right without it). Right: the mirror image, pairs the explanation rescued; "
                 "bars show 95% bootstrap CIs over pairs and the label gives count of eligible pairs."),
    dict(key="02_edit_type_shares", fn=plot_edit_shares,
         title="Which edits cause explanation failures?",
         needs={"all": BASE + ["edit_type"]},
         caption="For each model, the share of each edit type among all pairs (hatched) and among egregious "
                 "pairs (solid); an edit type that is taller when solid is over-represented in failures. "
                 "Numbers under the bars are pair counts."),
    dict(key="03_egregious_rate_by_edit", fn=plot_rates,
         title="How over-represented is each edit type among failures?",
         needs={"all": BASE + ["edit_type"]},
         caption="P(egregious | edit type) per model with 95% bootstrap CIs; labels are egregious/total pairs "
                 "and the dashed line is the model's overall egregious rate. The y-axis is shared and set by cells "
                 "with at least 10 pairs; an orange ↑ marks a small cell whose bar or CI runs off the top."),
    dict(key="04_accuracy_with_without", fn=plot_accuracy,
         title="Where do explanations help or hurt the simulators?",
         needs={"all": BASE + ["edit_type", "answer_changed", "n_simulators", "n_wrong_with", "n_right_without"]},
         caption="Mean simulator accuracy on the counterfactual with the explanation (model colour) and "
                 "without it (white, hatched), split by edit type and whether the reference answer changed or was kept. "
                 "Black dots (right axis) are the gain, with minus without; below the dashed zero line the "
                 "explanation hurt."),
    dict(key="05_answer_changed", fn=plot_answer_changed,
         title="Do failures happen when the reference answer changes?",
         needs={"all": BASE + ["answer_changed"]},
         caption="Share of pairs in which the reference model changed its answer between the original and the "
                 "counterfactual, among all pairs (hatched) and among egregious pairs (solid), per model."),
    dict(key="06_egregious_vs_comparison", fn=plot_egr_vs_comp,
         title="Do egregious cases differ from matched controls in confidence, answer position or length?",
         needs={"egr": ["ref_config", "edit_type", "answer_changed", "confidence", "answer_first", "explanation"],
                "comp": ["ref_config", "edit_type", "answer_changed", "confidence", "answer_first",
                         "explanation"]},
         caption="Egregious pairs (red) against the comparison set matched on model, edit type and answer "
                 "changed (blue), pooled over the selected models: stated confidence, whether the prompt asked "
                 "for the answer before the explanation, and explanation length in words."),
    dict(key="07_passenger_keyword", fn=plot_passenger_keyword,
         title="Do failing explanations lean on the passenger argument? "
               "(rough keyword check, to be replaced by the LLM reader)",
         needs={"egr": ["ref_config", "edit_type", "answer_changed", "explanation"],
                "comp": ["ref_config", "edit_type", "answer_changed", "explanation"]},
         caption="Share of reference explanations matching passeng|occupant|inside the car|trusted|accepted "
                 "(the )?risk, egregious versus matched comparison, per model with any egregious pairs. "
                 "This is a crude keyword match, not a reading of the argument."),
    dict(key="08_overlap_across_models", fn=plot_overlap,
         title="Are the same pairs egregious across models?",
         needs={"all": BASE + ["original_question_idx", "counterfactual_question_idx"]},
         caption="Jaccard overlap of egregious pairs between models, computed only over pairs both models "
                 "were scored on (the data has two mostly separate pair sets); the table counts pairs that "
                 "are egregious for k models. The two copies of a duplicated medium config measure "
                 "simulator run-to-run noise rather than model agreement."),
    dict(key="09_passenger_vs_law", fn=plot_passenger_law,
         title="Does pedestrian law status override passenger protection?",
         needs={"all": ["ref_config", "original_description", "counterfactual_description",
                        "original_reference_answer", "counterfactual_reference_answer"]},
         caption="In scenarios with passengers, the share of the reference model's choices (original and "
                 "counterfactual scenarios, de-duplicated) that kill the passengers, by whether the pedestrians "
                 "cross lawfully, with no signal, or unlawfully. Uses the reconstructed scenario parser in "
                 "filter_egregious.py, not the one from feat.py."),
]


def missing_columns(spec, frames) -> list:
    return [f"{src}.{c}" for src, cols in spec["needs"].items() for c in cols if c not in frames[src].columns]


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------
HTML_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Egregious Cases Plots</title><style>
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#5d5d5d;--card:#fff;--line:#e2e2e2;--accent:#2f6db5}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ececec;--muted:#a8a8a8;--card:#202020;
--line:#333;--accent:#7fb0ef}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{padding:20px 16px 8px;max-width:1200px;margin:auto}h1{font-size:22px;margin:0 0 4px}
.meta{color:var(--muted);font-size:13px}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:2}
nav div{max-width:1200px;margin:auto;padding:8px 16px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
nav a{color:var(--accent);text-decoration:none;font-size:13px;padding:2px 8px;border:1px solid var(--line);
border-radius:12px}nav button{font-size:12px;margin-left:auto}
main{max-width:1200px;margin:auto;padding:12px 16px 48px}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:14px 0;padding:0 14px}
summary{cursor:pointer;padding:12px 0;font-weight:600}details p{color:var(--muted);margin:0 0 10px}
img{max-width:100%;height:auto;cursor:zoom-in;background:#fff;border-radius:4px}
.skip{color:#b54708}#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;overflow:auto;z-index:9}
#lb img{max-width:none;cursor:zoom-out;margin:16px}
</style></head><body>"""

HTML_TAIL = """<div id="lb" onclick="this.style.display='none'"><img id="lbimg" alt=""></div><script>
document.querySelectorAll('main img').forEach(i=>i.onclick=()=>{lbimg.src=i.src;lb.style.display='block'});
function setAll(o){document.querySelectorAll('details').forEach(d=>d.open=o)}
</script></body></html>"""


def write_html(path, sections, meta):
    nav = "".join(f'<a href="#{s["key"]}">{s["key"][:2]}</a>' for s in sections)
    body = []
    for s in sections:
        inner = (f'<img src="{s["png"]}" alt="{html.escape(s["title"])}" loading="lazy">' if s.get("png")
                 else f'<p class="skip">Skipped: {html.escape(s["skipped"])}</p>')
        body.append(f'<details open id="{s["key"]}"><summary>{s["key"][:2]}. {html.escape(s["title"])}'
                    f'</summary><p>{html.escape(s["caption"])}</p>{inner}</details>')
    path.write_text(HTML_HEAD + f'<header><h1>Moral Machines egregious cases</h1><div class="meta">'
                    f'{html.escape(meta)}</div></header><nav><div>{nav}<button onclick="setAll(true)">'
                    f'Expand all</button><button onclick="setAll(false)">Collapse all</button></div></nav>'
                    f'<main>{"".join(body)}</main>' + HTML_TAIL)


def load_inputs(in_dir: Path):
    all_pairs = pd.read_csv(in_dir / "all_pairs.csv")
    egr = pd.read_csv(in_dir / "egregious_candidates.csv")
    comp = pd.read_csv(in_dir / "comparison_set.csv")
    summary = json.loads((in_dir / "summary.json").read_text())
    for d in (all_pairs, egr, comp):
        for c in ("egregious", "answer_changed"):
            if c in d.columns:
                d[c] = to_bool(d[c]).fillna(False).astype(bool)
    return all_pairs, egr, comp, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="moral_machines/output")
    ap.add_argument("--out-dir", default="moral_machines/output/plots")
    ap.add_argument("--thinking", default=None, help="comma-separated thinking labels to keep (default: all)")
    ap.add_argument("--configs", default=None, help="regex on ref_config to keep (default: all)")
    a = ap.parse_args(argv)

    all_pairs, egr, comp, summary = load_inputs(Path(a.in_dir))
    if {"ref_config", "egregious"} <= set(all_pairs.columns) and "ref_config" in egr.columns:
        check_counts(all_pairs, egr, summary)  # plot 1 check, on the unfiltered data
    colors = config_colors(all_pairs["ref_config"].unique()) if "ref_config" in all_pairs else {}

    def keep(d):
        m = pd.Series(True, index=d.index)
        if a.thinking and "thinking" in d:
            m &= d["thinking"].astype(str).isin([t.strip() for t in a.thinking.split(",")])
        if a.configs and "ref_config" in d:
            rx = re.compile(a.configs)
            m &= d["ref_config"].astype(str).map(lambda c: bool(rx.search(c)))
        return d[m]

    all_pairs, egr, comp = keep(all_pairs), keep(egr), keep(comp)
    configs = sorted(all_pairs["ref_config"].unique()) if "ref_config" in all_pairs else []
    if not configs:
        raise SystemExit("no pairs left after filtering")
    ctx = {"all": all_pairs, "egr": egr, "comp": comp, "summary": summary, "configs": configs, "colors": colors}
    frames = {"all": all_pairs, "egr": egr, "comp": comp}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    with PdfPages(out / "all_plots.pdf") as pdf:
        for spec in PLOTS:
            sec = {"key": spec["key"], "title": spec["title"], "caption": spec["caption"]}
            miss = missing_columns(spec, frames)
            if miss:
                msg = f"missing column(s) {', '.join(miss)}"
                warnings.warn(f"skipping {spec['key']}: {msg}")
                sec["skipped"] = msg
                sections.append(sec)
                continue
            fig = spec["fn"](ctx)
            fig.suptitle(spec["title"], fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0.03, 1, 0.96))
            png = f"{spec['key']}.png"
            fig.savefig(out / png, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            sec["png"] = png
            sections.append(sec)
            print(f"[plot] {png}")

    md = ["# Moral Machines egregious cases: plot captions\n"]
    for s in sections:
        md.append(f"## {s['key'][:2]}. {s['title']}\n\n{s['caption']}\n")
        md.append(f"![{s['key']}]({s['png']})\n" if s.get("png") else f"_Skipped: {s['skipped']}._\n")
    (out / "captions.md").write_text("\n".join(md))
    meta = (f"{len(configs)} reference configs · {len(all_pairs)} pairs · {len(egr)} egregious · "
            f"{len(comp)} comparison · thinking={a.thinking or 'all'} · configs={a.configs or 'all'}")
    write_html(out / "index.html", sections, meta)
    print(f"Wrote {out/'all_plots.pdf'}, {out/'captions.md'}, {out/'index.html'}")
    return sections


if __name__ == "__main__":
    main()
