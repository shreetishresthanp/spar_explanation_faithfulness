# %% [markdown]
# # Case-order swap: is Gemma 3 27B's answer-last CASE 1 preference about position or about content?
#
# Run from the repo root in the project env, after paired_format_test.py --run (its calls are the unswapped arm):
#   conda activate faithfulness-env
#   python analysis/gemma_stress_test/case_order_swap_test.py --smoke     # 4 calls: prompt check, provider, cost
#   python analysis/gemma_stress_test/case_order_swap_test.py --run       # all swapped calls (resumes), then analysis
#   python analysis/gemma_stress_test/case_order_swap_test.py --analyze   # analysis only
# Paid calls go through OpenRouter (OPENROUTER_API_KEY in .env).
#
# Design: 2 x 2 within scenario, answer position (first / last) x case order (original / swapped), SAMPLES samples
# per cell. Scenarios: N_PAIRS original scenarios plus each one's counterfactual (2 x N_PAIRS scenarios,
# 2 x N_PAIRS x 2 formats x SAMPLES swapped calls). Originals are sampled stratified by the confounding test's 8 cells
# (Case 1 kills passengers x Case 1 swerves x legality): an equal share per cell, small cells taken whole and their
# unused share spread over the others (SELECT_SEED; the selection is saved). Overall estimates are weighted back to
# the full 1,000-scenario mix (weight = cell size / cell sample) with a bootstrap resampling within cells; per-cell
# estimates are unweighted. The original-order cells are the
# paired test's calls (paired_format_calls.jsonl), so only the swapped order is called here, with identical settings,
# provider, seeds and parser (all imported from paired_format_test.py). Swapping exchanges the two case paragraphs and
# keeps the "Case 1." / "Case 2." headers and the instruction line in place; the prompt is then built with
# MoralMachines.create_reference_prompt, the function that built the stored prompts. So after the swap, the label
# CASE 1 and the first-listed position both point at what was Case 2's content.
#
# Outcomes per reply: label = the CASE it names; content = whether it chose the ORIGINAL Case 1 content (CASE 1 in
# the original order, CASE 2 in the swapped order). Original Case 1 is always the passenger-sacrifice option when
# passengers are involved, so the swap also separates "prefers passenger sacrifice" from "prefers CASE 1".
# Per answer position, paired over scenarios:
#   position bias      = P(content | original order) - P(content | swapped) = P(CASE 1 | orig) + P(CASE 1 | swap) - 1;
#                        > 0 means a pull towards whichever case is listed first / labelled CASE 1;
#   content preference = order-averaged P(chose original Case 1 content).
# Decision rule for answer last, decided before running (original scenarios):
#   P(CASE 1 | swapped, answer last) > 85%  -> position/label bias: answer-last picks whatever is listed first;
#   P(CASE 1 | swapped, answer last) < 15%  -> content preference: it follows the original Case 1 content;
#   otherwise                               -> mixed; read the position-bias and content-preference estimates.
# Limitation: the label "CASE 1" always sits on the first-listed case, so label and position are not separated.
# Writes experiments/moral_machines/gemma_stress_test/case_swap_{calls.jsonl,selection.csv,summary_<parsing>.csv} and
# figures/moral_machines/gemma_stress_test/case_swap_test_<parsing>.png.

# %%
import argparse
import asyncio
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_format_test as pft  # noqa: E402  (sets the working directory to the repo root)
from src.templates.moral_machines import MoralMachines  # noqa: E402

SWAP_CALLS = pft.OUT / "case_swap_calls.jsonl"
SELECTION = pft.OUT / "case_swap_selection.csv"
N_PAIRS, SELECT_SEED = 500, 0
ORDERS = ["original", "swapped"]
DECIDE_HI, DECIDE_LO = 0.85, 0.15
CASES = re.compile(r"^(?P<stem>.*?\n\nCase 1\.\n)(?P<a>.*?)(?P<mid>\n\nCase 2\.\n)(?P<b>.*?)(?P<end>\n*)$", re.S)
ORDER_COLORS = {"original": "#2a78d6", "swapped": "#eb6834"}
INK, MUTED = pft.INK, pft.MUTED


# %% swapped prompts
def swap_cases(question):
    """Exchange the two case paragraphs; headers, stem and whitespace stay where they are."""
    m = CASES.match(question)
    if m is None:
        raise ValueError(f"cannot find the two cases in: {question[:120]!r}")
    return m["stem"] + m["b"] + m["mid"] + m["a"] + m["end"]


def cell(question):
    """Scenario cell of the confounding test, from Case 1's text: kills passengers | swerves | legality."""
    case1 = CASES.match(question)["a"]
    killed = re.search(r"death of .+?, who (?:were|was) (.+?)\.", case1).group(1)
    legal = 1 if "abiding by the law" in killed else -1 if "flouting the law" in killed else 0
    return f"{'inside the car' in killed}|{'will swerve' in case1}|{legal}"


def allocate(sizes, total):
    """Equal share per cell, capped at the cell's size; what small cells cannot use goes to the others."""
    alloc, left, open_cells = {}, total, dict(sizes)
    while open_cells:
        share = left // len(open_cells)
        small = {k: n for k, n in open_cells.items() if n <= share}
        if not small:
            for i, k in enumerate(sorted(open_cells, key=open_cells.get, reverse=True)):
                alloc[k] = share + (i < left - share * len(open_cells))
            break
        for k, n in small.items():
            alloc[k], left = n, left - n
            del open_cells[k]
    return alloc


def select_pairs(d):
    """N_PAIRS original scenarios, stratified by cell, plus each one's counterfactual (same parquet row)."""
    cols = ["original_dataset", "original_reference_response_model_info_model", "original_question_idx",
            "counterfactual_question_idx"]
    t = pq.read_table(pft.DATA, columns=cols, filters=[("original_dataset", "=", "moral_machines")]).to_pandas()
    pairs = t[t[cols[1]] == pft.TARGET].drop_duplicates(cols[2])[cols[2:]]
    pairs.columns = ["orig_idx", "cf_idx"]
    orig = d[d["set"] == "original"].set_index("idx")
    pairs["cell"] = orig.loc[pairs["orig_idx"], "cell"].values
    sizes = pairs["cell"].value_counts().to_dict()
    alloc = allocate(sizes, N_PAIRS)
    chosen = pd.concat([g.sample(alloc[c], random_state=SELECT_SEED) for c, g in pairs.groupby("cell")])
    chosen["weight"] = chosen["cell"].map(lambda c: sizes[c] / alloc[c])  # back to the full scenario mix
    chosen.to_csv(SELECTION, index=False)
    print("selected pairs per cell (of available):",
          {c: f"{alloc[c]}/{sizes[c]}" for c in sorted(sizes)}, f"-> {SELECTION}")
    return chosen


def load_swapped():
    d = pft.load_scenarios()
    d["cell"] = d["question"].map(cell)
    chosen = select_pairs(d)
    keep = (set(zip(["original"] * len(chosen), chosen["orig_idx"]))
            | set(zip(["counterfactual"] * len(chosen), chosen["cf_idx"])))
    d = d[[k in keep for k in zip(d["set"], d["idx"])]].reset_index(drop=True)
    w = dict(zip(chosen["orig_idx"], chosen["weight"])) | dict(zip(chosen["cf_idx"], chosen["weight"]))
    d["weight"] = d["idx"].map(w)
    d["swapped_question"] = d["question"].map(swap_cases)
    assert (d["swapped_question"].map(swap_cases) == d["question"]).all(), "swap is not its own inverse"
    assert (d["swapped_question"] != d["question"]).all(), "some swaps changed nothing"
    for f in pft.FORMATS:
        d[f"swapped_prompt_{f}"] = [MoralMachines.create_reference_prompt(q, answer_last=(f == "last"))
                                    for q in d["swapped_question"]]
    print(f"{len(d)} scenarios selected ({d.groupby('set').size().to_dict()}); swapped prompts built; "
          "swap checked to be its own inverse")
    return d


def make_jobs(d, samples):
    have = set(pft.load_calls(SWAP_CALLS).get("key", pd.Series(dtype=str)))
    jobs = []
    for r in d.itertuples():
        for f in pft.FORMATS:
            for sample in samples:
                key = "swapped|" + pft.call_key(r.set, r.idx, f, sample)
                if key not in have:
                    jobs.append({"key": key, "set": r.set, "idx": int(r.idx), "stored_format": r.stored_format,
                                 "format": f, "sample": sample, "order": "swapped", "prompt_source": "swap-template",
                                 "prompt": getattr(r, f"swapped_prompt_{f}")})
    return jobs


def smoke(d):
    pick = pd.concat([d[(d["set"] == "original") & (d["stored_format"] == f)].head(1) for f in pft.FORMATS])
    print("\nexample swapped prompt (answer last):\n", pick.iloc[0]["swapped_prompt_last"])
    jobs = make_jobs(pick, [0])
    asyncio.run(pft.run_calls(jobs, SWAP_CALLS))
    c = pft.load_calls(SWAP_CALLS)
    c = c[c["key"].isin([j["key"] for j in jobs])]
    print(c[["key", "provider", "answer", "finish_reason", "cost"]].to_string(index=False))
    total = len(d) * len(pft.FORMATS) * pft.SAMPLES
    if c["cost"].notna().any():
        print(f"\nmean cost per call ${c['cost'].mean():.6f} -> projected ${c['cost'].mean() * total:.2f} "
              f"for {total} swapped calls")


# %% analysis
METRICS = [("P(CASE 1), original order", "label1_original"), ("P(CASE 1), swapped order", "label1_swapped"),
           ("position bias", "position_bias"), ("content preference (original Case 1)", "content_pref")]


def weighted_boot(df, col, rng):
    """Weighted mean (back to the full scenario mix) with a bootstrap that resamples scenarios within each cell."""
    vals, w = df[col].values, df["weight"].values
    groups = [np.flatnonzero(df["cell"].values == k) for k in df["cell"].unique()]
    reps = [np.average(vals[i], weights=w[i]) for i in
            (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(pft.N_BOOT))]
    return np.average(vals, weights=w), *np.percentile(reps, [2.5, 97.5])


def load_both(parsing):
    """All calls from both orders, with the label answer and whether the original Case 1 content was chosen."""
    c = pd.concat([pft.load_calls().assign(order="original"), pft.load_calls(SWAP_CALLS).assign(order="swapped")],
                  ignore_index=True)
    c["answer_strict"] = c["answer"]
    if parsing == "lenient":
        c["answer"] = [pft.lenient(r, a) for r, a in zip(c["raw"], c["answer_strict"])]
    valid = c["answer"].notna()
    c["label1"] = np.where(valid, c["answer"].eq("CASE 1"), np.nan)
    c["content1"] = np.where(valid, c["answer"].eq(np.where(c["order"] == "original", "CASE 1", "CASE 2")), np.nan)
    return c


def analyze(d, parsing="lenient"):
    c = load_both(parsing)
    if (c["order"] == "swapped").sum() == 0:
        print("no swapped calls yet")
        return
    print(f"\ncalls: {c.groupby('order').size().to_dict()}; providers {c['provider'].value_counts().to_dict()}")
    print(f"{parsing} parsing; parse rate (strict = paper's parser):")
    print(c.groupby(["order", "format"]).agg(strict=("answer_strict", lambda a: a.notna().mean()),
                                             used=("answer", lambda a: a.notna().mean())).round(3).to_string())

    rng = np.random.default_rng(pft.SEED)
    per = (c.groupby(["set", "idx", "format", "order"])[["label1", "content1"]].mean().unstack("order")
           .dropna().reset_index())
    per.columns = ["_".join(x).strip("_") for x in per.columns]
    per = per.merge(d[["set", "idx", "cell", "weight"]], on=["set", "idx"])  # selected scenarios only
    per["position_bias"] = per["content1_original"] - per["content1_swapped"]
    per["content_pref"] = (per["content1_original"] + per["content1_swapped"]) / 2
    rows = []
    for s in pft.SETS + ["both"]:
        ps = per if s == "both" else per[per["set"] == s]
        for scen in ["all"] + sorted(ps["cell"].unique()):
            pc = ps if scen == "all" else ps[ps["cell"] == scen]
            for f in pft.FORMATS:
                pf = pc[pc["format"] == f]
                if pf.empty:
                    continue
                for metric, col in METRICS:
                    est, lo, hi = (weighted_boot(pf, col, rng) if scen == "all"
                                   else pft.boot_mean(pf[col].values, rng))
                    rows.append({"set": s, "scenarios": scen, "format": f, "metric": metric, "value": est, "lo": lo,
                                 "hi": hi, "n": len(pf)})
    summary = pd.DataFrame(rows)
    summary.to_csv(pft.OUT / f"case_swap_summary_{parsing}.csv", index=False)
    show = summary[summary["scenarios"] == "all"].pivot_table(index=["set", "metric"], columns="format",
                                                              values="value", sort=False)
    print("\nall scenarios (weighted to the full scenario mix), by answer position:\n", show.round(3).to_string())
    print("\nby cell, original scenarios (cell = Case 1 kills passengers | Case 1 swerves | legality, "
          "1 = kills law-abiding, -1 = kills lawbreakers, 0 = no light):\n",
          summary[(summary["set"] == "original") & (summary["scenarios"] != "all")
                  & summary["metric"].isin(["P(CASE 1), swapped order", "position bias"])]
          .pivot_table(index="scenarios", columns=["metric", "format"], values="value", sort=False)
          .round(3).assign(n=summary.groupby("scenarios")["n"].first()).to_string())

    o = summary[(summary["set"] == "original") & (summary["scenarios"] == "all")].set_index(["format", "metric"])
    sw = o.at[("last", "P(CASE 1), swapped order"), "value"]
    verdict = ("position/label bias: answer-last picks whatever is listed first" if sw > DECIDE_HI else
               "content preference: answer-last follows the original Case 1 content" if sw < DECIDE_LO else
               "mixed: read position bias and content preference")
    complete = (per[per["set"] == "original"].groupby(["idx", "format"]).ngroups
                == len(pft.FORMATS) * (d["set"] == "original").sum())
    print(f"\nDECISION (original scenarios{'' if complete else ', swapped arm incomplete'}): "
          f"P(CASE 1 | swapped, answer last) = {sw:.1%} -> {verdict}")
    for f in pft.FORMATS:
        pb, cp = o.loc[(f, "position bias")], o.loc[(f, "content preference (original Case 1)")]
        print(f"  answer {f}: position bias {pb['value'] * 100:+.1f} pts ({pb['lo'] * 100:+.1f} to "
              f"{pb['hi'] * 100:+.1f}); content preference {cp['value']:.1%} ({cp['lo']:.1%} to {cp['hi']:.1%})")
    plot(summary, parsing)


def plot(summary, parsing):
    sets = [s for s in pft.SETS if s in set(summary["set"])]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 4.6), sharey=True, squeeze=False)
    w = 0.38
    for ax, s in zip(axes[0], sets):
        t = summary[(summary["set"] == s) & (summary["scenarios"] == "all")]
        for k, order in enumerate(ORDERS):
            r = t[t["metric"] == f"P(CASE 1), {order} order"].set_index("format").reindex(pft.FORMATS)
            xs = np.arange(len(pft.FORMATS)) + (k - 0.5) * w
            bars = ax.bar(xs, r["value"], w, yerr=[r["value"] - r["lo"], r["hi"] - r["value"]],
                          color=ORDER_COLORS[order], label=f"{order} case order",
                          error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in r["value"]], padding=8, fontsize=9, color=MUTED)
        ax.axhline(0.5, color=INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(pft.FORMATS)), [f"answer {f}" for f in pft.FORMATS])
        ax.set_ylim(0, 1.25)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{s} scenarios")
    axes[0, 0].set_ylabel("P(answer is CASE 1), same scenarios")
    axes[0, 0].legend(loc="upper left", ncol=2)
    fig.suptitle(f"Case-order swap: Gemma 3 27B, Moral Machines ({parsing} parsing)\n"
                 "same CASE 1 rate in both orders = position bias; mirrored rates = content preference",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(pft.FIG / f"case_swap_test_{parsing}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {pft.FIG / f'case_swap_test_{parsing}.png'} and {pft.OUT / f'case_swap_summary_{parsing}.csv'}")


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="4 swapped calls to check prompts, provider and cost")
    mode.add_argument("--run", action="store_true", help="all swapped calls not yet saved, then the analysis")
    mode.add_argument("--analyze", action="store_true", help="analysis only, from saved calls")
    ap.add_argument("--parsing", choices=["lenient", "strict"], default="lenient")
    args = ap.parse_args()
    scenarios = load_swapped()
    if args.smoke:
        smoke(scenarios)
    elif args.run:
        jobs = make_jobs(scenarios, range(pft.SAMPLES))
        print(f"{len(jobs)} calls to make")
        failed = asyncio.run(pft.run_calls(jobs, SWAP_CALLS)) if jobs else 0
        if failed:
            print(f"{failed} calls failed after {pft.MAX_RETRIES} tries and were not saved; rerun --run to retry them")
        analyze(scenarios, args.parsing)
    else:
        analyze(scenarios, args.parsing)
