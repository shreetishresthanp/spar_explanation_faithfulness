"""Build the two Moral Machines analysis tables from the prediction parquets.

pairs_features.parquet  one row per unique (original text, counterfactual text) pair (1,858 on the
                        current data). State columns for each scenario (orig_* / cf_*), change columns
                        (d_* deltas, *_changed flags, filter_egregious edit flags) and the edit_type label,
                        plus raw_match and the pair set(s) the pair occurs in.
answers_long.parquet    one row per parquet row = pair x reference-model config x source file (25,398).
                        Reference answers, simulator answers with / without explanation and their
                        accuracy, joined to pairs_features by pair_key. Rows with missing answers are kept
                        and flagged (usable=False, drop_reason) instead of dropped.

Pair sets: a source file belongs to set "A" when every text in it equals the moral_machines_raw.csv
prompt at its question_idx, otherwise to set "B" (an older generation whose social_value scenarios
differ from the CSV). Currently A = claude_4_5, gemini_3, gpt_5_2_itc, gpt_5, sonnet_itc; B = qwen_3,
gemma_3, extra_three_ref_models.

raw_match is recorded per occurrence (text, question_idx): the text equals the CSV prompt at that index.
answers_long has it per row (orig_raw_match / cf_raw_match); in pairs_features a text counts as matched if
any of its occurrences matches (5 texts occur under two indices: the CSV prompt at one, a set-B row at
the other), and *_csv_idx names the matching CSV row. Analyses that need the generator's own settings
(the "generated" stage) should use only rows where raw_match is True.

Themes use scenario_features.infer_theme, which assumes the paper's generator rules.

Usage (from the repo root):
    python -m moral_machines.build_tables [parquets ...] [--out-dir moral_machines/output]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from moral_machines import filter_egregious as fe
from moral_machines import scenario_features as sf

DEFAULT_PARQUETS = "data_parquet/*.parquet"
DEFAULT_RAW_CSV = "data/raw/moral_machines_raw.csv"
SIDE_FIELDS = ("total", "humans", "animals")


def pair_key(orig_text: str, cf_text: str) -> str:
    """Stable 16-hex id of a (original, counterfactual) text pair."""
    return hashlib.sha1((orig_text + "\x1f" + cf_text).encode()).hexdigest()[:16]


def load_rows(paths: Iterable[str]) -> tuple[pd.DataFrame, dict]:
    """Moral Machines rows of every parquet, with source_file; plus the resolved column map."""
    df = fe.load(sorted(paths))
    cols = fe.resolve_columns(df.columns)
    df = df[df[cols["dataset"]] == "moral_machines"].reset_index(drop=True)
    return df, cols


def occurrence_raw_match(df: pd.DataFrame, cols: dict, raw: Optional[pd.DataFrame]) -> dict[tuple, bool]:
    """(text, question_idx) -> True if the text equals the CSV prompt at that index (False if no CSV)."""
    out = {}
    for tcol, icol in ((cols["orig_desc"], cols["orig_idx"]), (cols["cf_desc"], cols["cf_idx"])):
        for t, i in set(zip(df[tcol], df[icol].astype(int))):
            out[(t, i)] = raw is not None and i in raw.index and raw.at[i, "prompt"] == t
    return out


def text_csv_idx(occ: dict[tuple, bool]) -> dict[str, Optional[int]]:
    """text -> smallest question_idx at which it equals the CSV prompt (None if never)."""
    out: dict[str, Optional[int]] = {}
    for (t, i), ok in sorted(occ.items(), key=lambda kv: kv[0][1]):
        if out.get(t) is None:
            out[t] = i if ok else None
    return out


def file_pair_sets(df: pd.DataFrame, cols: dict, occ: dict[tuple, bool]) -> dict[str, str]:
    """source_file -> 'A' if every (text, question_idx) occurrence in it raw-matches, else 'B'."""
    out = {}
    for f, g in df.groupby("source_file"):
        ok = all(occ[(t, int(i))] for tc, ic in ((cols["orig_desc"], cols["orig_idx"]), (cols["cf_desc"], cols["cf_idx"]))
                 for t, i in zip(g[tc], g[ic]))
        out[f] = "A" if ok else "B"
    return out


def _state(s: sf.Scenario, prefix: str) -> dict:
    """Flat state columns of one scenario."""
    row = {f"{prefix}layout": s.layout, f"{prefix}case1_action": s.case1_action, f"{prefix}is_law": s.is_law,
           f"{prefix}case1_kills": s.case1_kills, f"{prefix}B_location": s.b_location,
           f"{prefix}margin_all": s.margin_all, f"{prefix}margin_humans": s.margin_humans,
           f"{prefix}theme": s.theme, f"{prefix}theme_confident": s.theme_confident,
           f"{prefix}theme_group_first_side": s.theme_group_first_side}
    for side in "AB":
        d = s.side(side)
        row[f"{prefix}{side}_chars"] = json.dumps(d["chars"])
        row[f"{prefix}{side}_light"] = d["light"]
        for k in SIDE_FIELDS:
            row[f"{prefix}{side}_{k}"] = d[k]
    return row


def _changes(o: sf.Scenario, c: sf.Scenario) -> dict:
    """Original -> counterfactual change columns, compared side by side (A vs A, B vs B)."""
    row = {"layout_changed": o.layout != c.layout, "case1_action_changed": o.case1_action != c.case1_action,
           "theme_changed": o.theme != c.theme,
           "d_margin_all": c.margin_all - o.margin_all, "d_margin_humans": c.margin_humans - o.margin_humans}
    for side in "AB":
        so, sc = o.side(side), c.side(side)
        row[f"{side}_light_changed"] = so["light"] != sc["light"]
        row[f"{side}_chars_changed"] = so["chars"] != sc["chars"]
        for k in SIDE_FIELDS:
            row[f"d_{side}_{k}"] = sc[k] - so[k]
    return row


def build_pairs(df: pd.DataFrame, cols: dict, csv_idx: dict[str, Optional[int]], sets: dict[str, str],
                raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per unique (original text, counterfactual text) pair."""
    o_t, c_t, o_i, c_i = cols["orig_desc"], cols["cf_desc"], cols["orig_idx"], cols["cf_idx"]
    recs = []
    for (ot, ct), g in df.groupby([o_t, c_t], sort=False):
        o, c = sf.parse(ot), sf.parse(ct)
        flags = fe.edit_flags(ot, ct)
        pset = sorted({sets[f] for f in g["source_file"]})
        o_idx, c_idx = sorted({int(x) for x in g[o_i]}), sorted({int(x) for x in g[c_i]})
        rec = {
            "pair_key": pair_key(ot, ct),
            "original_question_idxs": o_idx, "counterfactual_question_idxs": c_idx,
            "pair_sets": "|".join(pset), "in_set_A": "A" in pset, "in_set_B": "B" in pset,
            "source_files": "|".join(sorted(set(g["source_file"]))), "n_answer_rows": int(len(g)),
            "orig_raw_match": csv_idx[ot] is not None, "cf_raw_match": csv_idx[ct] is not None,
            "raw_match": csv_idx[ot] is not None and csv_idx[ct] is not None,
            "orig_csv_idx": csv_idx[ot], "cf_csv_idx": csv_idx[ct],
            "orig_csv_dimension": raw.at[csv_idx[ot], "scenario_dimension"] if csv_idx[ot] is not None else None,
            "cf_csv_dimension": raw.at[csv_idx[ct], "scenario_dimension"] if csv_idx[ct] is not None else None,
            **_state(o, "orig_"), **_state(c, "cf_"), **_changes(o, c), **flags,
            "edit_type": fe.edit_type(ot, ct),
            "original_description": ot, "counterfactual_description": ct,
        }
        recs.append(rec)
    return pd.DataFrame(recs)


def build_answers(df: pd.DataFrame, cols: dict, sets: dict[str, str], pairs: pd.DataFrame,
                  occ: dict[tuple, bool]) -> pd.DataFrame:
    """One row per parquet row (pair x config x file); missing answers flagged, not dropped."""
    kill = {k: (r["orig_case1_kills"], r["cf_case1_kills"]) for k, r in pairs.set_index("pair_key").iterrows()}
    d = df.assign(_thinking=df[cols["thinking"]].map(fe.thinking_label))
    d["_config"] = d[cols["model"]].astype(str) + " (" + d["_thinking"] + ")"
    d["_pk"] = [pair_key(a, b) for a, b in zip(d[cols["orig_desc"]], d[cols["cf_desc"]])]
    dup = d.duplicated(["_config", "_pk"], keep=False)

    def killed(case1_kills: str, ans: Optional[str]) -> Optional[str]:
        if ans is None:
            return None
        return case1_kills if ans == "CASE 1" else ("B" if case1_kills == "A" else "A")

    recs = []
    for (_, row), is_dup in zip(d.iterrows(), dup):
        o_ref, c_ref = fe.clean_answer(row[cols["orig_ref"]]), fe.clean_answer(row[cols["cf_ref"]])
        w, wo, names = fe.simulator_answers(row, cols)
        if o_ref is None or c_ref is None:
            reason = "missing_reference_answer"
        elif len(w) != len(wo):
            reason = "simulator_count_mismatch"
        elif any(a is None for a in w + wo):
            reason = "missing_simulator_answer"
        else:
            reason = None
        n = len(w)
        n_right_with = sum(a == c_ref for a in w) if c_ref else None
        n_right_without = sum(a == c_ref for a in wo) if c_ref else None
        usable = reason is None
        ok, ck = kill[row["_pk"]]
        recs.append({
            "pair_key": row["_pk"], "source_file": row["source_file"], "pair_set": sets[row["source_file"]],
            "ref_config": row["_config"], "model": row[cols["model"]], "thinking": row["_thinking"],
            "duplicate_copy": bool(is_dup),
            "original_question_idx": int(row[cols["orig_idx"]]),
            "counterfactual_question_idx": int(row[cols["cf_idx"]]),
            "orig_raw_match": occ[(row[cols["orig_desc"]], int(row[cols["orig_idx"]]))],
            "cf_raw_match": occ[(row[cols["cf_desc"]], int(row[cols["cf_idx"]]))],
            "original_reference_answer": o_ref, "counterfactual_reference_answer": c_ref,
            "answer_changed": (o_ref != c_ref) if o_ref and c_ref else None,
            "orig_killed_side": killed(ok, o_ref), "cf_killed_side": killed(ck, c_ref),
            "simulator_names": [str(x) for x in names],
            "simulator_answers_with": [a if a is not None else "" for a in w],
            "simulator_answers_without": [a if a is not None else "" for a in wo],
            "n_simulators": n,
            "n_right_with": n_right_with, "n_right_without": n_right_without,
            "acc_with": n_right_with / n if usable and n else None,
            "acc_without": n_right_without / n if usable and n else None,
            # filter_egregious defaults: every simulator wrong with, a majority right without
            "egregious": (n - n_right_with >= n and n_right_without >= n // 2 + 1) if usable else None,
            "usable": usable, "drop_reason": reason,
            "confidence": fe._parsed_field(row, cols, "confidence"),
            "explanation": fe._parsed_field(row, cols, "explanation"),
            "answer_first": row[cols["answer_first"]] if cols["answer_first"] else None,
        })
    return pd.DataFrame(recs)


def build(paths: Iterable[str], raw_csv: Optional[str] = DEFAULT_RAW_CSV) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(pairs_features, answers_long, file -> pair set)."""
    df, cols = load_rows(paths)
    raw = pd.read_csv(raw_csv, index_col=0) if raw_csv and Path(raw_csv).exists() else None
    occ = occurrence_raw_match(df, cols, raw)
    sets = file_pair_sets(df, cols, occ)
    pairs = build_pairs(df, cols, text_csv_idx(occ), sets, raw)
    answers = build_answers(df, cols, sets, pairs, occ)
    return pairs, answers, sets


def main(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="*", help=f"default: {DEFAULT_PARQUETS}")
    ap.add_argument("--raw-csv", default=DEFAULT_RAW_CSV)
    ap.add_argument("--out-dir", default="moral_machines/output")
    a = ap.parse_args(argv)
    pairs, answers, sets = build(a.parquets or sorted(glob.glob(DEFAULT_PARQUETS)), a.raw_csv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out / "pairs_features.parquet", index=False)
    answers.to_parquet(out / "answers_long.parquet", index=False)
    print(f"pairs_features: {len(pairs)} rows, {pairs.shape[1]} columns -> {out / 'pairs_features.parquet'}")
    print(f"answers_long:   {len(answers)} rows ({int(answers['usable'].sum())} usable), {answers.shape[1]} "
          f"columns -> {out / 'answers_long.parquet'}")
    print("pair sets by file:", json.dumps(sets, indent=1))
    print(pairs["pair_sets"].value_counts().to_string())
    return pairs, answers


if __name__ == "__main__":
    main()
