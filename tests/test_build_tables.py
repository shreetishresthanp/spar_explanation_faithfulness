"""Consistency tests for moral_machines/build_tables.py on the real prediction parquets (skipped if absent)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_machines import build_tables as bt  # noqa: E402
from moral_machines import filter_egregious as fe  # noqa: E402
from moral_machines import scenario_features as sf  # noqa: E402

PARQUETS = sorted(str(p) for p in (ROOT / "data_parquet").glob("*.parquet"))
RAW = ROOT / bt.DEFAULT_RAW_CSV
pytestmark = [pytest.mark.data_quality,
              pytest.mark.skipif(not PARQUETS or not RAW.exists(), reason="needs data_parquet/ and the raw CSV")]


@pytest.fixture(scope="module")
def built():
    return bt.build(PARQUETS, str(RAW))


@pytest.fixture(scope="module")
def mm_rows():
    df, cols = bt.load_rows(PARQUETS)
    return df, cols


def test_pairs_one_row_per_text_pair(built, mm_rows):
    pairs, _, _ = built
    df, cols = mm_rows
    assert pairs["pair_key"].is_unique
    assert len(pairs) == len(set(zip(df[cols["orig_desc"]], df[cols["cf_desc"]])))
    assert pairs["n_answer_rows"].sum() == len(df)


def test_pair_sets(built):
    pairs, answers, sets = built
    assert set(sets.values()) <= {"A", "B"}
    # set A files match the CSV everywhere; set B files contain rows that do not
    for f, s in sets.items():
        a = answers[answers["source_file"] == f]
        assert (a["orig_raw_match"] & a["cf_raw_match"]).all() == (s == "A"), f
    assert set(pairs["pair_sets"]) <= {"A", "B", "A|B"}


def test_labels_match_filter_egregious(built):
    pairs, _, _ = built
    for r in pairs.itertuples():
        assert r.edit_type == fe.edit_type(r.original_description, r.counterfactual_description)
        flags = fe.edit_flags(r.original_description, r.counterfactual_description)
        assert all(getattr(r, k) == v for k, v in flags.items())


def test_raw_match_settings_equal_csv(built):
    pairs, _, _ = built
    raw = pd.read_csv(RAW, index_col=0)
    m = pairs[pairs["raw_match"]]
    assert len(m) > 0.5 * len(pairs)
    for r in m.itertuples():
        for text, idx, dim, theme in ((r.original_description, r.orig_csv_idx, r.orig_csv_dimension, r.orig_theme),
                                      (r.counterfactual_description, r.cf_csv_idx, r.cf_csv_dimension, r.cf_theme)):
            row = raw.loc[int(idx)]
            assert row["prompt"] == text
            assert sf.parse(text) == sf.from_generator_info(row)
            assert dim == row["scenario_dimension"] == theme


def test_state_round_trips(built):
    pairs, _, _ = built
    for r in pairs.head(300).itertuples():
        s = sf.Scenario(r.orig_layout, r.orig_case1_action, json.loads(r.orig_A_chars), json.loads(r.orig_B_chars),
                        r.orig_A_light, r.orig_B_light)
        assert sf.render(s) == r.original_description


def test_answers_one_row_per_parquet_row_and_match_filter(built, mm_rows):
    pairs, answers, _ = built
    df, _ = mm_rows
    assert len(answers) == len(df)
    assert set(answers["pair_key"]) == set(pairs["pair_key"])
    elig, dropped, _ = fe.annotate(df, thinking="all", keep_duplicates=True)
    assert int(answers["usable"].sum()) == len(elig)
    assert int((~answers["usable"]).sum()) == sum(d["count"] for d in dropped)
    # same per-row numbers as filter_egregious for usable rows (key: config, pair indices, file)
    u = answers[answers["usable"]].copy()
    u["n_wrong_with"] = u["n_simulators"] - u["n_right_with"]
    key = ["model", "thinking", "original_question_idx", "counterfactual_question_idx", "source_file"]
    m = u.merge(elig, on=key, suffixes=("", "_fe"), validate="one_to_one")
    assert len(m) == len(u)
    for c in ("n_wrong_with", "n_right_without", "egregious", "answer_changed", "original_reference_answer",
              "counterfactual_reference_answer"):
        assert (m[c].astype(object) == m[f"{c}_fe"].astype(object)).all(), c


def test_killed_side_matches_killed_location(built):
    pairs, answers, _ = built
    p = pairs.set_index("pair_key")
    for r in answers[answers["usable"]].head(2000).itertuples():
        text = p.at[r.pair_key, "counterfactual_description"]
        loc = fe.killed_location(text, r.counterfactual_reference_answer)
        side_loc = p.at[r.pair_key, "cf_B_location"] if r.cf_killed_side == "B" else \
            sf.parse(text).a_location
        assert loc == side_loc
