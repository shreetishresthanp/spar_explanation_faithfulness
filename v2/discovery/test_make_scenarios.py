"""Tests for make_scenarios.py (paper originals + Case-order mirrors):  pytest v2/discovery/test_make_scenarios.py -q"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_scenarios as ms  # noqa: E402
from factorial_scenarios import GROUP_LABELS, STATUS, describe  # noqa: E402


@pytest.fixture(scope="module")
def pairs():
    return ms.load_pairs()


@pytest.fixture(scope="module")
def raw():
    return ms.load_raw()


@pytest.fixture(scope="module")
def built(pairs, raw):
    return ms.build(pairs, raw)


@pytest.fixture(scope="module")
def df(built):
    return built[0]


def blocks(q):
    head, rest = q.split("\nCase 1.\n")
    c1, c2 = rest.split("\nCase 2.\n")
    return head, c1, c2


# --- source ------------------------------------------------------------------------------------------------------
def test_source_is_the_shared_paper_pairs(pairs):
    assert len(pairs) == 993 and not pairs.duplicated(ms.KEY).any()


def test_unmirrored_prompts_are_verbatim_from_data_parquet(df, pairs):
    stored = dict(zip(pairs.original_question, pairs.original_question_prompt))
    o = df[~df.mirrored]
    assert all(stored[q] == p for q, p in zip(o.question, o.prompt))


def test_no_counterfactual_scenarios(df, pairs):
    assert not set(df.question) & set(pairs.counterfactual_question)
    assert set(df[~df.mirrored].question) <= set(pairs.original_question)


# --- sample size and split ---------------------------------------------------------------------------------------
def test_sizes(df, built):
    alloc = built[1]
    assert (df.split == "train").sum() == 1000 and alloc.train.sum() == 500
    assert (df.split == "test").sum() == 2 * alloc.test.sum()
    assert df.groupby("pair_id").size().eq(2).all() and df.groupby("base_id").size().eq(2).all()
    assert df.question.is_unique


def test_floor_and_allocation(df, built):
    alloc = built[1]
    assert (alloc.train >= 37).all() and (alloc.train + alloc.test <= alloc.available).all()
    got = df[~df.mirrored].groupby(["theme", "split"]).size().unstack().reindex(alloc.index)
    assert (got.train == alloc.train).all() and (got.test == alloc.test).all()
    # Themes above the floor are within one scenario of proportional among themselves.
    big = alloc[alloc.train > 37]
    share = big.available / big.available.sum() * big.train.sum()
    assert (big.train - share).abs().max() <= 1


def test_mirror_stays_in_same_split(df):
    assert df.groupby("base_id").split.nunique().eq(1).all()


def test_seed_changes_sample_but_not_allocation(pairs, raw, df, built):
    other, alloc2 = ms.build(pairs, raw, seed=1)
    assert alloc2.equals(built[1]) and set(other.pair_id) != set(df.pair_id)


# --- mirrors -----------------------------------------------------------------------------------------------------
def test_mirror_only_swaps_the_cases(df):
    for _, g in df.groupby("base_id"):
        o, m = g[~g.mirrored].iloc[0], g[g.mirrored].iloc[0]
        (h1, a1, a2), (h2, b1, b2) = blocks(o.question), blocks(m.question)
        assert h1 == h2 and a1 == b2 and a2 == b1
        for k in ("victims", "n", "group", "light"):
            assert o[f"case1_{k}"] == m[f"case2_{k}"] and o[f"case2_{k}"] == m[f"case1_{k}"]
        assert m.case1_swerve != o.case1_swerve
        assert (o.split, o.answer_first, o.pair_id) == (m.split, m.answer_first, m.pair_id)
        # prompt differs from the paper's only in the swapped scenario text
        assert m.prompt == o.prompt.replace(o.question, m.question)


def test_columns_match_text(df):
    for r in df.itertuples():
        _, c1, c2 = blocks(r.question)
        for c, vic, light in ((c1, r.case1_victims, r.case1_light), (c2, r.case2_victims, r.case2_light)):
            dies = c.split("Meanwhile")[0]
            assert f"death of {describe(json.loads(vic))}, who" in dies
            assert light == "none" or STATUS[light] in dies
        assert c1.startswith("A self-driving car with sudden brake failure will swerve") == r.case1_swerve
        assert ("inside the car." in c1.split("Meanwhile")[0]) == r.case1_passengers
        assert ("inside the car" in r.question) == r.in_car and ("signal" in r.question) == r.law


def test_mirror_breaks_the_label_confound(df):
    orig = df[~df.mirrored]
    assert (orig.case1_passengers == orig.in_car).all()                      # the paper's confound
    assert df[df.in_car].case1_passengers.mean() == 0.5
    first = df.apply(lambda r: r.case1_group == GROUP_LABELS[r.theme][0], axis=1)
    assert first.groupby(df.theme).mean().eq(0.5).all()


def test_answer_first_crossed_with_mirror(df):
    t = pd.crosstab(df.mirrored, df.answer_first)
    assert t.loc[True].equals(t.loc[False])


def test_report_renders(df, built):
    class A:
        train_rows, floor, seed = 1000, 37, 0
    page = ms.report(df, built[1], 993, A)
    assert "1. Theme breakdown" in page and "2. Mirror breakdown" in page and "<table" in page


# --- failure cases -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [0, -4, 999, 1001])
def test_train_rows_must_be_positive_even(pairs, raw, rows):
    with pytest.raises(ValueError, match="even"):
        ms.build(pairs, raw, train_rows=rows)


def test_floor_too_high_for_rare_theme(pairs, raw):
    with pytest.raises(ValueError, match="can supply"):                     # species has 50 originals
        ms.build(pairs, raw, train_rows=1000, floor=38)


def test_floor_times_themes_exceeds_budget(pairs, raw):
    with pytest.raises(ValueError, match="exceeds"):
        ms.build(pairs, raw, train_rows=400, floor=37)


def test_budget_exceeds_supply(pairs, raw):
    with pytest.raises(ValueError, match="available"):
        ms.build(pairs, raw, train_rows=2000, floor=10)


@pytest.mark.parametrize("frac", [0, 1, -0.2])
def test_bad_test_frac(pairs, raw, frac):
    with pytest.raises(ValueError):
        ms.build(pairs, raw, test_frac=frac)


def test_missing_raw_factors(pairs, raw):
    q = pairs.original_question.iloc[0]
    with pytest.raises(KeyError, match="no factors"):
        ms.build(pairs, {k: v for k, v in raw.items() if k != q})


def test_wrong_factors(pairs, raw):
    bad = {q: dict(f, interv=not f["interv"]) for q, f in raw.items()}
    with pytest.raises(ValueError, match="re-rendered"):
        ms.build(pairs, bad)


def test_stored_prompt_mismatch(pairs, raw):
    p = pairs.copy()
    p["original_question_prompt"] = p.original_question_prompt + " "
    with pytest.raises(ValueError, match="stored prompt"):
        ms.build(p, raw)


def test_source_files_disagree(tmp_path):
    a = pd.read_parquet(ms.SOURCES[0], columns=ms.COLS)
    a = a[a.original_dataset == "moral_machines"].head(20)
    b = a.copy()
    b.loc[b.index[0], "original_question_prompt"] += "x"
    a.to_parquet(tmp_path / "a.parquet")
    b.to_parquet(tmp_path / "b.parquet")
    with pytest.raises(ValueError, match="differ"):
        ms.load_pairs([tmp_path / "a.parquet", tmp_path / "b.parquet"])
