"""Tests for factorial_scenarios.py:  pytest v2/discovery/test_factorial_scenarios.py -q"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from factorial_scenarios import STATUS, THEMES, make_scenarios, render, sample_groups, tu  # noqa: E402

BINS = ["x_swerve", "in_car", "law", "x_case1", "answer_first"]


@pytest.fixture(scope="module")
def df():
    return make_scenarios(2400, seed=0)


def cases(q):
    c1, c2 = q.split("Case 1.\n")[1].split("\nCase 2.\n")
    return c1, c2


def deaths_and_spares(block):
    return (re.search(r"the death of (.+?), who", block).group(1), re.search(r"spare (.+?), who", block).group(1))


# --- fidelity: same configuration as the original generator -> identical text -------------------------------------
@pytest.mark.parametrize("seed", range(300))
def test_matches_original_generator(seed):
    random.seed(seed)
    dim = random.choice(THEMES)
    in_car, interv, law = random.choice([True, False]), random.choice([True, False]), random.choice([True, False])
    system, user, info = tu.generate_moral_machine_scenarios(dim, in_car, interv, law)
    lights = ["none" if v == "NA" else v for v in info["traffic_light_pattern"]]
    # Original: Case 1 kills count_dict_1 (passengers when in_car), and Case 1's action is the swerve iff interventionism.
    ours = render(Counter(info["count_dict_1"]), Counter(info["count_dict_2"]), x_swerve=interv, in_car=in_car,
                  x_passengers=True, light_x=lights[0], light_y=lights[1], x_case1=True)
    assert ours == f"{system}\n{user}"


# --- the dataset -------------------------------------------------------------------------------------------------
def test_text_matches_factors(df):
    for r in df.itertuples():
        c1, c2 = cases(r.question)
        x_block, y_block = (c1, c2) if r.x_case1 else (c2, c1)
        x_desc, y_desc = deaths_and_spares(x_block)
        assert deaths_and_spares(y_block) == (y_desc, x_desc)                       # each case spares the other's victims
        assert x_block.split("will ")[1].startswith("swerve") == r.x_swerve
        assert ("inside the car" in r.question) == r.in_car
        if r.in_car:
            assert ("who " in x_block and "inside the car." in x_block.split("Meanwhile")[0]) == r.x_passengers
        assert ("signal" in r.question) == r.law
        if r.light_x != "none":
            assert STATUS[r.light_x] in x_block.split("Meanwhile")[0]
        if r.light_y != "none":
            assert STATUS[r.light_y] in y_block.split("Meanwhile")[0]
        assert sum(json.loads(r.x_chars).values()) == r.n_x
        assert re.search(rf"death of {re.escape(x_desc)}, who {'were' if r.n_x > 1 else 'was'} ", x_block)


def test_swapping_case_order_only_swaps_blocks():
    rng = np.random.default_rng(1)
    x, y = sample_groups("gender", 3, rng)
    a = render(x, y, True, False, False, "red", "green", x_case1=True)
    b = render(x, y, True, False, False, "red", "green", x_case1=False)
    (a1, a2), (b1, b2) = cases(a), cases(b)
    assert (a1, a2) == (b2, b1) and a.split("Case 1.")[0] == b.split("Case 1.")[0]


def test_balance_and_independence(df):
    # 2400 = 3 full factorial tiles (768 cells) + 96 random cells, so only the partial tile adds imbalance.
    assert (df[BINS].mean() - 0.5).abs().max() <= 0.02
    counts = df.theme.value_counts()
    assert counts.max() - counts.min() <= 0.02 * len(df)
    corr = np.abs(np.triu(df[BINS + ["x_passengers"]][df.in_car].astype(float).corr(), 1))  # within in_car
    assert np.nanmax(corr) < 0.04
    corr_all = np.abs(np.triu(df[BINS].astype(float).corr(), 1))
    assert corr_all.max() < 0.03
    for col in BINS:                                                                # no theme x factor confound
        assert df.groupby("theme")[col].mean().between(0.45, 0.55).all(), col
    assert df[df.in_car].x_passengers.mean() == pytest.approx(0.5, abs=0.02)
    assert (df[df.law & ~df.in_car].light_x == "red").mean() == pytest.approx(0.5, abs=0.03)


def test_exactly_orthogonal_on_full_tiles():
    full = make_scenarios(768, seed=5)
    assert (full[BINS].mean() == 0.5).all() and full.theme.value_counts().nunique() == 1
    assert np.abs(np.triu(full[BINS].astype(float).corr(), 1)).max() < 1e-12


def test_lights_consistent(df):
    assert (df[~df.law][["light_x", "light_y"]] == "none").all().all()
    car = df[df.in_car & df.law]
    assert ((car.light_x == "none") == car.x_passengers).all() and ((car.light_y == "none") != car.x_passengers).all()
    ped = df[~df.in_car & df.law]
    assert (ped.light_x != ped.light_y).all() and ped.light_x.isin(["red", "green"]).all()


def test_unique_split_and_prompt(df):
    assert df.question.is_unique and df.scenario_id.is_unique
    share = df.groupby("theme").split.apply(lambda s: (s == "test").mean())
    assert share.between(0.24, 0.26).all()
    for r in df.head(200).itertuples():
        first = r.prompt.index("[ANSWER]") < r.prompt.index("[EXPLANATION]")
        assert first == r.answer_first and r.question in r.prompt


def test_groups_follow_theme():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = sample_groups("utilitarianism", int(rng.integers(1, 6)), rng)
        assert sum(y.values()) > sum(x.values()) and sum(y.values()) <= 5
        x, y = sample_groups("species", 2, rng)
        assert set(y) <= set(tu.pets) and set(x) <= set(tu.humans)
        x, y = sample_groups("gender", 3, rng)
        assert set(x) <= set(tu.female) and set(y) <= set(tu.male)


def test_deterministic():
    a, b, c = make_scenarios(60, seed=3), make_scenarios(60, seed=3), make_scenarios(60, seed=4)
    assert a.equals(b) and not a.question.equals(c.question)


# --- failure cases -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, -5, 5])
def test_rejects_bad_n(n):
    with pytest.raises(ValueError):
        make_scenarios(n)


@pytest.mark.parametrize("frac", [0, 1, -0.1, 1.5])
def test_rejects_bad_test_frac(frac):
    with pytest.raises(ValueError):
        make_scenarios(60, test_frac=frac)


def test_rejects_unknown_theme():
    with pytest.raises(ValueError):
        sample_groups("random", 2, np.random.default_rng(0))


@pytest.mark.parametrize("kw", [
    dict(in_car=True, x_passengers=True, light_x="red", light_y="none"),     # passengers with a light
    dict(in_car=False, x_passengers=False, light_x="red", light_y="red"),    # both pedestrians on red
    dict(in_car=False, x_passengers=False, light_x="red", light_y="none"),   # only one pedestrian light
    dict(in_car=False, x_passengers=False, light_x="amber", light_y="red"),  # unknown light
])
def test_rejects_impossible_configs(kw):
    with pytest.raises(ValueError):
        render(Counter({"man": 1}), Counter({"woman": 1}), x_swerve=True, x_case1=True, **kw)


def test_rejects_empty_group():
    with pytest.raises(ValueError):
        render(Counter(), Counter({"woman": 1}), True, False, False, "none", "none", True)


def test_unique_retry_exhaustion(monkeypatch):
    # If every draw yields the same text, generation must fail loudly rather than emit duplicates.
    import factorial_scenarios as ms
    monkeypatch.setattr(ms, "sample_groups", lambda *a: (Counter({"man": 1}), Counter({"dog": 1})))
    with pytest.raises(RuntimeError):
        ms.make_scenarios(60, seed=0, max_tries=3)


def test_bumps_are_rare(df):
    assert df.n_pairs_bumped.mean() < 0.05
    assert df.n_pairs.value_counts(normalize=True).between(0.15, 0.25).all()
