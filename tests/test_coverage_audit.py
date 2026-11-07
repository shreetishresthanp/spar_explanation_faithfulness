"""Unit tests for moral_machines/coverage_audit.py on a small synthetic pairs table."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_machines import coverage_audit as ca  # noqa: E402

pytestmark = pytest.mark.unit

A = json.dumps([["man", 1], ["woman", 1]])
B = json.dumps([["boy", 1], ["girl", 1]])


def pair(key, law=False, margin=False, case_order=False, layout="two_lanes", theme="gender", kills="A", first="A"):
    return {"pair_key": key, "in_set_A": True, "in_set_B": False,
            "law_changed": law, "species_changed": False, "margin_changed": margin, "size_changed": False,
            "case1_action_changed": False, "layout_changed": False,
            "orig_case1_kills": kills, "cf_case1_kills": ("B" if kills == "A" else "A") if case_order else kills,
            "orig_A_chars": A, "cf_A_chars": A, "orig_B_chars": B, "cf_B_chars": B,
            **{f"{w}_{c}": v for w in ("orig", "cf") for c, v in
               [("layout", layout), ("theme", theme), ("theme_group_first_side", first)]}}


@pytest.fixture
def df():
    rows = [pair("law_only", law=True), pair("law_margin", law=True, margin=True),
            pair("law_order", law=True, case_order=True)]
    rows += [pair(f"car{i}", layout="car_vs_ped", kills="A", first="AB"[i % 2]) for i in range(4)]
    rows += [pair(f"lane{i}", kills="AB"[i % 2], first="A") for i in range(3)]
    return pd.DataFrame(rows)


def table1(df):
    return ca.changes_table(ca.factor_matrix(df)).set_index("factor")


def test_law_only_counts_alone(df):
    t = table1(df.iloc[[0]])
    assert t.loc["law", "n_alone"] == 1 and t.loc["law", "n_with_others"] == 0


def test_law_plus_margin_counts_with_others_for_both(df):
    t = table1(df.iloc[[1]])
    assert t.loc["law", "n_with_others"] == 1 and t.loc["margin", "n_with_others"] == 1
    assert t.loc["law", "n_alone"] == 0 and t.loc["margin", "n_alone"] == 0


def test_unchanged_factor_is_never(df):
    t = table1(df)
    assert t.loc["species", "status"] == "never"
    assert (t["n_alone"] + t["n_with_others"] == t["n_total"]).all()


def test_case_order_plus_law_is_two_factors(df):
    m = ca.factor_matrix(df.iloc[[2]])
    assert m.iloc[0].sum() == 2 and m.iloc[0]["case order"] and m.iloc[0]["law"]
    assert table1(df.iloc[[2]]).loc["law", "n_alone"] == 0


def test_car_always_kills_passengers_gives_always(df):
    c1 = ca.case1_table(df)
    car = c1[(c1["scenarios"] == "original") & (c1["layout"] == "car_vs_ped")].iloc[0]
    assert car["pct_case1_kills_passengers"] == 100 and car["pct_case1_kills_first_group"] == 50
    assert bool(car["always"])
    lane = c1[(c1["scenarios"] == "original") & (c1["layout"] == "two_lanes")].iloc[0]
    assert pd.isna(lane["pct_case1_kills_passengers"])


def test_summary_contains_all_three_tables(df, tmp_path):
    path = tmp_path / "pairs.parquet"
    df.to_parquet(path)
    ca.main(["--input", str(path), "--out-dir", str(tmp_path), "--by-pair-set"])
    md = (tmp_path / "coverage_summary.md").read_text()
    for heading in ["## 1. What changes", "## 2. What Case 1 kills", "## 3. Factors NSG can't evaluate"]:
        assert heading in md
    assert "| factor | n_alone |" in md and "| layout | theme |" in md and "| item | detail |" in md
    for name in ["changes", "combos", "case1", "blindspots"]:
        assert (tmp_path / f"coverage_{name}.csv").exists()
