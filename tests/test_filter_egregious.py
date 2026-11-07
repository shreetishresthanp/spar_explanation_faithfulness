"""Unit tests for moral_machines/filter_egregious.py on small synthetic DataFrames."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moral_machines.filter_egregious import (  # noqa: E402
    annotate, build_comparison, clean_answer, edit_type, main, parse_scenario,
)

pytestmark = pytest.mark.unit

P = "counterfactual_predictor_response_"
STEM = ("Consider that pedestrians are crossing a zebra crossing that spans two lanes, divided by an island. "
        "Please indicate which case is better for autonomous driving by choosing either 'Case 1' or 'Case 2'. "
        "Respond with one word only.\n\n")


def scen(ahead, other, ahead_law="", other_law=""):
    """Two-lane pedestrian scenario. ahead/other: character strings like '1 girl'; *_law: '', 'green', 'red'."""
    def who(chars, law, where):
        verb = "was" if chars.startswith("1 ") and "," not in chars and " and " not in chars else "were"
        lawtxt = {"": "crossing",
                  "green": "abiding by the law by crossing on the green signal",
                  "red": "flouting the law by crossing on the red signal"}[law]
        return f"{chars}, who {verb} {lawtxt} {where}"
    a = who(ahead, ahead_law, "ahead of the car")
    o = who(other, other_law, "in the other lane")
    return (STEM + "Case 1.\nA self-driving car with sudden brake failure will continue ahead and drive through a "
            f"pedestrian crossing ahead. This will result in the death of {a}. Meanwhile, it will spare {o}.\n\n"
            "Case 2.\nA self-driving car with sudden brake failure will swerve and drive through a pedestrian "
            f"crossing in the other lane. This will result in the death of {o}. Meanwhile, it will spare {a}.\n")


DEFAULT_O = scen("1 man", "1 woman")
DEFAULT_C = scen("2 men", "1 woman")  # margin_changed


_IDX = iter(range(10_000, 99_999))


def row(cf_ref="CASE 1", orig_ref="CASE 1", w="CASE 2", wo="CASE 1", w_list=None, wo_list=None,
        model="m/a", dataset="moral_machines", thinking="medium", o=DEFAULT_O, c=DEFAULT_C):
    i = next(_IDX)
    return {
        "original_dataset": dataset,
        "original_reference_response_model_info_model": model,
        "original_reference_response_model_info_thinking": thinking,
        "original_question_idx": i, "counterfactual_question_idx": 100_000_000 + i,
        "original_description": o, "counterfactual_description": c,
        "original_reference_response_answer": orig_ref,
        "counterfactual_reference_response_answer": cf_ref,
        P + "with_explanation_answer": w, P + "without_explanation_answer": wo,
        P + "with_explanation_predictor_answers": w_list or [],
        P + "with_explanation_predictor_names": [f"p{k}" for k in range(len(w_list or []))],
        P + "without_explanation_predictor_answers": wo_list or [],
        P + "without_explanation_predictor_names": [f"p{k}" for k in range(len(wo_list or []))],
        "original_reference_response_parsed_response": {"confidence": "HIGH", "explanation": "because"},
        "original_answer_first": True,
    }


def run(rows, **kw):
    return annotate(pd.DataFrame(rows), "medium", **kw)


# --- answer cleaning --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (" case 1 ", "CASE 1"), ("CASE1", "CASE 1"), ("Case 2", "CASE 2"), ("Case 1", "CASE 1"),
    (None, None), ("", None), ("maybe", None), ("CASE 3", None), (float("nan"), None),
])
def test_clean_answer(raw, expected):
    assert clean_answer(raw) == expected


# --- scope filters ----------------------------------------------------------------------------
def test_non_mm_and_wrong_thinking_excluded():
    elig, dropped, scope = run([row(), row(dataset="income"), row(thinking="high")])
    assert len(elig) == 1
    assert scope == {"input_rows": 3, "moral_machines_rows": 2, "thinking_rows": 1, "duplicate_rows_removed": 0}
    assert dropped == []


def test_missing_reference_dropped_and_counted():
    elig, dropped, _ = run([row(), row(cf_ref=None), row(orig_ref="")])
    assert len(elig) == 1
    assert dropped == [{"ref_config": "m/a", "reason": "missing_reference_answer", "count": 2}]


def test_missing_simulator_answer_dropped_and_counted():
    elig, dropped, _ = run([row(w="garbage"),
                            row(w_list=["CASE 2"] * 5, wo_list=["CASE 1"] * 4 + [None])])
    assert elig.empty
    assert dropped == [{"ref_config": "m/a", "reason": "missing_simulator_answer", "count": 2}]


# --- egregious logic --------------------------------------------------------------------------
@pytest.mark.parametrize("w,wo,keep", [
    ("CASE 2", "CASE 1", True),    # wrong with, right without
    ("CASE 1", "CASE 1", False),   # right with, right without
    ("CASE 2", "CASE 2", False),   # wrong with, wrong without
    ("CASE 1", "CASE 2", False),   # right with, wrong without
])
def test_single_simulator_truth_table(w, wo, keep):
    elig, _, _ = run([row(cf_ref="CASE 1", w=w, wo=wo)])
    assert elig["n_simulators"].iloc[0] == 1
    assert bool(elig["egregious"].iloc[0]) is keep


@pytest.mark.parametrize("n_wrong_with,n_right_without,keep", [
    (5, 3, True), (4, 3, False), (5, 2, False), (5, 5, True),
])
def test_five_simulators(n_wrong_with, n_right_without, keep):
    w = ["CASE 2"] * n_wrong_with + ["CASE 1"] * (5 - n_wrong_with)
    wo = ["CASE 1"] * n_right_without + ["CASE 2"] * (5 - n_right_without)
    # single-answer columns deliberately disagree, to prove the lists are used
    elig, _, _ = run([row(cf_ref="CASE 1", w="CASE 1", wo="CASE 2", w_list=w, wo_list=wo)])
    r = elig.iloc[0]
    assert (r.n_simulators, r.n_wrong_with, r.n_right_without) == (5, n_wrong_with, n_right_without)
    assert bool(r.egregious) is keep


@pytest.mark.parametrize("encode", [
    lambda l: str(l),                                   # "['CASE 2', ...]"
    lambda l: str(__import__("json").dumps(l)),         # '["CASE 2", ...]'
    lambda l: "[" + " ".join(f"'{a}'" for a in l) + "]",  # numpy repr, no commas
])
def test_text_encoded_lists_are_parsed(encode):
    w = ["CASE 2"] * 5
    wo = ["CASE 1"] * 3 + ["CASE 2"] * 2
    r = row(cf_ref="CASE 1", w="CASE 1", wo="CASE 2")
    r[P + "with_explanation_predictor_answers"] = encode(w)
    r[P + "without_explanation_predictor_answers"] = encode(wo)
    elig, _, _ = run([r])
    e = elig.iloc[0]
    assert (e.n_simulators, e.n_wrong_with, e.n_right_without, bool(e.egregious)) == (5, 5, 3, True)


def test_explicit_thresholds_override_defaults():
    w = ["CASE 2"] * 4 + ["CASE 1"]
    wo = ["CASE 1"] * 3 + ["CASE 2"] * 2
    elig, _, _ = run([row(w_list=w, wo_list=wo)], min_wrong_with=4, min_right_without=3)
    assert bool(elig["egregious"].iloc[0])


def test_empty_lists_fall_back_to_single_simulator():
    elig, _, _ = run([row(w="case 2", wo="Case1", w_list=[], wo_list=[])])
    r = elig.iloc[0]
    assert r.n_simulators == 1 and bool(r.egregious)


def test_answer_changed():
    elig, _, _ = run([row(orig_ref="CASE 1", cf_ref="case 1"), row(orig_ref="CASE 1", cf_ref="CASE 2")])
    assert elig["answer_changed"].tolist() == [False, True]


def test_annotation_columns_and_pair_id():
    elig, _, _ = run([row()])
    r = elig.iloc[0]
    assert r.pair_id == f"m/a|{r.original_question_idx}|{r.counterfactual_question_idx}"
    assert (r.confidence, r.explanation, r.answer_first) == ("HIGH", "because", True)
    assert r.edit_type == "margin_changed"
    assert (r.margin_changed, r.size_changed, r.species_changed) == (True, False, False)


# --- parser & edit_type -----------------------------------------------------------------------
def test_parser_singular_and_plural():
    g = parse_scenario(scen("1 girl", "3 cats and 1 girl", "red", "green"))
    assert g["ahead"]["chars"] == {"girl": 1} and g["ahead"]["law"] == "red"
    assert g["other_lane"]["chars"] == {"cat": 3, "girl": 1} and g["other_lane"]["law"] == "green"
    assert "who was" in scen("1 girl", "2 dogs") and "who were" in scen("1 girl", "2 dogs")
    g = parse_scenario(scen("2 elderly women, 1 baby, and 1 homeless person", "2 babies"))
    assert g["ahead"]["chars"] == {"elderly woman": 2, "baby": 1, "homeless person": 1}
    assert g["other_lane"]["chars"] == {"baby": 2}


def test_parser_in_car_scenario():
    t = ("Consider that pedestrians are crossing a zebra crossing that spans two lanes. Please indicate which "
         "case is better for autonomous driving by choosing either 'Case 1' or 'Case 2'. Respond with one word "
         "only.\n\nCase 1.\nA self-driving car with sudden brake failure will swerve and crash into a concrete "
         "barrier. This will result in the death of 1 man, who was inside the car. Meanwhile, it will spare "
         "2 dogs, who were crossing.\n\nCase 2.\nA self-driving car with sudden brake failure will continue "
         "ahead and drive through a pedestrian crossing ahead. This will result in the death of 2 dogs, who were "
         "crossing. Meanwhile, it will spare 1 man, who was inside the car.\n")
    g = parse_scenario(t)
    assert g["car"]["chars"] == {"man": 1} and g["ahead"]["chars"] == {"dog": 2}  # Case 1 swerves into the barrier -> pedestrians ahead


@pytest.mark.parametrize("o,c,expected", [
    # light at each lane flips AND species/margin change -> law_moved wins
    (scen("1 girl", "1 cat", "green", "red"), scen("2 cats", "3 cats and 1 girl", "red", "green"), "law_moved"),
    (scen("1 girl", "1 cat", "green", "red"), scen("1 cat", "1 girl", "red", "green"), "law_moved"),
    # one extra person on one side: gap 0 -> 1
    (scen("1 man", "1 woman"), scen("2 men", "1 woman"), "margin_changed"),
    # one extra person on BOTH sides: gap stays 0
    (scen("1 man", "1 woman"), scen("2 men", "2 women"), "size_changed"),
    # adding a cat to one side changes species AND margin -> species wins
    (scen("1 man", "1 woman"), scen("1 man and 1 cat", "1 woman"), "species_changed"),
    # same totals, animal moved across -> species
    (scen("1 man and 1 cat", "2 women"), scen("2 men", "1 woman and 1 cat"), "species_changed"),
    (scen("1 man", "1 woman"), scen("1 woman", "1 man"), "demographic_reshuffle"),
    (scen("1 male doctor and 1 female executive", "2 men"),
     scen("1 male executive and 1 female doctor", "2 men"), "within_category_reshuffle"),
])
def test_edit_type(o, c, expected):
    assert edit_type(o, c) == expected


# --- comparison set ---------------------------------------------------------------------------
def _pool():
    rows = []
    for model in ("m/a", "m/b"):
        rows += [row(model=model, w="CASE 2", wo="CASE 1") for _ in range(3)]          # egregious
        rows += [row(model=model, w="CASE 1", wo="CASE 1") for _ in range(8)]          # non-egregious match
        rows += [row(model=model, w="CASE 1", wo="CASE 1", orig_ref="CASE 2") for _ in range(5)]  # other key
    return run(rows)[0]


def test_comparison_matched_reproducible_disjoint():
    elig = _pool()
    egr = elig[elig.egregious]
    comp, short = build_comparison(elig, seed=7)
    assert len(comp) == len(egr) == 6 and short == []
    keys = ["ref_config", "edit_type", "answer_changed"]
    assert comp.groupby(keys).size().to_dict() == egr.groupby(keys).size().to_dict()
    assert not set(comp.pair_id) & set(egr.pair_id)
    assert comp.pair_id.is_unique
    comp2, _ = build_comparison(elig.sample(frac=1, random_state=1), seed=7)
    assert comp.pair_id.tolist() == comp2.pair_id.tolist()
    comp3, _ = build_comparison(elig, seed=8)
    assert set(comp3.pair_id) != set(comp.pair_id)


def test_comparison_shortfall_logged():
    rows = [row(w="CASE 2", wo="CASE 1") for _ in range(4)] + [row(w="CASE 1", wo="CASE 1")]
    elig = run(rows)[0]
    comp, short = build_comparison(elig, seed=0)
    assert len(comp) == 1
    assert short == [{"ref_config": "m/a", "edit_type": "margin_changed", "answer_changed": False,
                      "needed": 4, "available": 1}]


def test_main_end_to_end(tmp_path):
    rows = [row(w="CASE 2", wo="CASE 1"), row(w="CASE 1", wo="CASE 1"), row(cf_ref=None)]
    p = tmp_path / "in.parquet"
    pd.DataFrame(rows).to_parquet(p)
    s = main([str(p), "--out-dir", str(tmp_path / "out")])
    assert s["counts_per_ref_config"]["m/a"] == {"eligible": 2, "egregious": 1, "comparison": 1, "dropped_missing": 1}
    for f in ("all_pairs.csv", "egregious_candidates.csv", "comparison_set.csv", "summary.json"):
        assert (tmp_path / "out" / f).exists()
    allp = pd.read_csv(tmp_path / "out" / "all_pairs.csv")
    egr = pd.read_csv(tmp_path / "out" / "egregious_candidates.csv")
    assert len(allp) == 2                                  # the missing-reference row is excluded
    assert allp.egregious.tolist() == [True, False]
    assert list(allp.columns) == list(egr.columns)          # same columns, incl. the egregious flag
    assert set(egr.pair_id) == set(allp[allp.egregious].pair_id)


# --- multiple thinking levels / files ---------------------------------------------------------
def test_thinking_all_separates_configs_and_labels_missing():
    rows = [row(thinking="medium"), row(thinking="high"), row(thinking=None), row(thinking=True)]
    elig, _, _ = annotate(pd.DataFrame(rows), "all")
    assert sorted(elig.ref_config) == ["m/a (True)", "m/a (high)", "m/a (medium)", "m/a (unset)"]
    assert elig.pair_id.str.startswith("m/a (").all()
    elig2, _, _ = annotate(pd.DataFrame(rows), "medium,unset")
    assert sorted(elig2.thinking) == ["medium", "unset"]


def test_single_thinking_keeps_plain_pair_id():
    elig, _, _ = run([row()])
    assert elig.ref_config.iloc[0] == "m/a" and elig.pair_id.iloc[0].startswith("m/a|")


def test_duplicate_config_pair_across_files_removed():
    r = row()
    a, b = dict(r, source_file="a.parquet"), dict(r, source_file="b.parquet")
    other_level = dict(r, source_file="b.parquet",
                       original_reference_response_model_info_thinking="high")
    elig, _, scope = annotate(pd.DataFrame([a, b, other_level]), "all")
    assert scope["duplicate_rows_removed"] == 1
    assert len(elig) == 2 and elig.source_file.tolist() == ["a.parquet", "b.parquet"]


def test_file_thinking_restricts_levels_per_file():
    from moral_machines.filter_egregious import annotate as ann
    rows = [dict(row(thinking=t), source_file="itc.parquet") for t in ("high", "medium", "low", "none")]
    rows += [dict(row(thinking="low"), source_file="regular.parquet")]
    elig, _, _ = ann(pd.DataFrame(rows), "all", file_thinking={"itc.parquet": ["high", "medium"]})
    assert sorted(zip(elig.source_file, elig.thinking)) == [
        ("itc.parquet", "high"), ("itc.parquet", "medium"), ("regular.parquet", "low")]


def test_keep_duplicates_tags_copies_and_compares_them():
    from moral_machines.filter_egregious import compare_duplicates, duplicate_summary
    base = row(cf_ref="CASE 1", w_list=["CASE 2"] * 5, wo_list=["CASE 1"] * 5)        # egregious
    a = dict(base, source_file="reg.parquet")
    b = dict(base, source_file="itc.parquet",
             **{P + "with_explanation_predictor_answers": ["CASE 1"] + ["CASE 2"] * 4})  # 1 sim differs
    solo = dict(row(), source_file="reg.parquet")
    elig, _, scope = annotate(pd.DataFrame([a, b, solo]), "all", keep_duplicates=True)
    assert scope["duplicate_rows_kept"] == 2 and scope["duplicate_rows_removed"] == 0
    assert sorted(elig.ref_config) == ["m/a (medium)", "m/a (medium) [itc]", "m/a (medium) [reg]"]
    assert elig.pair_id.is_unique
    d = compare_duplicates(elig)
    assert len(d) == 1
    r = d.iloc[0]
    assert (r.file_a, r.file_b) == ("itc.parquet", "reg.parquet")
    assert (r.n_sims_differ_with, r.n_sims_differ_without, r.any_simulator_differs) == (1, 0, True)
    assert (r.egregious_a, r.egregious_b, r.egregious_differs) == (False, True, True)
    assert duplicate_summary(d)["m/a (medium)"]["egregious_differs"] == 1


def test_duplicates_removed_by_default_first_file_wins():
    base = row()
    elig, _, scope = annotate(pd.DataFrame([dict(base, source_file="x.parquet"),
                                            dict(base, source_file="y.parquet")]), "all")
    assert scope["duplicate_rows_removed"] == 1 and elig.source_file.tolist() == ["x.parquet"]


# --- independent edit flags -------------------------------------------------------------------
@pytest.mark.parametrize("o,c,expected", [
    # user's margin example: 4 vs 4 -> 5 vs 4
    (scen("4 men", "4 women"), scen("5 men", "4 women"),
     dict(law_changed=False, species_changed=False, margin_changed=True, size_changed=False,
          demographic_changed=False)),
    # user's size example: 4 vs 4 -> 5 vs 5
    (scen("4 men", "4 women"), scen("5 men", "5 women"),
     dict(law_changed=False, species_changed=False, margin_changed=False, size_changed=True,
          demographic_changed=False)),
    # a cat added to one side: species AND margin both true (neither blocks the other)
    (scen("1 man", "1 woman"), scen("1 man and 1 cat", "1 woman"),
     dict(law_changed=False, species_changed=True, margin_changed=True, size_changed=False,
          demographic_changed=False)),
    # a cat added to each side: species AND size
    (scen("1 man", "1 woman"), scen("1 man and 1 cat", "1 woman and 1 dog"),
     dict(law_changed=False, species_changed=True, margin_changed=False, size_changed=True,
          demographic_changed=False)),
    # sides swap 2 vs 1 -> 1 vs 2: the gap flips sign, so the trade-off reversed -> margin
    (scen("2 men", "1 woman"), scen("1 man", "2 women"),
     dict(law_changed=False, species_changed=False, margin_changed=True, size_changed=False,
          demographic_changed=False)),
    # same totals, types differ -> demographic only
    (scen("1 man", "1 woman"), scen("1 woman", "1 man"),
     dict(law_changed=False, species_changed=False, margin_changed=False, size_changed=False,
          demographic_changed=True)),
    # light flips and a person is added: law and margin both true
    (scen("1 man", "1 woman", "green", "red"), scen("2 men", "1 woman", "red", "green"),
     dict(law_changed=True, species_changed=False, margin_changed=True, size_changed=False,
          demographic_changed=False)),
])
def test_edit_flags(o, c, expected):
    from moral_machines.filter_egregious import edit_flags
    assert edit_flags(o, c) == expected


def test_flag_columns_in_output():
    from moral_machines.filter_egregious import EDIT_FLAGS
    elig, _, _ = run([row(o=scen("1 man", "1 woman"), c=scen("1 man and 1 cat", "1 woman"))])
    r = elig.iloc[0]
    assert all(f in elig.columns for f in EDIT_FLAGS)
    assert (r.species_changed, r.margin_changed, r.edit_type) == (True, True, "species_changed")
