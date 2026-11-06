"""Offline tests for writer_pilot.py (no API calls):  pytest v2/discovery/test_writer_pilot.py -q"""
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import writer_pilot as wp  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return wp.load_raw()


@pytest.fixture(scope="module")
def pools(raw):
    return {m: wp.split(wp.exclude_fresh_test(wp.load_originals(m, raw))) for m in wp.MODELS}


@pytest.fixture(scope="module")
def conds(pools, raw):
    return wp.conditions(pools, raw)


# --- the explanation never reaches the writer ---------------------------------------------------------------------
def test_loader_reads_no_response_text():
    assert not [c for c in wp.LOAD_COLS if re.search(r"raw_response|parsed|cot|explanation|predictor", c)]


def test_writer_prompts_contain_no_explanation_text(pools):
    """Take each model's real explanations and check no 60-char piece of one appears in any writer prompt."""
    for name, df in pools.items():
        path, model = wp.MODELS[name]
        d = pd.read_parquet(path, columns=["original_dataset", "original_question", "original_reference_response_raw_response",
                                           "original_reference_response_model_info_model",
                                           "original_reference_response_model_info_thinking"])
        d = d[(d.original_dataset == "moral_machines") & (d.original_reference_response_model_info_model == model)
              & (d.original_reference_response_model_info_thinking == "high")].drop_duplicates("original_question")
        expl = {q: r for q, r in zip(d.original_question, d.original_reference_response_raw_response)}
        tr = df[df.split == "train"]
        prompt = "".join(wp.writer_prompt(ch) for ch in wp.chunks(tr, 125, seed=1000))
        for q in tr.scenario.sample(60, random_state=0):
            text = re.sub(r"\[ANSWER\].*?\n\n", "", str(expl[q]), flags=re.S)
            pieces = [text[i:i + 60] for i in range(0, max(1, len(text) - 60), 150) if len(text[i:i + 60].strip()) > 40]
            assert pieces and not any(p in prompt for p in pieces), name


def test_label_prompt_has_no_answers(conds):
    te = conds["sonnet45_high"][conds["sonnet45_high"].split == "test"].head(5)
    assert "Answer:" not in wp.render_items(te, with_answer=False) and "Answer: CASE" in wp.render_items(te, True)


def test_writer_instructions_do_not_cue_a_factor():
    instructions = wp.WRITER_PROMPT.split("Scenarios:")[0].lower()
    for cue in ["position", "label", "order", "case 1", "case 2", "first", "before", "after", "gender", "passenger",
                "women", "law", "swerve", "number"]:
        assert cue not in instructions, cue
    assert "presented and" in instructions  # open to presentation-level regularities, without naming one


# --- split and conditions -----------------------------------------------------------------------------------------
def test_split_stratified_and_disjoint(pools):
    for df in pools.values():
        t = pd.crosstab(df.theme, df.split)
        assert (t > 0).all().all() and t.index.size == 6
        share = t.test / t.sum(1)
        assert share.between(0.2, 0.3).all()
        assert df.item.is_unique and df.scenario.is_unique


def test_fresh_test_split_excluded(pools):
    f = pd.read_parquet(wp.FRESH)
    held = set(f[(f.split == "test") & ~f.mirrored].question)
    assert held and all(not (set(df.scenario) & held) for df in pools.values())


def test_only_high_thinking_originals(raw):
    df = wp.load_originals("gpt52_high", raw)
    assert len(df) <= 891 and df.answer.isin(["CASE 1", "CASE 2"]).all()


def test_shuffled_keeps_base_rate_and_changes_answers(conds):
    for m in wp.MODELS:
        real, sh = conds[m], conds[f"{m}_shuffled"]
        for s in ("train", "test"):
            a, b = real[real.split == s].answer, sh[sh.split == s].answer
            assert a.value_counts().equals(b.value_counts().reindex(a.value_counts().index))
        assert (real.answer != sh.answer).mean() > 0.2
        assert real.drop(columns="answer").equals(sh.drop(columns="answer"))


def test_planted_overlay(conds, raw):
    real, p = conds["sonnet45_high"], conds["planted"]
    rule = np.array([wp.planted_rule(q, raw) for q in p.scenario])
    applies, changed = rule != 0, (p.answer.to_numpy() != real.answer.to_numpy())
    assert not changed[~applies].any()                               # only scenarios the rule applies to change
    assert 0.5 < applies.mean() < 0.8 and 0.08 < changed.mean() < 0.25
    agree = (p.answer.str[-1].astype(int).to_numpy() == rule)[applies].mean()
    assert agree > (real.answer.str[-1].astype(int).to_numpy() == rule)[applies].mean() + 0.1
    assert p.drop(columns="answer").equals(real.drop(columns="answer"))


def test_planted_rule_definition(raw):
    from collections import Counter
    q = "x"
    raw2 = {q: dict(c1=Counter({"woman": 2, "man": 1}), c2=Counter({"girl": 1, "dog": 3}))}
    assert wp.planted_rule(q, raw2) == 2                              # Case 2 kills fewer women/girls
    raw2[q]["c2"] = Counter({"girl": 2})
    assert wp.planted_rule(q, raw2) == 0


def test_chunks_cover_train_once(pools):
    tr = pools["sonnet45_high"][pools["sonnet45_high"].split == "train"]
    ch = wp.chunks(tr, 125, seed=3)
    assert sum(map(len, ch)) == len(tr) and pd.concat(ch).item.is_unique and max(map(len, ch)) <= 126
    assert [c.item.tolist() for c in wp.chunks(tr, 125, seed=3)] == [c.item.tolist() for c in ch]
    assert [c.item.tolist() for c in wp.chunks(tr, 125, seed=4)] != [c.item.tolist() for c in ch]


def test_known_factors(conds, raw):
    te = conds["sonnet45_high"][conds["sonnet45_high"].split == "test"]
    K = wp.known_factors(te, raw)
    assert len(K) == len(te) and not K.isna().any().any() and K.answer_first.equals(te.answer_first.astype(float))


# --- parsing, validation, merging ---------------------------------------------------------------------------------
@pytest.mark.parametrize("text,ok", [('{"a": 1}', True), ('```json\n{"a": 1}\n```', True), ("here: {\"a\": 1} done", True),
                                     ("no json", False), ('{"a": }', False), ("", False), (None, False)])
def test_parse_json(text, ok):
    assert (wp.parse_json(text) is not None) == ok


def test_validators():
    assert wp.valid_write({"hypotheses": [{"rule": "When X, it picks Y"}]})
    assert not wp.valid_write({"hypotheses": []}) and not wp.valid_write({"hypotheses": [{"rule": " "}]})
    assert wp.valid_merge({"groups": [{"rule": "r", "members": ["H1", "H2"]}]}, ["H2", "H1"])
    assert not wp.valid_merge({"groups": [{"rule": "r", "members": ["H1"]}]}, ["H1", "H2"])          # dropped id
    assert not wp.valid_merge({"groups": [{"rule": "r", "members": ["H1", "H1", "H2"]}]}, ["H1", "H2"])  # duplicate
    assert wp.valid_label({"labels": {"S1": [1, 0], "S2": [2, 2]}}, ["S1", "S2"], 2)
    assert not wp.valid_label({"labels": {"S1": [1, 3]}}, ["S1"], 2)
    assert not wp.valid_label({"labels": {"S1": [1]}}, ["S1"], 2)
    assert not wp.valid_label({"labels": {"S1": [1, 0]}}, ["S1", "S2"], 2)


def test_group_stats_counts_runs_and_chunks():
    raw_h = pd.DataFrame(dict(hid=["H0", "H1", "H2", "H3"], run=[0, 0, 1, 2], chunk=[0, 1, 0, 0], rule=list("abcd")))
    g = wp.group_stats("c", raw_h, [{"rule": "A", "members": ["H0", "H1", "H2"]}, {"rule": "B", "members": ["H3"]}], 2, 3)
    a, b = g.set_index("rule").loc["A"], g.set_index("rule").loc["B"]
    assert (a.n_runs, a.n_chunks, a.kept) == (2, 3, True) and (b.n_runs, b.kept) == (1, False)


# --- regression test ----------------------------------------------------------------------------------------------
def test_rule_tests_detects_real_and_rejects_noise():
    rng = np.random.default_rng(0)
    n = 200
    K = rng.normal(size=(n, 3))
    h = rng.choice([1, -1, 0], size=n)
    y = np.where(h != 0, (h > 0).astype(float), rng.integers(0, 2, n))
    flip = rng.random(n) < 0.15
    y = np.where(flip, 1 - y, y)
    real = wp.rule_tests(y, h, K)
    noise = wp.rule_tests(y, rng.choice([1, -1, 0], size=n), K)
    assert real["p_beyond_known"] < 1e-6 and real["acc_applicable"] > 0.75
    assert noise["p_beyond_known"] > 0.01
    assert np.isnan(wp.rule_tests(y, np.zeros(n), K)["p_alone"])


def test_rule_redundant_with_known_factor_does_not_pass_beyond():
    rng = np.random.default_rng(1)
    k = rng.choice([1.0, -1.0], size=300)
    y = (rng.random(300) < np.where(k > 0, 0.85, 0.15)).astype(float)
    r = wp.rule_tests(y, k, k[:, None])
    assert r["p_alone"] < 1e-6 and r["p_beyond_known"] > 0.05


# --- API wrapper (fake client) ------------------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, content, status=200):
        self.content_, self.status = content, status

    def raise_for_status(self):
        if self.status != 200:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return {"choices": [{"message": {"content": self.content_}}], "usage": {"cost": 0.01}}


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def post(self, url, timeout, json):
        self.calls.append(json)
        return self.replies.pop(0)


def test_caller_retries_then_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(wp.asyncio, "sleep", lambda s, _real=asyncio.sleep: _real(0))
    good = '{"hypotheses": [{"rule": "When X, it picks Y"}]}'
    client = FakeClient([FakeResponse("", 500), FakeResponse("not json"), FakeResponse(good)])
    caller = wp.Caller(tmp_path / "calls.jsonl", client)
    out = asyncio.run(caller("k1", "prompt", "write", wp.valid_write))
    assert out["hypotheses"][0]["rule"] == "When X, it picks Y" and len(client.calls) == 3
    assert client.calls[0]["reasoning"] == {"effort": "high"} and client.calls[0]["model"] == wp.WRITER
    again = wp.Caller(tmp_path / "calls.jsonl", FakeClient([]))                  # cache hit: no call made
    assert asyncio.run(again("k1", "prompt", "write", wp.valid_write)) == out


def test_caller_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(wp.asyncio, "sleep", lambda s, _real=asyncio.sleep: _real(0))
    caller = wp.Caller(tmp_path / "c.jsonl", FakeClient([FakeResponse("bad")] * 4))
    with pytest.raises(RuntimeError, match="failed after 4"):
        asyncio.run(caller("k", "p", "write", wp.valid_write))
    assert not (tmp_path / "c.jsonl").exists()


def test_stage_test_end_to_end_offline(conds, raw):
    """Labelled with the planted rule, the planted condition must pass beyond the known factors; the same rule on
    shuffled answers must not pass at all."""
    c = {"planted": conds["planted"], "sonnet45_high_shuffled": conds["sonnet45_high_shuffled"]}
    merged = pd.DataFrame(dict(condition=list(c), gid=["p:G00", "sh:G00"], rule=["fewer female deaths"] * 2,
                               n_runs=[3, 3], n_chunks=[15, 15], kept=[True, True]))
    rows = [dict(condition=name, item=r.item, gid=g, label=wp.planted_rule(r.scenario, raw))
            for (name, df), g in zip(c.items(), ["p:G00", "sh:G00"]) for r in df[df.split == "test"].itertuples()]
    res = wp.stage_test(c, merged, pd.DataFrame(rows), raw).set_index("gid")
    assert res.loc["p:G00"].passes_beyond_known and res.loc["p:G00"].p_beyond_known < 1e-5
    assert not res.loc["sh:G00"].passes_alone


def test_stage_test_requires_all_labels(conds, raw):
    c = {"planted": conds["planted"]}
    te = c["planted"][c["planted"].split == "test"]
    merged = pd.DataFrame(dict(condition=["planted"], gid=["p:G00"], rule=["r"], n_runs=[3], n_chunks=[9], kept=[True]))
    part = pd.DataFrame(dict(condition="planted", item=te.item.head(30), gid="p:G00", label=1))
    with pytest.raises(ValueError, match="labels missing"):
        wp.stage_test(c, merged, part, raw)
    assert wp.stage_test(c, merged, part, raw, require_all=False).n_test.iloc[0] == 30


def test_merge_keeps_only_recurring_rules():
    raw_h = pd.DataFrame(dict(hid=["H0", "H1", "H2"], run=[0, 1, 1], chunk=[0, 0, 1], rule=list("abc")))
    g = wp.group_stats("c", raw_h, [{"rule": "A", "members": ["H0", "H1"]}, {"rule": "B", "members": ["H2"]}], 2, 3)
    assert not g.kept.any()                                          # A recurs in 2 runs but only 2 chunks


def test_bad_inputs():
    with pytest.raises(ValueError):
        wp.split(pd.DataFrame(dict(theme=["a"], scenario=["x"])), test_frac=1)
    with pytest.raises(KeyError):
        wp.load_originals("sonnet45_high", {})
