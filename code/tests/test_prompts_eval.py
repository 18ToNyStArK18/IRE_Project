import re

import numpy as np
import pytest

from coldrank.baselines.heuristics import HEURISTICS, newest_set
from coldrank.eval.metrics import borda, pool_metrics, summarize
from coldrank.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff
from coldrank.llm.chat import format_chat
from coldrank.prompts.examples import EXAMPLE_1, EXAMPLE_2, NEWS_EXAMPLE_1, NEWS_EXAMPLE_2
from coldrank.prompts.render import (
    GEN_MIN, METHODS, parse_ranking, render_listwise, render_pointwise, shuffled_candidates,
)


def make_pool(n_warm=40, n_cold=10, n_hist=30, pool_id="p0"):
    cands = []
    for k in range(n_warm):
        cands.append({"item_id": f"w{k:02d}", "title": f"Warm game {k}", "date": f"2015-01-{k % 28 + 1:02d}",
                      "t0": 0, "category": "PC > Games", "brand": "B", "is_cold": False, "is_gold": False,
                      "sim_to_history": k / 100, "pop_preT": 100 - k, "warm_rank": k + 1})
    for k in range(n_cold):
        cands.append({"item_id": f"c{k}", "title": f"Cold game {k}", "date": f"2023-03-{k + 1:02d}",
                      "t0": 1, "category": "PS5 > Games", "brand": "B", "is_cold": True, "is_gold": k == 3,
                      "sim_to_history": 0.9 if k == 3 else 0.1, "pop_preT": 0, "warm_rank": None})
    hist = [{"item_id": f"h{k}", "title": f"Hist {k}", "date": "2014-01-01", "category": "PC > Games",
             "brand": "B", "ts": k} for k in range(n_hist)]
    return {"pool_id": pool_id, "gold_item_id": "c3", "candidates": sorted(cands, key=lambda c: c["item_id"]),
            "history": hist}


# ---------------- examples ---------------- #
@pytest.mark.parametrize("ex", [EXAMPLE_1, EXAMPLE_2, NEWS_EXAMPLE_1, NEWS_EXAMPLE_2])
def test_examples_gold_is_not_newest(ex):
    gold = ex["candidates"][ex["base_final"][0] - 1]
    newest = max(ex["candidates"], key=lambda c: c["date"])
    assert gold is not newest
    for key in ("base", "fast", "structural"):
        if key in ex:
            assert not re.search(r"\b(dates?|recent|recently|new|newer|newest|old|older|released?)\b", ex[key].lower())
            assert sorted(ex[f"{key}_final"]) == list(range(1, len(ex["candidates"]) + 1))


# ---------------- rendering ---------------- #
@pytest.mark.parametrize("method", METHODS)
def test_render_listwise_all_methods(method):
    pool = make_pool()
    r = render_listwise(pool, method, 0, 20, 10)
    assert sorted(r.id_map) == list(range(1, 51))
    assert set(r.id_map.values()) == {c["item_id"] for c in pool["candidates"]}
    if method in ("direct_v2", "direct_sw"):
        assert "[50]" in r.user and "FINAL: [12] > [3]" in r.user and "do not simply copy" in r.user
        assert r.prefill == ""
    else:
        assert "[50]" in r.user and "FINAL: [id]" in r.user
    assert r.user.count("Hist ") == 20  # last 20 history items
    if method == "direct":
        assert r.prefill == "FINAL: ["
    if method == "structural":
        assert "how recent" in r.user and "(each item: title | launch date | category | brand)" in r.user
    if method == "structural_norecency":
        assert "how recent" not in r.user and "Step 3: Give each path an importance weight, considering how prominent the factor is in the history." in r.user
    if method in ("base", "fast"):
        assert ("Example 2:" in r.user) == (method == "base")


def test_shuffles_differ_and_are_deterministic():
    pool = make_pool()
    a = [c["item_id"] for c in shuffled_candidates(pool, 0)]
    b = [c["item_id"] for c in shuffled_candidates(pool, 1)]
    assert a != b and a == [c["item_id"] for c in shuffled_candidates(pool, 0)]


def test_hidden_dates_are_omitted():
    pool = make_pool()
    for c in pool["candidates"]:
        c["date"] = None
    r = render_listwise(pool, "direct", 0, 20, 10)
    cand_part = r.user.split("Candidate items")[1]
    assert "2023-03" not in cand_part and "launch date" not in cand_part
    assert "(each item: title | category)" in cand_part


@pytest.mark.parametrize("method", ["base", "fast", "structural", "structural_norecency"])
def test_hidden_dates_also_hidden_in_examples(method):
    # A3 hides every date of the task; the worked examples must not show theirs (dates of EXAMPLE_1/2)
    pool = make_pool()
    shown = render_listwise(pool, method, 0, 20, 10, id_style="codes").user
    assert "2017-03-03" in shown and "launch date" in shown
    for it in pool["candidates"] + pool["history"]:
        it["date"] = None
    hidden = render_listwise(pool, method, 0, 20, 10, id_style="codes").user
    assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d\b", hidden) and "launch date" not in hidden


def test_history_trimmed_to_fit():
    pool = make_pool(n_hist=40)
    count = lambda s, u, pf: len((s + u + pf).split()) * 2  # noqa: E731  (fake tokenizer)
    full = render_listwise(pool, "base", 0, 40, 10, count, 100_000)
    budget = full.prompt_tokens + GEN_MIN["base"] - 50  # force trimming
    r = render_listwise(pool, "base", 0, 40, 10, count, budget)
    assert r.n_history < 40 and (budget - r.prompt_tokens >= GEN_MIN["base"] or r.n_history == 5)
    assert r.user.count("Hist ") == r.n_history
    # newest history items are the ones kept
    assert "Hist 39" in r.user and "Hist 0 " not in r.user


def test_pointwise_prompt():
    pool = make_pool()
    s, u = render_pointwise(pool, pool["candidates"][0], 20)
    assert u.endswith("Answer Yes or No.") and "[" not in u.split("Candidate item")[1]


def test_chat_formats():
    assert format_chat("llama2", "S", "U").startswith("[INST] <<SYS>>\nS\n<</SYS>>\n\nU [/INST]")
    assert "<s>" not in format_chat("llama2", "S", "U") and "<s>" not in format_chat("mistral", "S", "U")
    assert format_chat("mistral", "S", "U", "FINAL: [") == "[INST] S\n\nU [/INST] FINAL: ["
    assert format_chat("chatml", "S", "U").endswith("<|im_start|>assistant\n")
    assert format_chat("phi2", "S", "U").endswith("Output:")


# ---------------- parsing ---------------- #
def test_parse_ranking():
    assert parse_ranking("blah\nFINAL: [3] > [1] > [3] > [99] > [2]", 50) == ([3, 1, 2], "final")
    assert parse_ranking("FINAL: [id] > [id]\n...\nFINAL: [7] > [8]", 50) == ([7, 8], "final")  # last FINAL
    assert parse_ranking("final:[ 4 ]>[5]", 50) == ([4, 5], "final")
    assert parse_ranking("My ranking:\n[4] > [9] > [1]\nThanks", 50) == ([4, 9, 1], "fallback_line")
    assert parse_ranking("I recommend item 4.", 50) == ([], "fail")
    assert parse_ranking("FINAL: [12] > [3]\n\n[1] [2] [3]", 50) == ([12, 3], "final")


# ---------------- metrics ---------------- #
def test_borda():
    assert borda("p", [["a", "b"], ["b", "a"], ["b"]], 10) == ["b", "a"]
    assert borda("p", [[], []], 10) == []
    r = borda("p", [["a"], ["b"]], 10)  # tie broken by hash, deterministic
    assert set(r) == {"a", "b"} and r == borda("p", [["b"], ["a"]], 10)


def test_pool_metrics():
    pool = make_pool()
    m = pool_metrics(pool, ["c3", "w00"])
    assert m["recall1"] == 1 and m["ndcg10"] == 1 and m["cold1"] is True
    m = pool_metrics(pool, ["w00", "c1", "c3"])
    assert m["recall1"] == 0 and abs(m["ndcg10"] - 0.5) < 1e-9 and m["cold1"] is False
    m = pool_metrics(pool, [])
    assert m["recall1"] == 0 and m["empty"] and m["cold1"] is None
    s = summarize([pool_metrics(pool, ["c3"]), pool_metrics(pool, [])])
    assert s["recall1"] == 0.5 and s["cold1"] == 1.0 and s["empty_rate"] == 0.5


# ---------------- heuristics ---------------- #
def test_heuristics_full_rankings_and_expectations():
    pools = [make_pool(pool_id=f"p{k}") for k in range(400)]
    for name, fn in HEURISTICS.items():
        r = fn(pools[0])
        assert sorted(r) == sorted(c["item_id"] for c in pools[0]["candidates"]), name
    assert newest_set(pools[0]) == {f"c{k}" for k in range(10)}
    r1 = lambda name: np.mean([HEURISTICS[name](p)[0] == "c3" for p in pools])  # noqa: E731
    assert abs(r1("random") - 0.02) < 0.02
    assert abs(r1("random_among_newest") - 0.10) < 0.04
    assert r1("newest_first") == 0.0          # c9 is newest, gold c3 is not
    assert r1("newest_filter_content") == 1.0  # gold has the highest similarity among cold
    assert r1("sasrec") == 0.0 and HEURISTICS["sasrec"](pools[0])[0] == "w00"
    assert HEURISTICS["popularity"](pools[0])[0] == "w00"


# ---------------- stats ---------------- #
def test_stats():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1], n_boot=2000)
    assert lo <= m <= hi
    d = paired_bootstrap_diff([1, 1, 1, 0], [0, 1, 0, 0], n_boot=2000)
    assert d["diff"] == 0.5
    t = mcnemar_exact([1] * 20 + [0] * 5, [0] * 20 + [0] * 5)
    assert t["a_only"] == 20 and t["p"] < 1e-4


def test_display_title_and_category():
    from coldrank.prompts.render import display_category, display_title
    long = "Logitech G FITS True Wireless Gaming Earbuds, Custom Molded Fit, Lightspeed + Bluetooth, Four Mics"
    t = display_title(long)
    assert len(t) <= 60 and long.startswith(t) and not t.endswith(",")
    assert display_title("Short title") == "Short title"
    assert display_category("Legacy Systems > Nintendo Systems > Wii > Games") == "Wii > Games"
    assert display_category("PlayStation 4 > Accessories > Accessory Kits > Extra") == "PlayStation 4 > Accessories > Accessory Kits"
    assert display_category("") == ""


# ---------------- letter codes (prompt.id_style: codes) ---------------- #
@pytest.mark.parametrize("method", ["direct", "base", "fast", "structural", "structural_norecency", "ssc"])
def test_codes_render_and_parse(method):
    from coldrank.prompts.render import CODE_LETTERS, EXAMPLE_CODES
    pool = make_pool()
    r = render_listwise(pool, method, 0, 20, 10, id_style="codes")
    codes = list(r.code_map)
    assert len(codes) == 50 == len(set(codes)) and sorted(r.code_map.values()) == list(range(1, 51))
    assert all(len(c) == 2 and set(c) <= set(CODE_LETTERS) for c in codes)         # consonant pairs only
    assert r == render_listwise(pool, method, 0, 20, 10, id_style="codes")           # deterministic
    task = r.user.split("Now the actual task.")[-1]
    assert "[1]" not in task and "[50]" not in task                                  # no numbered IDs
    assert "FINAL: [code]" in r.user and r.prefill == ""
    if method in ("base", "fast", "structural", "structural_norecency"):
        demo = r.user.split("Now the actual task.")[0]
        assert "[AE]" in demo and not re.search(r"\[\d\]", demo)                     # examples use vowel codes
    a, b = codes[4], codes[0]
    assert parse_ranking(f"blah\nFINAL: [{a}] > [{b.lower()}] > [{a}]", 50, r.code_map) == ([5, 1], "final")
    # copying the worked example's answer (vowel codes) must not parse as a ranking
    assert parse_ranking("FINAL: [" + "] > [".join(EXAMPLE_CODES) + "]", 50, r.code_map) == ([], "fail")
