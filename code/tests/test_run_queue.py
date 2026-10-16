from concurrent.futures import ThreadPoolExecutor

from coldrank.llm.run import claim, read_jobs


def test_claim_is_exclusive(tmp_path):
    job = {"method": "direct", "condition": "baseline", "neg": "random", "split": "dev", "limit": None}
    assert claim(str(tmp_path / "c"), job) is True
    assert claim(str(tmp_path / "c"), job) is False           # second worker skips it
    other = dict(job, limit=2)
    assert claim(str(tmp_path / "c"), other) is True          # a --limit run is a different task


def test_claim_race_one_winner(tmp_path):
    jobs = [{"method": m, "condition": c, "neg": "random", "split": "dev", "limit": None}
            for m in ("direct", "pointwise") for c in ("baseline", "a2_swap", "a3_hide")]

    def worker(_):
        return [j["method"] + j["condition"] for j in jobs if claim(str(tmp_path / "c"), j)]

    with ThreadPoolExecutor(8) as ex:
        won = [t for r in ex.map(worker, range(8)) for t in r]
    assert sorted(won) == sorted(j["method"] + j["condition"] for j in jobs)  # each task exactly once


def test_read_jobs(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("direct baseline random dev\n# comment\n\npointwise a1_oldgold trap test 5  # x\n")
    jobs = read_jobs(str(f))
    assert jobs == [
        {"method": "direct", "condition": "baseline", "neg": "random", "split": "dev", "limit": None},
        {"method": "pointwise", "condition": "a1_oldgold", "neg": "trap", "split": "test", "limit": 5},
    ]


class _GoldPicker:
    """Fake engine: in each prompt, answers with the ID of the gold item's line if present,
    otherwise [1]. Lets us check the sliding-window bookkeeping exactly."""

    def __init__(self, gold_title):
        self.gold_title = gold_title
        self.calls = 0

    def count_tokens(self, prompt):
        return len(prompt) // 4

    def generate(self, prompts, max_new):
        import re
        self.calls += len(prompts)
        out = []
        for p in prompts:
            cands = p.split("Candidate items")[1]
            m = re.search(r"\[(\d+)\] " + re.escape(self.gold_title) + r" \|", cands)
            out.append(f"FINAL: [{m.group(1) if m else 1}]")
        return out


def test_sliding_window_moves_gold_to_top():
    from coldrank.common import load_config
    from coldrank.llm.run import run_sliding, degeneracy
    from tests.test_prompts_eval import make_pool

    cfg = load_config("amazon_vg")
    cfg["prompt"]["id_style"] = "numbers"  # the fake engine answers with numbered IDs
    pools = [make_pool(pool_id=f"p{k}") for k in range(5)]
    eng = _GoldPicker("Cold game 3")
    raw, preds = run_sliding(eng, "chatml", pools, cfg, 100_000)
    assert eng.calls == 5 * 3 * 4                      # pools x shuffles x windows (30-50, 20-40, 10-30, 0-20)
    assert {r["window"][0] for r in raw} == {30, 20, 10, 0}
    assert all(p["ranking"][0] == "c3" for p in preds)  # the gold always reaches the top
    d = degeneracy(raw)
    assert 0 <= d["identity_rate"] <= 1 and d["top1_mode_share"] <= 1
