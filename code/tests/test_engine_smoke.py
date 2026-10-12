"""End-to-end LLM path with a tiny real model on CPU (HF backend).

Downloads HuggingFaceTB/SmolLM2-135M-Instruct (~270 MB) on first run.
Skip with: pytest -m "not slow"
"""

import math

import pytest

from coldrank.common import load_config

pytestmark = pytest.mark.slow
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from coldrank.llm.engine import make_engine

    mcfg = load_config("models")["smol_135m"]
    try:
        return make_engine(mcfg, "hf", tmp_path_factory.mktemp("cache"), "smol_135m", device="cpu", batch_size=4)
    except OSError as e:  # offline
        pytest.skip(f"model not available: {e}")


def test_yes_no_matches_manual(engine):
    from coldrank.llm.chat import format_chat

    p = format_chat("chatml", "S", "Is the sky blue? Answer Yes or No.")
    got = engine.yes_no([p])[0]
    hf = engine.engine
    ids = hf.tok(p, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(hf.model(**ids).logits[0, -1].float(), -1)
    lse = lambda xs: math.log(sum(math.exp(lp[i].item()) for i in xs))  # noqa: E731
    assert abs(got - (lse(hf.yes_ids) - lse(hf.no_ids))) < 1e-4
    # batching with left padding must not change the score
    longer = format_chat("chatml", "S", "Is the sky blue on a clear day at noon? Answer Yes or No.")
    a = engine.engine.yes_no([p, longer])
    assert abs(a[0] - got) < 1e-3


def test_cache_roundtrip(engine):
    from coldrank.llm.chat import format_chat
    from coldrank.llm.engine import CachedEngine

    p = format_chat("chatml", "S", "Say hi.")
    a = engine.generate([p], [8])
    again = CachedEngine(engine.engine, engine.path.parent, "smol_135m")  # reload from disk
    calls = []
    orig = engine.engine.generate
    engine.engine.generate = lambda *x: calls.append(1) or orig(*x)
    try:
        assert again.generate([p], [8]) == a and calls == []
    finally:
        engine.engine.generate = orig


def test_runners_end_to_end(engine):
    from coldrank.llm.run import run_listwise, run_pointwise
    from tests.test_prompts_eval import make_pool

    cfg = load_config("amazon_vg")
    cfg["prompt"]["shuffles"] = 2
    pools = [make_pool(pool_id="pA"), make_pool(pool_id="pB")]
    raw, preds = run_listwise(engine, "chatml", pools, "direct", cfg, 4096)
    assert len(raw) == 4 and len(preds) == 2
    ids = {c["item_id"] for c in pools[0]["candidates"]}
    for r in raw:
        assert r["output"].startswith("FINAL: [") and r["prompt_tokens"] > 500
        assert set(r["ranking"]) <= ids
    for p in preds:
        assert set(p["ranking"]) <= ids and len(p["ranking"]) == len(set(p["ranking"]))
    raw, preds = run_pointwise(engine, "chatml", pools[:1], cfg, 4096)
    assert sorted(preds[0]["ranking"]) == sorted(ids)
    assert all(math.isfinite(v) for v in raw[0]["scores"].values())
