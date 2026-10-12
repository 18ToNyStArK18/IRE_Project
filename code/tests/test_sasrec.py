import pandas as pd
import torch

from coldrank.warm.sasrec import SASRec, build_sequences, evaluate, kcore, left_pad, topk_for_instances, train

import numpy as np


def _model(n_items=20, L=8):
    torch.manual_seed(0)
    m = SASRec(n_items, L, hidden=16, n_layers=2, n_heads=2, dropout=0.0)
    m.eval()
    return m


def test_no_nan_with_heavy_padding():
    m = _model()
    seq = torch.tensor([left_pad([3], 8), left_pad([1, 2, 3, 4, 5, 6, 7, 8], 8), left_pad([], 8)])
    h = m(seq)
    assert torch.isfinite(h).all()
    m.train()
    assert torch.isfinite(m(seq)).all()


def test_causal_and_padding_invariance():
    m = _model()
    a = torch.tensor([left_pad([1, 2, 3, 4], 8)])
    b = torch.tensor([left_pad([1, 2, 3, 9], 8)])  # only the last item differs
    ha, hb = m(a), m(b)
    assert torch.allclose(ha[0, :-1], hb[0, :-1], atol=1e-6)  # earlier positions unaffected
    assert not torch.allclose(ha[0, -1], hb[0, -1])
    # padding content must not matter: padded positions are never attended to by real ones
    m2 = _model()
    with torch.no_grad():
        m2.item_emb.weight[0] = 5.0  # corrupt padding embedding
    assert torch.allclose(m(a)[0, -4:], m2(a)[0, -4:], atol=1e-5)


def test_kcore():
    df = pd.DataFrame({"user_id": ["a"] * 3 + ["b"] * 3 + ["c"], "item_id": ["x", "y", "z", "x", "y", "z", "x"], "ts": range(7)})
    out = kcore(df, 2)
    assert set(out["user_id"]) == {"a", "b", "c"} - {"c"}  # c has 1 interaction
    assert set(out["item_id"]) == {"x", "y", "z"}


def test_learns_successor_pattern():
    # every user walks i -> i+1 (mod 30); the model must learn "next = current + 1"
    rows = []
    rng = np.random.default_rng(0)
    for u in range(300):
        start = int(rng.integers(30))
        for k in range(8):
            rows.append((f"u{u}", f"it{(start + k) % 30:02d}", k))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])
    vocab, seqs = build_sequences(df)
    hp = dict(max_seq_len=10, hidden=32, n_layers=2, n_heads=2, dropout=0.0, lr=0.01,
              batch_size=64, max_epochs=40, patience=40, seed=0)
    model, curve = train(seqs, len(vocab), hp, "cpu")
    assert curve[-1]["val_recall10"] > 0.95
    # top-k excludes history and returns warm items
    inst = [{"instance_id": "x", "history": [["it03", 0], ["it04", 1], ["it05", 2]]}]
    pop = df.groupby("item_id").size().sort_values(ascending=False)
    out = topk_for_instances(model, vocab, inst, 5, 10, "cpu", pop)[0]
    assert out["items"][0] == "it06"
    assert not {"it03", "it04", "it05"} & set(out["items"])
    # unknown history -> popularity fallback, flagged
    out2 = topk_for_instances(model, vocab, [{"instance_id": "y", "history": [["zzz", 0]]}], 5, 10, "cpu", pop)[0]
    assert out2["source"] == "popularity_fallback" and len(out2["items"]) == 5


def test_trim_padding_is_exact():
    from coldrank.warm.sasrec import trim_padding
    m = _model(L=8)
    full = np.array([left_pad([1, 2, 3], 8), left_pad([4, 5], 8)])
    (trimmed,) = trim_padding(full)
    assert trimmed.shape[1] == 3
    a = m(torch.as_tensor(full))[:, -3:]
    b = m(torch.as_tensor(trimmed))
    real = torch.as_tensor(trimmed != 0)
    assert torch.allclose(a[real], b[real], atol=1e-5)
    assert torch.allclose(m.scores(torch.as_tensor(full)), m.scores(torch.as_tensor(trimmed)), atol=1e-4)
