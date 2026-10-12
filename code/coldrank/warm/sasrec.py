"""Warm-start ranker: SASRec (Kang & McAuley 2018), our stand-in for Netflix's
production ranker. Hyperparameters follow Dong et al. (2026)'s config.

Training data: positives with ts < T, iterative k-core. Validation: each user's last
training item (leave-one-out), early stopping on NDCG@10. Loss: full softmax
cross-entropy over the item vocabulary (all items are scored, no sampled negatives).

Outputs (in the split directory):
  sasrec.pt                  model + vocab
  sasrec_topk.jsonl          per instance: top-N warm items (excluding H_u) with scores
  sasrec_report.json         validation curve + sanity metrics vs popularity

Usage:
    python -m coldrank.warm.sasrec --config amazon_vg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from coldrank.common import date_to_ts, load_config, read_jsonl, seed_everything, write_jsonl
from coldrank.data.split import split_dir


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions."""
    while True:
        uc = df.groupby("user_id").size()
        ic = df.groupby("item_id").size()
        bad_u, bad_i = uc[uc < k].index, ic[ic < k].index
        if len(bad_u) == 0 and len(bad_i) == 0:
            return df
        df = df[~df["user_id"].isin(bad_u) & ~df["item_id"].isin(bad_i)]


def build_sequences(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, list[int]]]:
    """item vocab (1-based; 0 = padding) and per-user chronological index sequences."""
    items = sorted(df["item_id"].unique())
    vocab = {it: k + 1 for k, it in enumerate(items)}
    df = df.sort_values(["user_id", "ts", "item_id"], kind="mergesort")
    seqs = {u: [vocab[i] for i in g["item_id"]] for u, g in df.groupby("user_id", sort=False)}
    return vocab, seqs


def left_pad(seq: list[int], n: int) -> list[int]:
    seq = seq[-n:]
    return [0] * (n - len(seq)) + seq


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int, hidden: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.max_len = max_len
        self.n_heads = n_heads
        causal = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)
        self.register_buffer("eye", torch.eye(max_len, dtype=torch.bool), persistent=False)
        # Small init (BERT-style). nn.Embedding's default N(0, 1) makes the initial
        # output logits (hidden @ item_emb.T) huge, so the softmax starts saturated.
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def attn_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """(B * n_heads, L, L) boolean mask, True = blocked.

        Blocks future keys (causal) and padding keys, but always allows a position
        to attend to itself. Without the diagonal, a left-padded position has no
        visible key, softmax returns NaN, and the NaN leaks into real positions
        through the value matmul (0 * NaN = NaN).
        """
        L = seq.size(1)
        pad = seq == 0
        m = self.causal[:L, :L].unsqueeze(0) | (pad.unsqueeze(1) & ~self.eye[:L, :L].unsqueeze(0))
        return m.repeat_interleave(self.n_heads, dim=0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) left-padded item ids, L <= max_len -> hidden states (B, L, H).

        Positions are counted from the right edge (the last item is always position
        max_len - 1), so dropping leading all-padding columns (trim_padding) gives
        exactly the same outputs for the real items."""
        L = seq.size(1)
        pos = torch.arange(self.max_len - L, self.max_len, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) * (self.item_emb.embedding_dim ** 0.5) + self.pos_emb(pos)
        x = self.drop(x)
        h = self.encoder(x, mask=self.attn_mask(seq))
        return self.norm(h)

    def scores(self, seq: torch.Tensor) -> torch.Tensor:
        """Scores over all items (B, n_items + 1) from the last position."""
        h = self.forward(seq)[:, -1, :]
        return h @ self.item_emb.weight.T


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def _batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for s in range(0, n, bs):
        yield idx[s : s + bs]


def _bucketed_batches(lengths: np.ndarray, bs: int, rng: np.random.Generator):
    """Batches of users with similar sequence length (so little padding), in random
    batch order; users are shuffled before the stable sort so ties vary per epoch."""
    perm = rng.permutation(len(lengths))
    order = perm[np.argsort(lengths[perm], kind="stable")]
    batches = [order[s : s + bs] for s in range(0, len(order), bs)]
    for k in rng.permutation(len(batches)):
        yield batches[k]


def trim_padding(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop leading columns that are padding in every row of the first array."""
    nonpad = (arrays[0] != 0).any(0)
    start = int(np.argmax(nonpad)) if nonpad.any() else arrays[0].shape[1] - 1
    return tuple(a[:, start:] for a in arrays)


@torch.no_grad()
def evaluate(model: SASRec, inputs: np.ndarray, targets: np.ndarray, device: str, k: int = 10, bs: int = 1024):
    """Leave-one-out NDCG@k / Recall@k with the input items masked out."""
    model.eval()
    ndcg = rec = 0.0
    for b in _batches(len(inputs), bs, None):
        seq = torch.as_tensor(trim_padding(inputs[b])[0], device=device)
        s = model.scores(seq)
        s[:, 0] = -torch.inf
        s.scatter_(1, seq, -torch.inf)  # mask seen items (and padding id 0 again)
        tgt = torch.as_tensor(targets[b], device=device)
        tgt_s = s.gather(1, tgt[:, None])
        rank = (s > tgt_s).sum(1)  # 0-based
        hit = rank < k
        rec += hit.sum().item()
        ndcg += (hit.float() / torch.log2(rank.float() + 2)).sum().item()
    return ndcg / len(inputs), rec / len(inputs)


def train(seqs: dict[str, list[int]], n_items: int, hp: dict, device: str) -> tuple[SASRec, list[dict]]:
    L = hp["max_seq_len"]
    users = sorted(seqs)
    # train on seq[:-1] (all but last), validate on last
    tr_in, tr_out, va_in, va_tgt = [], [], [], []
    for u in users:
        s = seqs[u]
        tr = s[:-1]
        tr_in.append(left_pad(tr[:-1], L))
        tr_out.append(left_pad(tr[1:], L))
        va_in.append(left_pad(s[:-1], L))
        va_tgt.append(s[-1])
    tr_in, tr_out = np.array(tr_in), np.array(tr_out)
    tr_len = (tr_in != 0).sum(1)
    va_in, va_tgt = np.array(va_in), np.array(va_tgt)
    va_order = np.argsort((va_in != 0).sum(1), kind="stable")  # similar lengths per eval batch
    va_in, va_tgt = va_in[va_order], va_tgt[va_order]

    seed_everything(hp["seed"])
    model = SASRec(n_items, L, hp["hidden"], hp["n_layers"], hp["n_heads"], hp["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    rng = np.random.default_rng(hp["seed"])
    best, best_state, bad, curve = -1.0, None, 0, []
    for ep in range(1, hp["max_epochs"] + 1):
        model.train()
        t = time.time()
        tot = 0.0
        for b in _bucketed_batches(tr_len, hp["batch_size"], rng):
            xb, yb = trim_padding(tr_in[b], tr_out[b])
            x = torch.as_tensor(xb, device=device)
            y = torch.as_tensor(yb, device=device)
            h = model.forward(x)
            real = y > 0  # score only real target positions (same loss as ignore_index=0, far cheaper)
            logits = h[real] @ model.item_emb.weight.T  # (n_real, n_items + 1)
            loss = F.cross_entropy(logits, y[real])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        ndcg, rec = evaluate(model, va_in, va_tgt, device)
        curve.append({"epoch": ep, "loss": tot / len(tr_in), "val_ndcg10": ndcg, "val_recall10": rec,
                      "sec": round(time.time() - t, 1)})
        print(json.dumps(curve[-1]))
        if ndcg > best:
            best, bad = ndcg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, curve


@torch.no_grad()
def topk_for_instances(
    model: SASRec, vocab: dict[str, int], instances: list[dict], n: int, max_len: int, device: str,
    popularity: pd.Series,
) -> list[dict]:
    """Top-n warm items per instance, excluding the user's history. Users whose
    history has no in-vocabulary item fall back to popularity (flagged)."""
    model.eval()
    inv = np.empty(len(vocab) + 1, dtype=object)
    for it, k in vocab.items():
        inv[k] = it
    pop_order = [i for i in popularity.index if i in vocab]
    out = []
    for r in instances:
        hist_items = [h[0] for h in r["history"]]
        seq = [vocab[i] for i in hist_items if i in vocab]
        hist_set = set(hist_items)
        if seq:
            s = model.scores(torch.as_tensor([left_pad(seq, max_len)], device=device))[0]
            s[0] = -torch.inf
            s[torch.as_tensor(seq, device=device)] = -torch.inf
            top = torch.topk(s, min(n + len(hist_set), len(vocab)))
            ranked = [(inv[k], float(v)) for k, v in zip(top.indices.tolist(), top.values.tolist())]
            src = "sasrec"
        else:
            ranked = [(i, float(len(pop_order) - k)) for k, i in enumerate(pop_order[: n + len(hist_set)])]
            src = "popularity_fallback"
        ranked = [(i, v) for i, v in ranked if i not in hist_set][:n]
        out.append({"instance_id": r["instance_id"], "source": src, "n_hist_in_vocab": len(seq),
                    "items": [i for i, _ in ranked], "scores": [v for _, v in ranked]})
    return out


def warm_sanity(topk: list[dict], instances: list[dict], popularity: pd.Series, vocab: dict, k: int = 10) -> dict:
    """Recall@k / NDCG@k on old-target instances (warm next items): SASRec vs popularity."""
    tk = {r["instance_id"]: r["items"] for r in topk}
    pop = [i for i in popularity.index if i in vocab]
    res = {"n": 0, "sasrec_recall": 0.0, "sasrec_ndcg": 0.0, "pop_recall": 0.0, "pop_ndcg": 0.0}
    for r in instances:
        if r["target_item"] not in vocab:
            continue
        res["n"] += 1
        hist = {h[0] for h in r["history"]}
        for name, lst in (("sasrec", tk[r["instance_id"]]), ("pop", [i for i in pop if i not in hist])):
            lst = lst[:k]
            if r["target_item"] in lst:
                rank = lst.index(r["target_item"])
                res[f"{name}_recall"] += 1
                res[f"{name}_ndcg"] += 1 / np.log2(rank + 2)
    for key in list(res):
        if key != "n" and res["n"]:
            res[key] /= res["n"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hp = cfg["sasrec"]
    sd = split_dir(cfg)
    T = date_to_ts(str(cfg["cutoff_T"]))

    pos = pd.read_parquet(sd / "positives.parquet")
    pre = pos[pos["ts"] < T]
    train_df = kcore(pre, hp["kcore"])
    vocab, seqs = build_sequences(train_df)
    print(f"[sasrec] pre-T positives {len(pre):,}; {hp['kcore']}-core: {len(train_df):,} interactions, "
          f"{len(seqs):,} users, {len(vocab):,} items; device={args.device}")

    model, curve = train(seqs, len(vocab), hp, args.device)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "hp": hp}, sd / "sasrec.pt")

    popularity = pre.groupby("item_id").size().sort_values(ascending=False, kind="mergesort")
    instances = []
    for name in ("cold_dev", "cold_test", "old_dev", "old_test"):
        instances += list(read_jsonl(sd / f"{name}.jsonl"))
    topk = topk_for_instances(model, vocab, instances, args.topn, hp["max_seq_len"], args.device, popularity)
    write_jsonl(sd / "sasrec_topk.jsonl", topk)

    old = [r for r in instances if r["kind"] == "old_target"]
    sanity = warm_sanity(topk, old, popularity, vocab)
    report = {
        "n_train_interactions": len(train_df), "n_users": len(seqs), "n_items": len(vocab),
        "best_val_ndcg10": max(c["val_ndcg10"] for c in curve), "curve": curve,
        "fallback_rate": float(np.mean([r["source"] != "sasrec" for r in topk])),
        "old_target_sanity_at10": sanity,
    }
    (sd / "sasrec_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
