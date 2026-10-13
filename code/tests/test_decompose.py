import numpy as np
import pandas as pd
import pytest

from coldrank.eval.decompose import R_RANDOM, decompose, rates, summary_md
from coldrank.eval.stats import bootstrap_slope, ols_slope

N = 200  # instances


def _rows(cond, neg, recall, cold=None, trap=None, prefix="cold", model="m", method="direct", n=N,
          top=None, relabeled=()):
    out = []
    for i in range(n):
        out.append({"condition": cond, "neg": neg, "model": model, "method": method,
                    "instance_id": f"{prefix}_{i:04d}", "recall1": float(recall[i]),
                    "cold1": np.nan if cold is None else float(cold[i]),
                    "trap1": np.nan if trap is None else float(trap[i]), "empty": False,
                    "top_item": None if top is None else top[i], "relabeled": relabeled})
    return out


def _pattern(rate, n=N):
    """Deterministic 0/1 vector with exactly round(rate*n) ones, first instances first."""
    k = int(round(rate * n))
    return np.array([1] * k + [0] * (n - k))


def test_recency_driven_model():
    # Perfect when dates are true; random-level when hidden; zero when swapped.
    rows = (_rows("baseline", "random", _pattern(0.30)) + _rows("a3_hide", "random", _pattern(0.05))
            + _rows("a2_swap", "random", _pattern(0.0)) + _rows("a4_shuffle", "random", _pattern(0.02))
            + _rows("a6_allcold", "random", _pattern(0.03), n=N - 1))
    q = decompose(pd.DataFrame(rows), [1, 2, 5, 10]).set_index("quantity")
    assert q.loc["d_evidence", "value"] == pytest.approx(0.05 - R_RANDOM)
    assert q.loc["d_recency", "value"] == pytest.approx(0.25) and q.loc["d_recency", "p"] < 1e-6
    # decomposition adds up to R_true - R_random
    assert q.loc["d_evidence", "value"] + q.loc["d_recency", "value"] == pytest.approx(0.30 - R_RANDOM)
    assert q.loc["swap_sens", "value"] == pytest.approx(0.30)
    assert q.loc["shuffle_effect", "value"] == pytest.approx(0.28)
    assert q.loc["allcold_effect", "n"] == N - 1          # paired on the shared instances only
    lo, hi = q.loc["d_recency", "lo"], q.loc["d_recency", "hi"]
    assert lo <= 0.25 <= hi


def test_relevance_driven_model_has_no_recency_effect():
    same = _pattern(0.40)
    rows = _rows("baseline", "random", same) + _rows("a3_hide", "random", same) + _rows("a2_swap", "random", same)
    q = decompose(pd.DataFrame(rows), [1, 2, 5, 10]).set_index("quantity")
    assert q.loc["d_recency", "value"] == 0 and q.loc["d_recency", "p"] == 1.0
    assert q.loc["swap_sens", "value"] == 0
    assert q.loc["d_evidence", "value"] == pytest.approx(0.40 - R_RANDOM)


def test_old_family_trap_effect_flood_slope_and_gap():
    # Recall@1 falls by 0.02 per added fresh-dated trap; Trap@1 rises by 0.05 per trap.
    levels = [0, 1, 2, 5, 10]
    rows = _rows("a1ctrl", "trap", _pattern(0.40), trap=_pattern(0.10), prefix="old")
    for n in levels:
        cond = "a1_oldgold" if n == 0 else f"a5_n{n}"
        rows += _rows(cond, "trap", _pattern(0.30 - 0.02 * n), trap=_pattern(0.20 + 0.05 * n), prefix="old")
    rows += _rows("baseline", "random", _pattern(0.50))
    q = decompose(pd.DataFrame(rows), [1, 2, 5, 10]).set_index("quantity")
    assert q.loc["trap_effect", "value"] == pytest.approx(0.10)
    assert q.loc["flood_slope", "value"] == pytest.approx(-0.02)
    assert q.loc["flood_slope_trap", "value"] == pytest.approx(0.05)
    assert q.loc["flood_slope", "lo"] <= -0.02 <= q.loc["flood_slope", "hi"]
    gap = q.loc["overcredit_gap"]
    assert gap["value"] == pytest.approx(0.50 - 0.30) and gap["neg"] == "random-vs-trap"
    assert "UNPAIRED" in gap["test"]


def test_missing_conditions_are_skipped_and_summary_renders():
    rows = _rows("baseline", "hard", _pattern(0.1))
    df = pd.DataFrame(rows)
    dec = decompose(df, [1, 2, 5, 10])
    assert dec.empty
    rows += _rows("a3_hide", "hard", _pattern(0.05))
    df = pd.DataFrame(rows)
    dec = decompose(df, [1, 2, 5, 10])
    md = summary_md(dec, rates(df), "t")
    assert "hard negatives" in md and "| m | direct |" in md


def test_baseline_and_hide_must_share_instances():
    rows = _rows("baseline", "random", _pattern(0.1)) + _rows("a3_hide", "random", _pattern(0.1), n=N - 5)
    with pytest.raises(AssertionError):
        decompose(pd.DataFrame(rows), [1, 2, 5, 10])


def test_bootstrap_slope():
    x = [0, 1, 2, 5, 10]
    Y = np.column_stack([_pattern(0.5 - 0.03 * v) for v in x])
    s = bootstrap_slope(Y, x, n_boot=2000)
    assert s["slope"] == pytest.approx(ols_slope(x, Y.mean(0))) == pytest.approx(-0.03)
    assert s["lo"] <= -0.03 <= s["hi"]


def test_flood_lift_isolates_the_date_effect():
    # Every A5_n relabels items r0..r(n-1). A date-blind model picks the same item in A1
    # and A5 (here r0 in 30% of instances): Trap@1 rises mechanically, but lift is 0.
    # A date-driven model switches to a relabeled item in A5: lift > 0.
    levels = [1, 2, 5, 10]
    a1_top = ["r0" if i < 60 else "gold" for i in range(N)]
    for kind, a5_top, expect in (("blind", a1_top, 0.0), ("driven", ["r0"] * N, 0.70)):
        rows = _rows("a1_oldgold", "trap", _pattern(0.2), prefix="old", top=a1_top, model=kind)
        for n in levels:
            rel = tuple(f"r{j}" for j in range(n))
            rows += _rows(f"a5_n{n}", "trap", _pattern(0.2), prefix="old", top=a5_top, relabeled=rel, model=kind)
        q = decompose(pd.DataFrame(rows), levels).set_index("quantity")
        for n in levels:
            assert q.loc[f"flood_lift_n{n}", "value"] == pytest.approx(expect), (kind, n)
