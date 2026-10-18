# EB-NeRD vs Amazon: Llama-2-7B at K = 15 (dev + test)

*18 Oct 2026. The EB-NeRD (Ekstra Bladet news) port of everything run on Amazon Video Games in [`analysis_2026-10-16_k15.md`](analysis_2026-10-16_k15.md) (the "Amazon report"). Setup and decisions: [`plan_ebnerd.md`](plan_ebnerd.md), `design.md` § EB-NeRD port. All numbers were recomputed from the raw outputs of ADA job **25738** (84 tasks: the K = 15 baseline × 7 methods and pointwise + Direct-Rec on every K = 15 variant, dev and test; gnode056, 11 h 26 min, exit 0).*

---

## 1. Summary

1. **On news, Llama-2-7B pointwise is at chance on test.** Recall@1 is **0.066** with random and with hard negatives (chance 0.067; Amazon 0.149 / 0.105). Dev looked better (0.107), but its CI was wide (150 pools); test (800) settles it.
2. **So there is no cold-start "success" to explain on news.** Δrecency = **+0.009** (n.s.), Δevidence = −0.009 (n.s.). On Amazon Δrecency was +0.074\*. **The domain difference is significant:** +0.065, z = 4.3, p = 2×10⁻⁵.
3. **But the model's date preference is just as strong on news.** Exchanging two displayed dates shifts the scores by **−0.070 / +0.059 log-odds** (Amazon −0.075 / +0.069). Swap sensitivity (+0.020\*), shuffle effect (+0.018\*) and flood lift (+0.076\*) are significant, as on Amazon.
4. **Why the same preference helps on Amazon and not on news:** the date push has a fixed size, but on news the gold trails its best competitor by much more (median 0.43 vs 0.24 log-odds). A push the size of a date swap could lift the gold to the top in only 17% of news pools against 30% on Amazon. The competitors are SASRec's warm picks: recent articles on the stories the user already reads.
5. **The prior ignores the size of the date gap.** On Amazon the gold is ~1,000 days newer than the warm candidates, on news only ~18 hours, yet the per-swap score change is the same (±0.07). The model reacts to "looks newer", not to how much newer, and not to whether recency matters in the domain.
6. **The date heuristics are much stronger on news** (newest-first 0.560 vs 0.346), so the shortcut is larger in the domain where recency is genuine, but this model does not exploit it.
7. **All other methods are at or below chance on news**, as on Amazon. Fast and SSC give no usable ranking for 51–58% of prompts (Amazon 38–42%).

---

## 2. What differs from Amazon (setup)

| | Amazon Video Games | EB-NeRD news |
|---|---|---|
| Cutoff, window | 2022-09-01, 180 days | 2023-05-25, 1 day |
| Positive | rating ≥ 3 | click |
| Displayed date | date | date + time (news ages in hours) |
| Item text | title, category (+ brand for Structural) | **English translation** (opus-mt-da-en) of title + subtitle (cut at 100 chars), section (+ topics for Structural) |
| A1 traps | low similarity to the user | **shown to the user and not clicked** |
| A1 at K = 50 | gold + 39 warm + 10 traps | gold + 46 warm + 3 traps (≥ 10 skipped new articles left only 865 users) |
| A6 | 799 test instances | **650** (K = 15 built directly: few articles are published before early targets) |
| Gold − median warm age | ~1,000 days | ~18 hours |
| K = 15 pools run | gold + 12 warm + 2 cold; A1: gold + 11 warm + 3 traps | identical |

---

## 3. Baseline, all methods (test; Recall@1 [95% CI]; chance 0.067)

| Method | News, random | News, hard | Amazon, random | Amazon, hard |
|---|---|---|---|---|
| **Pointwise** | **0.066** [0.050, 0.084] | **0.066** [0.050, 0.084] | **0.149** | **0.105** |
| Direct-Rec | 0.058 | 0.069 | 0.075 | 0.058 |
| Base | 0.076 | 0.052 | 0.054 | 0.069 |
| Fast | 0.031 | 0.030 | 0.035 | 0.030 |
| Structural | 0.050 | 0.048 | 0.028 | 0.046 |
| Structural, no recency | 0.037 | 0.045 | 0.041 | 0.033 |
| SSC | 0.029 | 0.026 | 0.040 | 0.051 |
| *newest-first* | *0.560* | *0.515* | *0.346* | *0.323* |
| *newest-3 + text* | *0.443* | *0.107* | *0.590* | *0.009* |
| *text similarity* | *0.060* | *0.037* | *0.098* | – |

- Empty rankings on news: Fast 58%, SSC 51–52%, Structural without recency 39–40%, Structural 18%, Base 7–8%, Direct-Rec 2%.
- **Dev (150 pools)**, for reference: pointwise 0.107 / 0.080, every other method 0.013–0.080.

## 4. Recall@1 per condition (test)

| Condition | Pointwise random | Pointwise hard | Direct random | Direct hard |
|---|---|---|---|---|
| baseline | 0.066 (Cold@1 0.16) | 0.066 (0.18) | 0.058 | 0.069 |
| A2 swap | 0.046 | 0.041 | 0.059 | 0.064 |
| A3 hide | 0.058 | 0.056 | 0.059 | 0.070 |
| A4 shuffle | 0.049 | 0.045 | 0.050 | 0.065 |
| A6 all-cold (650) | **0.097** | **0.094** | 0.086 | 0.085 |

| Old-target (1,000 pools) | Pointwise R@1 | Pointwise Trap@1 | Direct R@1 | Direct Trap@1 | Chance Trap@1 |
|---|---|---|---|---|---|
| A1 old-gold | 0.043 | 0.210 | 0.067 | 0.181 | 0.200 |
| A1-ctrl | 0.040 | – | 0.065 | – | – |
| A5 n = 1 | 0.042 | 0.323 | 0.065 | 0.260 | 0.267 |
| A5 n = 2 | 0.040 | 0.447 | 0.073 | 0.327 | 0.333 |
| A5 n = 5 | 0.039 | 0.702 | 0.074 | 0.536 | 0.533 |
| A5 n = 10 | 0.036 | 0.935 | 0.057 | 0.873 | 0.867 |

- **Cold@1 is below chance** (0.16–0.18 vs 0.20): on news the model does not put new articles on top (Amazon 0.33 / 0.51).
- **A5:** pointwise picks relabeled articles more often than their number predicts (Trap@1 0.70 vs 0.53 at n = 5); this is the flood lift.
- **A1 / A1-ctrl below chance (0.04):** the old gold loses to SASRec's warm picks with or without traps.

## 5. Decomposition, news vs Amazon (test; * = McNemar p < 0.05)

| Quantity | News pointwise, random | News pointwise, hard | **Amazon** pointwise, random | Amazon pointwise, hard |
|---|---|---|---|---|
| R_true → R_hide | 0.066 → 0.058 | 0.066 → 0.056 | 0.149 → 0.075 | 0.105 → 0.068 |
| **Δevidence** | −0.009 (n.s.) | −0.010 (n.s.) | +0.008 (n.s.) | +0.001 (n.s.) |
| **Δrecency** | **+0.009** [−0.007, +0.025] (n.s.) | +0.010 (n.s.) | **+0.074\*** | +0.037\* |
| **Swap sensitivity** | +0.020\* | +0.025\* | +0.062\* | +0.054\* |
| Shuffle effect | +0.018\* | +0.021\* | +0.056\* | +0.041\* |
| All-cold effect | **−0.031\*** | −0.029\* | +0.041\* | +0.048\* |
| Trap effect (A1-ctrl − A1) | −0.003 (n.s.) | – | −0.009 (n.s.) | – |
| **Flood lift** (n = 10) | **+0.076\*** | – | +0.121\* | – |
| Direct-Rec Δrecency | −0.001 (n.s.) | −0.001 (n.s.) | +0.018 (n.s.) | +0.004 (n.s.) |

**Domain difference in Δrecency (pointwise, random negatives):** Amazon − news = **+0.065**, SE 0.015, **z = 4.3, p = 2×10⁻⁵** (independent instances; SE from the two CIs). The planned domain × condition logistic regression is the formal version of this test.

**Negative all-cold effect:** on news, pointwise does *better* when every candidate is new (0.097 vs 0.066). Among same-day articles it can use content; mixed with SASRec's warm picks it cannot.

## 6. The date effect measured directly (A2 vs baseline: same items and positions, two dates exchanged)

| | Gold (now looks **old**) | Partner (now looks **new**) | Date gap gold vs warm (median) |
|---|---|---|---|
| **News**, random | **−0.070** [−0.074, −0.065] | **+0.059** [+0.055, +0.064] | ~18 hours |
| News, hard | −0.072 [−0.077, −0.067] | +0.058 [+0.054, +0.062] | |
| **Amazon**, random | −0.075 [−0.080, −0.071] | +0.069 [+0.065, +0.074] | ~1,030 days |
| Amazon, hard | −0.079 [−0.083, −0.074] | +0.071 [+0.066, +0.075] | |

**Score vs age within a pool** (Spearman, newer → higher score; baseline, random negatives):

| | Dates shown | Dates hidden (A3) | Added by the dates |
|---|---|---|---|
| News | +0.038 [+0.020, +0.056] | −0.033 | **+0.071** |
| Amazon | +0.095 [+0.074, +0.116] | −0.054 | **+0.149** |

## 7. Why the same prior pays off on Amazon and not on news

| Pointwise, baseline random, test | News | Amazon |
|---|---|---|
| Score SD within a pool (median) | 0.275 | 0.191 |
| Gold's deficit to the best other candidate (median) | **0.434** | **0.242** |
| Pools where a +0.14 push (one date swap's worth) would make the gold #1 | **17%** | **30%** |
| Gold ranked first among the 3 new items (chance 0.333) | 0.422 | 0.419 |

1. **The push is the same size in both domains** (§6), whatever the size of the date gap (18 hours vs 3 years). The model's recency prior is domain-insensitive and magnitude-insensitive: it reacts to which date looks newer.
2. **The content signal among new items is the same too:** the gold wins among the 3 new items 42% of the time in both domains (chance 33%).
3. **The difference is the competition.** On news the gold trails the best competitor by almost twice as much, so the fixed push lifts it to the top about half as often. The competitors are SASRec's warm picks, recent articles on the stories the user already follows, which the model scores highly on content; Cold@1 is below chance.
4. **Hence:** on Amazon, prior + shortcut = above-chance accuracy that vanishes without dates (over-crediting). On news, the same prior meets stronger warm competitors and never turns into accuracy, even though the date shortcut itself is stronger there (newest-first 0.56).

## 8. What this means for the paper

1. **Recency bias:** Llama-2-7B pointwise shows a **systematic, domain-insensitive recency prior**: a date swap moves an item by ±0.06–0.08 log-odds in a static catalogue and in news alike, regardless of whether the gap is hours or years and of whether recency is informative in the domain.
2. **Over-crediting is domain-dependent:** the prior creates a spurious cold-start advantage only where it can tip close decisions (Amazon: Δrecency +0.074\*, all of the above-chance accuracy). On news it does not (Δrecency +0.009, n.s.); the difference is significant (p = 2×10⁻⁵).
3. **The abstract's expected contrast is only half confirmed:** we expected recency to be a justified signal the model exploits in news. The shortcut is larger in news (0.56 vs 0.35 for newest-first), but this model does not exploit it; it applies the same weak prior in both domains. Report this as a finding: the model does not adapt its use of recency to the domain.
4. **Unchanged across domains:** traps are not preferred (Trap@1 ≤ chance); Direct-Rec and the reasoning methods are at chance and largely unparseable.

**Suggested wording:**
> *Llama-2-7B's pointwise reranker carries the same weak recency prior in both domains: exchanging two displayed dates shifts the scores by ±0.07 log-odds whether the items are three years or eighteen hours apart. In the static catalogue this prior accounts for the model's entire above-chance cold-start accuracy (Δrecency = +0.074, Δevidence ≈ 0). In news, where the recency shortcut is even stronger (newest-first Recall@1 0.56), the same prior is too small to overcome the higher scores the model gives the warm candidates, recent articles on the user's ongoing stories,, and the model stays at chance (Δrecency = +0.009, n.s.; domain difference p < 10⁻⁴). The model neither exploits recency where it is informative nor suppresses it where it is not.*

## 9. Caveats

- **Translation:** the model reads machine translations (greedy opus-mt-da-en). Names are sometimes translated literally ("Mette Skak" → "Mette Chess"). This could weaken content matching on news, but not the date effects, which are measured on identical text.
- **Different splits:** the domains differ in window (1 vs 180 days), date resolution, pool composition (A1 with 3 traps at K = 50, A6 with 650 instances) and the trap definition, so the domain comparison is not a pure manipulation of "domain".
- **Dev vs test:** dev suggested pointwise 0.107 on news; test (800) gives 0.066. Use the test numbers.
- **One model, one method above chance (on Amazon only).** Other K = 50 models penalised new dates (Amazon).
- **The domain z-test** treats the two estimates as independent with normal errors; the logistic regression (plan_adversarial.md §5) is still to be run.

## 10. Files and commands

- Outputs: `outputs/ebnerd/k15_*/<neg>/llama2_7b/<method>/{dev,test}/`; logs `outputs/logs/ebnerd_llama2_7b_k15-all_25738/`.
- `python -m coldrank.eval.report --config ebnerd --split test --condition <c>`; `python -m coldrank.eval.decompose --config ebnerd --prefix k15_ --split test`.
- §6–§7 diagnostics were computed from `raw.jsonl` (pointwise scores per candidate) with the same code as the Amazon report §10.3.
