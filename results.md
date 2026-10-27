# Results

Running log of results, newest sections appended at the end. Setup and decisions: `design.md`.

## Llama-2-7B fine-tuned on Amazon Video Games — dev set (2026-10-26)

**Model.** Llama-2-7B-Chat (`meta-llama/Llama-2-7b-chat-hf` @ `f5db02db…`, fp16), **fine-tuned on Amazon Video Games** with QLoRA (4-bit, 1 epoch, pointwise Yes/No, dates shown as in production; 23,778 training examples from 4,183 pools at 6 earlier cutoffs; job 40143). **Zero-shot** = the same fp16 weights without fine-tuning (job 40931); fine-tuned evaluation: job 40930.

**Evaluation.** Amazon VG dev, K = 15 candidates, pointwise scoring. 150 users per cold-target cell (Baseline, A2–A6), 144 per old-target cell (A1-ctrl, A1, A5). Recall@1 as random / hard negatives; chance = 1/15 = 0.067. **Change** = the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. The A5 pools also contain A1's 3 traps.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct item is new; dates shown | 0.080 / 0.060 | 0.373 / 0.460 | +0.293\* / +0.400\* vs zero-shot | Fine-tuning works: 5–7× chance, above the newest-first rule (0.347 / 0.307) |
| A2 Swap | correct item's date swapped with an old wrong item's | 0.080 / 0.053 | 0.007 / 0.007 | −0.367\* / −0.453\* | Accuracy follows the date, not the item: almost all of it moves with the "new" date |
| A3 Hide | all dates removed | 0.073 / 0.073 | 0.093 / 0.087 | −0.280\* / −0.373\* | Without dates it falls to about chance (Δrecency). This overstates the recency share somewhat, because the model never saw date-less prompts (dates-hidden control planned for later). |
| A4 Shuffle | dates randomly reassigned among candidates | 0.067 / 0.060 | 0.093 / 0.060 | −0.280\* / −0.400\* | Wrong dates break it as badly as no dates: the dates drive its ranking |
| A6 All-new | all 15 candidates are new items | 0.067 / 0.067 | 0.240 / 0.280 | −0.133\* / −0.180\* | Still 4× chance when dates can't separate the items, so it did learn some content; but it loses a third of its accuracy without the "only the gold is new" cue |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.146 | 0.389 | — | Reference for the rows below |
| A1 Traps | correct item old + 3 new irrelevant items | 0.118 | 0.139 | −0.250\* | New-looking irrelevant items steal the top spot: it picks a trap first 53% of the time (chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled with a new date | 0.111 | 0.125 | −0.264\* | Even one fake "new" date hurts |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.104 | 0.083 | −0.306\* | Damage grows with the number of new-looking items … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.118 | 0.062 | −0.326\* | … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.104 | 0.042 | −0.347\* | … to below chance: simply labelling items "new" is enough to bury the correct one |

**Summary.** Before fine-tuning, Llama-2-7B (fp16) is at chance and does not react to dates (no zero-shot change is significant). After fine-tuning on Amazon VG with Li et al.'s protocol it is accurate (0.37–0.46), but every manipulation of dates destroys that accuracy: it learned some relevance (A6), but mostly to pick whatever looks newest — the recency shortcut. Dev results; test (jobs 41008 / 41009) pending.

## Llama-2-7B fine-tuned on Amazon Video Games — test set (2026-10-27)

**Model.** The same fine-tuned Llama-2-7B-Chat as in the dev section (fine-tuned on Amazon Video Games, job 40143). Evaluation jobs: fine-tuned 41008, zero-shot (the same fp16 weights without fine-tuning) 41009.

**Evaluation.** Amazon VG test, K = 15 candidates, pointwise scoring. There are 800 users per cold-target cell (Baseline, A2–A6; A6 has 799) and 967 per old-target cell (A1-ctrl, A1, A5). Recall@1 is reported as random / hard negatives; chance = 1/15 = 0.067. **Change** = the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. The A5 pools also contain A1's 3 traps. Trap@1 = how often a new-looking wrong item is ranked first (chance 0.20).

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct item is new; dates shown | 0.083 / 0.072 | 0.445 / 0.546 | +0.362\* / +0.474\* vs zero-shot | Fine-tuning works: 7–8× chance, above the newest-first rule (0.346 / 0.323) |
| A2 Swap | correct item's date swapped with an old wrong item's | 0.086 / 0.066 | 0.005 / 0.006 | −0.440\* / −0.540\* | Accuracy follows the date, not the item: almost all of it moves with the "new" date |
| A3 Hide | all dates removed | 0.068 / 0.061 | 0.101 / 0.107 | −0.344\* / −0.439\* | Close to chance without dates. As on dev, this overstates the recency share somewhat, because the model never saw prompts without dates |
| A4 Shuffle | dates randomly reassigned among candidates | 0.083 / 0.066 | 0.077 / 0.077 | −0.367\* / −0.469\* | Chance: wrong dates break the model as badly as no dates |
| A6 All-new | all 15 candidates are new items | 0.076 / 0.053 | 0.253 / 0.292 | −0.193\* / −0.255\* | Still about 4× chance, so the model learned some content, but it loses 43–47% of its accuracy without the "only the correct item is new" cue |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.101 | 0.384 | — | Reference for the rows below |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.087 | 0.152 | −0.232\* | New-looking irrelevant items take the top spot: Trap@1 0.562 (zero-shot 0.198, i.e. chance) |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled with a new date | 0.091 | 0.103 | −0.280\* | One fake "new" date hurts further (Trap@1 0.675) |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.089 | 0.084 | −0.300\* | The damage grows with the number of new-looking items (Trap@1 0.737) … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.091 | 0.067 | −0.316\* | … (Trap@1 0.845) … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.093 | 0.060 | −0.324\* | … down to chance: labelling items "new" is enough to bury the correct one (Trap@1 0.930) |

**Summary.** Test replicates dev in every row (same signs, similar sizes, all fine-tuned changes p < 1e-20). Zero-shot, Llama-2-7B is at chance and barely reacts to dates: two zero-shot changes reach p < 0.05, but both are ≤ 0.02 (A6 hard −0.020, A1 −0.014). Its Trap@1 under flood only tracks how many new-looking items there are. Fine-tuned on Amazon VG with Li et al.'s protocol, it is accurate (0.45 / 0.55). Swapping, hiding or shuffling the dates drops it to chance or below, and new-looking items take the top spot 56–93% of the time. It learned some relevance (A6), but mostly the recency shortcut.

## Llama-2-7B fine-tuned on EB-NeRD (news) — dev set (2026-10-27)

**Model.** Llama-2-7B-Chat (`meta-llama/Llama-2-7b-chat-hf` @ `f5db02db…`, fp16), **fine-tuned on EB-NeRD** with the same recipe as Amazon: QLoRA (4-bit, 1 epoch, pointwise Yes/No, dates shown as in production, same effective batch 32 as 4 GPUs × 2 × 4 because news prompts are ~2.4× longer). Training data: 30,018 examples (4,181 users) from training pools at 7 earlier cutoffs (6–24 May 2023); job 40929, 5 h 01 min, final validation loss 0.200. **Zero-shot** = the same fp16 weights without fine-tuning (job 40932); fine-tuned evaluation: job 40933.

**Evaluation.** EB-NeRD dev, K = 15 candidates, pointwise scoring. There are 150 users per cell, except A6 with 131 (only 131 dev users have 14 new articles published before the click). Recall@1 is reported as random / hard negatives; chance = 1/15 = 0.067. **Change** = the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. Trap@1 chance = 0.20. The EB-NeRD A1-ctrl pools are not yet rebuilt with the letter-code fix; this does not affect pointwise prompts.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.100 / 0.073 | 0.433 / 0.413 | +0.333\* / +0.340\* vs zero-shot | Fine-tuning helps a lot (6× chance), but it is still **below the newest-first rule** (0.567 / 0.540): on news, simply sorting by date beats the fine-tuned model |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.060 / 0.040 | 0.000 / 0.000 | −0.433\* / −0.413\* | All of the accuracy moves with the date: it never picks the correct article once that article looks old |
| A3 Hide | all dates removed | 0.060 / 0.040 | 0.007 / 0.007 | −0.427\* / −0.407\* | Without dates it falls *below* chance: it has essentially nothing besides the date |
| A4 Shuffle | dates randomly reassigned among candidates | 0.047 / 0.027 | 0.053 / 0.067 | −0.380\* / −0.347\* | Chance: wrong dates break it completely |
| A6 All-new | all 15 candidates are new articles | 0.107 / 0.084 | 0.115 / 0.084 | −0.328\* / −0.328\* | **Chance** when dates cannot separate the candidates: unlike Amazon, it learned no usable content signal for news |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.020 | 0.247 | — | Reference for the rows below |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.047 | 0.127 | −0.120\* | Traps take the top spot 77% of the time (zero-shot 15%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled with a new date | 0.047 | 0.107 | −0.140\* | Trap@1 0.827 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.033 | 0.107 | −0.140\* | … 0.840 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.033 | 0.107 | −0.140\* | … 0.860 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.027 | 0.100 | −0.147\* | … 0.887. Already saturated after one fake date: the 3 traps alone take most top spots, so extra fake-new items change little |

**Summary.** On news the shortcut is even more complete than on Amazon. Zero-shot, Llama-2-7B is at chance, with a small date sensitivity (swap −0.040\*, shuffle −0.053\* / −0.047\*). The fine-tuned model reaches 0.43 / 0.41, yet it never beats the newest-first rule. It drops to 0 when the correct article's date is swapped and to below chance when dates are hidden, and is at chance when all candidates are new (A6). New-looking traps win 77–89% of top spots. Fine-tuning taught it the date rule and, unlike on Amazon, no content signal. Dev results; EB-NeRD test pending (after the A1-ctrl pool rebuild).

## Llama-2-7B fine-tuned on EB-NeRD (news) — test set (2026-10-27)

**Model.** This is the same fine-tuned Llama-2-7B-Chat as the EB-NeRD dev section above (fine-tuned on EB-NeRD, job 40929). Evaluation jobs: 41455 for the fine-tuned model and 41456 for zero-shot (the same fp16 weights without fine-tuning).

**Evaluation.** EB-NeRD test, K = 15 candidates, pointwise scoring.
- **Users:** 800 per cold-target cell (Baseline, A2–A4); 650 for A6 (only 650 test users have 14 new articles published before the click); 1,000 per old-target cell (A1-ctrl, A1, A5).
- **Recall@1** is given as random / hard negatives; chance = 1/15 = 0.067.
- **Change** is the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05.
- **Trap@1** chance = 0.20.
- **Pools:** A1-ctrl uses the rebuilt pools (letter-code fix; pointwise prompts unchanged).

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.055 / 0.054 | 0.394 / 0.487 | +0.339\* / +0.434\* vs zero-shot | A big gain over zero-shot (6–7× chance), but still **below the newest-first rule** (0.560 / 0.515) |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.036 / 0.039 | 0.001 / 0.000 | −0.393\* / −0.487\* | All of the accuracy moves with the date: once the correct article looks old, it is (almost) never picked |
| A3 Hide | all dates removed | 0.024 / 0.025 | 0.014 / 0.013 | −0.380\* / −0.475\* | Below chance without dates: it has essentially nothing besides the date |
| A4 Shuffle | dates randomly reassigned among candidates | 0.043 / 0.040 | 0.035 / 0.031 | −0.359\* / −0.456\* | Below chance: wrong dates break it completely |
| A6 All-new | all 15 candidates are new articles | 0.094 / 0.062 | 0.097 / 0.117 | −0.305\* / −0.368\* | Near chance when dates cannot separate the candidates: at most a weak content signal (1.5–1.75× chance) |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.065 | 0.241 | — | Reference for the rows below |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.059 | 0.106 | −0.135\* | Traps take the top spot 83% of the time (zero-shot 18%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled with a new date | 0.059 | 0.104 | −0.137\* | Trap@1 0.846 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.059 | 0.107 | −0.134\* | … 0.850 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.053 | 0.113 | −0.128\* | … 0.858 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.049 | 0.110 | −0.131\* | … 0.888. Already saturated: the 3 traps alone take most top spots |

**Summary.** Test replicates dev: same signs and similar sizes, and every fine-tuned change has p < 1e-26.
- **Zero-shot:** at or below chance (0.055 / 0.054), with small but significant date effects: swap −0.019\* / −0.015\*, hide −0.031\* / −0.029\*, shuffle −0.013\* / −0.014\*.
- **Fine-tuned:** 0.39 / 0.49, yet below the newest-first rule. It is at 0 when the correct article's date is swapped and below chance with dates hidden or shuffled. It is near chance when all candidates are new, and traps win 83–89% of top spots.
- **On news, fine-tuning with Li et al.'s protocol teaches the date rule and almost nothing else.**
