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

## Qwen2.5-7B fine-tuned on Amazon Video Games — dev set (2026-10-28)

**Model.** Qwen2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…`, fp16), **fine-tuned on Amazon Video Games** with the same recipe as Llama-2: QLoRA (4-bit, 1 epoch, pointwise Yes/No, dates shown as in production, effective batch 32). Training data: 23,712 examples from 4,166 pools at 6 earlier cutoffs (Sep 2019 – Mar 2022). Training ran on one 11 GB GPU (ada, job 40320, 7 h 03 min, final validation loss 0.245). **Zero-shot** = the same fp16 weights without fine-tuning (job 42662); fine-tuned evaluation: job 40321.

**Differences from the Llama-2 runs.**
- **One-GPU training:** batch 2 × 16 gradient accumulation (same effective batch 32), and the input embeddings and `lm_head` kept in fp16 (`keep_embeddings_fp16`) so Qwen's 152k-token vocabulary fits in memory (`configs/finetune_qwen25_1gpu.yaml`).
- **Evaluation:** both models are loaded 4-bit at load time (vLLM bitsandbytes) on one GPU instead of fp16 on two. Zero-shot and fine-tuned use the same setting, so the change columns are like-for-like. Absolute numbers are not strictly comparable with the fp16 Llama-2 rows.
- **Fast-vs-exact pointwise check:** tolerance 0.15 instead of 0.05 (the 4-bit loading adds up to 0.09 log-odds of rounding noise).

**Evaluation.** Amazon VG dev, K = 15 candidates, pointwise scoring. 150 users per cold-target cell (Baseline, A2–A4), 149 for A6, 142 per old-target cell (A1-ctrl, A1, A5). Recall@1 as random / hard negatives; chance = 1/15 = 0.067. **Change** = the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. The A5 pools also contain A1's 3 traps. Trap@1 chance = 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct item is new; dates shown | 0.040 / 0.027 | 0.440 / 0.533 | +0.400\* / +0.507\* vs zero-shot | Fine-tuning works: 6–8× chance, above the newest-first rule (0.347 / 0.360) |
| A2 Swap | correct item's date swapped with an old wrong item's | 0.027 / 0.013 | 0.000 / 0.007 | −0.440\* / −0.527\* | All of the accuracy moves with the "new" date |
| A3 Hide | all dates removed | 0.080 / 0.067 | 0.147 / 0.147 | −0.293\* / −0.387\* | About 2× chance without dates; two-thirds or more of the accuracy depends on the date (the model never saw date-less prompts, so this overstates the recency share somewhat) |
| A4 Shuffle | dates randomly reassigned among candidates | 0.047 / 0.033 | 0.093 / 0.120 | −0.347\* / −0.413\* | Wrong dates break it about as badly as no dates |
| A6 All-new | all 15 candidates are new items | 0.161 / 0.040 | 0.141 / 0.275 | −0.302\* / −0.262\* | Some content learned (4× chance on hard negatives), but random negatives fall to about 2× chance without the "only the correct item is new" cue |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.042 | 0.387 | — | Reference for the rows below |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.056 | 0.169 | −0.218\* | Traps take the top spot: Trap@1 0.585 (zero-shot 0.000) |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled with a new date | 0.056 | 0.141 | −0.246\* | Trap@1 0.655 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.056 | 0.113 | −0.275\* | … 0.725 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.056 | 0.099 | −0.289\* | … 0.810 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.106 | 0.077 | −0.310\* | … 0.901: at about chance once enough items look new |

**Summary.** Fine-tuned Qwen2.5-7B shows the same pattern as fine-tuned Llama-2. It is accurate (0.44 / 0.53, above newest-first), but swapping, hiding or shuffling the dates removes most of that accuracy, and new-looking items take 59–90% of top spots. Zero-shot it is *below* chance (0.040 / 0.027) and the dates barely move it: of A2–A4, only hide on hard negatives is significant (+0.040\*). It avoids new items for reasons other than their dates: its Recall@1 rises to 0.161 when every candidate is new (A6 random, +0.121\*), and it almost never ranks a trap first (Trap@1 0.000 on A1).

## Qwen2.5-7B fine-tuned on Amazon Video Games — test set (2026-10-28)

**Model.** The same fine-tuned Qwen2.5-7B-Instruct as in the dev section (job 40320), with the same 4-bit evaluation setting. Evaluation jobs: fine-tuned 40321, zero-shot 42662 (each job covers dev and test).

**Evaluation.** Amazon VG test, K = 15 candidates, pointwise scoring.
- **Users:** 800 per cold-target cell (Baseline, A2–A6); 962 per old-target cell (A1-ctrl, A1, A5).
- **Recall@1** is given as random / hard negatives; chance = 1/15 = 0.067.
- **Change** is the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05.
- **Trap@1** chance = 0.20. The A5 pools also contain A1's 3 traps.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct item is new; dates shown | 0.045 / 0.031 | 0.432 / 0.532 | +0.388\* / +0.501\* vs zero-shot | Fine-tuning works: 6–8× chance, above the newest-first rule (0.321 / 0.316) |
| A2 Swap | correct item's date swapped with an old wrong item's | 0.040 / 0.029 | 0.015 / 0.014 | −0.417\* / −0.519\* | Accuracy follows the date, not the item |
| A3 Hide | all dates removed | 0.048 / 0.026 | 0.138 / 0.146 | −0.295\* / −0.386\* | About 2× chance without dates (with the same caveat as on dev) |
| A4 Shuffle | dates randomly reassigned among candidates | 0.043 / 0.031 | 0.079 / 0.081 | −0.354\* / −0.451\* | About chance: wrong dates break it completely |
| A6 All-new | all 15 candidates are new items | 0.207 / 0.045 | 0.152 / 0.297 | −0.280\* / −0.235\* | Some content learned (4.4× chance on hard negatives), but it loses 44–65% of its accuracy without the "only the correct item is new" cue |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.052 | 0.381 | — | Reference for the rows below |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.057 | 0.133 | −0.248\* | New-looking irrelevant items take the top spot: Trap@1 0.600 (zero-shot 0.004) |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled with a new date | 0.057 | 0.108 | −0.273\* | One fake "new" date hurts further (Trap@1 0.672) |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.058 | 0.093 | −0.289\* | The damage grows with the number of new-looking items (Trap@1 0.738) … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.061 | 0.064 | −0.317\* | … (Trap@1 0.857) … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.087 | 0.055 | −0.326\* | … down to below chance (Trap@1 0.940) |

**Summary.** Test replicates dev in every row (same signs, similar sizes, all fine-tuned changes p < 1e-30).
- **Zero-shot:** below chance (0.045 / 0.031) and blind to dates: no A2–A4 change is significant. Its significant changes are A6 random (+0.163\*) and the A5 flood rows (+0.006\* to +0.035\*). It rarely ranks new items first (Trap@1 0.004 on A1), so its low baseline comes from avoiding new items on content, not from reading the dates.
- **Fine-tuned:** 0.43 / 0.53, above newest-first, but swapping, hiding or shuffling the dates drops it to about chance, and new-looking items take 60–94% of top spots. It learned some content (A6 hard 0.297), but mostly the recency shortcut.
- **Same as Llama-2 on Amazon:** fine-tuning with Li et al.'s protocol turns a model that does not use dates into one that mainly follows them, for both base models.
- **Pool counts differ from the Llama-2 sections** (142 / 962 old-target users instead of 144 / 967, and a different newest-first score on hard negatives), because these pools were built separately on ada. Compare the two models by the size of their changes, not by individual cells.

## Llama-2-7B fine-tuned on EB-NeRD with impression negatives — dev set (2026-10-28)

**Model.** The same Llama-2-7B-Chat, **fine-tuned on EB-NeRD with production-style training data**. For each cold click, the negatives are the articles shown in the same impression that the user did not click (up to 5, any age), instead of our training pools. There are 5,250 instances and 27,919 training examples. In this data, the correct article is newer than all its negatives in only 12.7% of cases.
- Learning rate **1e-4**, not 2e-4: at 2e-4 the run diverged to NaN in fp16 twice (jobs 43488, 44352). Everything else is unchanged from the pool-data model: QLoRA 4-bit, 1 epoch, pointwise Yes/No, dates shown, batch 32.
- Training: job 44613, 4 h 40 min, no NaN steps, final validation loss 0.423. Evaluation: job 44614.
- **Old FT** = the pool-data fine-tune from the EB-NeRD dev section above. **Zero-shot** is reused from that section.

**Evaluation.** As in the EB-NeRD dev section: 150 users per cell (A6 131), K = 15, Recall@1 as random / hard negatives, chance 0.067. **Change** = the new model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. Trap@1 chance 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, old FT (pool data) | Recall@1, new FT (impressions) | Change (new FT) | What the change means |
|---|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.100 / 0.073 | 0.433 / 0.413 | 0.540 / 0.233 | +0.440\* / +0.160\* vs zero-shot | Still well above chance, but still **below newest-first** (0.567 / 0.540). Against the old model: random +0.107 (p = 0.09), hard −0.180\* |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.060 / 0.040 | 0.000 / 0.000 | 0.000 / 0.000 | −0.540\* / −0.233\* | Unchanged: it never picks the correct article once it looks old |
| A3 Hide | all dates removed | 0.060 / 0.040 | 0.007 / 0.007 | 0.107 / 0.093 | −0.433\* / −0.140\* | Still collapses to about chance. Slightly less total than the old model's 0.007 (+0.100\* / +0.087\* vs old) |
| A4 Shuffle | dates randomly reassigned among candidates | 0.047 / 0.027 | 0.053 / 0.067 | 0.053 / 0.060 | −0.487\* / −0.173\* | Chance: wrong dates break it completely, as before |
| A6 All-new | all 15 candidates are new articles | 0.107 / 0.084 | 0.115 / 0.084 | 0.137 / 0.084 | −0.389\* / −0.122\* | Still chance: no usable content signal (old vs new n.s.) |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.020 | 0.247 | 0.147 | — | Worse than the old model (−0.100\*) at finding an *old* correct article |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.047 | 0.127 | 0.047 | −0.100\* | Traps win **90%** of top spots (old model 77%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled new | 0.047 | 0.107 | 0.047 | −0.100\* | Trap@1 0.900 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.033 | 0.107 | 0.047 | −0.100\* | … 0.907 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.033 | 0.107 | 0.033 | −0.113\* | … 0.940 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.027 | 0.100 | 0.007 | −0.140\* | … 0.993: near-total |

**Summary.** Production-style negatives do **not** remove the recency shortcut; if anything the model relies on it more. Swapping the dates still drops it to 0 and shuffling to chance. With only new candidates (A6) it is at chance, and new-looking traps take 90–99% of top spots (old model 77–89%). Hiding dates leaves slightly more than before (0.10 vs 0.007), but that is still about chance.

Likely reason: impression negatives remove the "the gold is the *newest*" cue, but not the "*old* means No" cue. The training instances are cold clicks, so the correct article is always new, while 19.5% of the negatives are old. So an old article is never the right answer in training.

Caveats:
- The learning rate differs from the old model (1e-4 vs 2e-4).
- Dev has n = 150; the test set (next section) confirms it.

## Llama-2-7B fine-tuned on EB-NeRD with impression negatives — test set (2026-10-28)

**Model.** The impression-negative model from the dev section above (lr 1e-4, job 44613). Test job 44615. **Old FT** = the pool-data fine-tune `llama2_7b_ft_ebnerd`; **zero-shot** = `llama2_7b_fp16`, both on the same test pools.

**Evaluation.** 800 users per cell (A6 650; A1 / A1-ctrl / A5 1,000), K = 15, Recall@1 as random / hard negatives, chance 0.067. All pools have a complete ranking for all three models. **Change** = the new model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); **New vs old** = paired change of the new model against the old one on the same row. \* = McNemar p < 0.05. Trap@1 chance 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, old FT (pool data) | Recall@1, new FT (impressions) | Change (new FT) | New vs old | What it means |
|---|---|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.055 / 0.054 | 0.394 / 0.487 | 0.486 / 0.269 | +0.431\* / +0.215\* vs zero-shot | +0.092\* / −0.219\* | Well above chance, still **below newest-first** (0.560 / 0.515) |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.036 / 0.039 | 0.001 / 0.000 | 0.001 / 0.000 | −0.485\* / −0.269\* | 0.000 / 0.000 | Same total collapse: never picks the correct article once it looks old |
| A3 Hide | all dates removed | 0.024 / 0.025 | 0.014 / 0.013 | 0.096 / 0.084 | −0.390\* / −0.185\* | +0.083\* / +0.071\* | About chance; slightly above the old model's near-zero |
| A4 Shuffle | dates randomly reassigned among candidates | 0.043 / 0.040 | 0.035 / 0.031 | 0.069 / 0.070 | −0.417\* / −0.199\* | +0.034\* / +0.039\* | Chance: wrong dates break it |
| A6 All-new | all 15 candidates are new articles | 0.094 / 0.062 | 0.097 / 0.117 | 0.157 / 0.123 | −0.323\* / −0.126\* | +0.060\* / +0.006 | Near chance (0.067): little content signal |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.065 | 0.241 | 0.132 | — | −0.109\* | Worse than the old model at finding an *old* correct article |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.059 | 0.106 | 0.051 | −0.081\* | −0.055\* | Traps win **89.8%** of top spots (old model 82.8%, zero-shot 18.3%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled new | 0.059 | 0.104 | 0.045 | −0.087\* | −0.059\* | Trap@1 0.911 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.059 | 0.107 | 0.042 | −0.090\* | −0.065\* | … 0.922 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.053 | 0.113 | 0.030 | −0.102\* | −0.083\* | … 0.944 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.049 | 0.110 | 0.028 | −0.104\* | −0.082\* | … 0.971: near-total |

**Summary.** The test set confirms dev on every row. Production-style (impression) negatives do **not** remove the recency shortcut; the model leans on it slightly more. Swapping dates gives 0, shuffling or hiding them gives about chance, all-new pools are near chance, and new-looking traps take 90–97% of top spots (old model 83–89%). The new model is better on random-negative Baseline (+0.092\*) but much worse on hard negatives (−0.219\*), and in neither case beats newest-first.

Caveats: learning rate 1e-4 vs the old model's 2e-4; single seed; positives are cold clicks only, so "old ⇒ No" is still a perfect rule in training.

## Qwen2.5-7B fine-tuned on EB-NeRD (news) — dev set (2026-10-29)

**Model.** Qwen2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…`), **fine-tuned on EB-NeRD** with the Amazon Qwen recipe (QLoRA 4-bit NF4, fp16 compute, embeddings kept fp16, LoRA r16 / α32, lr 2e-4, 1 epoch, pointwise Yes/No, dates shown, effective batch 32 as 4 GPUs × 2 × 4). Same training data as the Llama-2 EB-NeRD pool model: 30,018 examples from training pools at 7 earlier cutoffs. Training: job 45076, 4 h 47 min, no NaN steps, final validation loss 0.203. **Zero-shot** = the same base weights without fine-tuning (job 45079); fine-tuned evaluation: job 45540 (the first attempt, 45077, stopped at the pre-run check; see design.md).

**Evaluation.** As for the Amazon Qwen runs: both models loaded 4-bit (vLLM bitsandbytes), so zero-shot vs fine-tuned is like-for-like, but absolute numbers are not strictly comparable with the fp16 Llama-2 rows. EB-NeRD dev, the same pools as the Llama-2 EB-NeRD sections: 150 users per cell (A6 131), K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking for both models. **Change** = the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. Trap@1 chance 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.040 / 0.020 | 0.533 / 0.507 | +0.493\* / +0.487\* vs zero-shot | Large gain (8× chance), but still **below newest-first** (0.567 / 0.540) |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.047 / 0.040 | 0.000 / 0.000 | −0.533\* / −0.507\* | All of the accuracy moves with the date |
| A3 Hide | all dates removed | 0.073 / 0.060 | 0.020 / 0.020 | −0.513\* / −0.487\* | Below chance without dates |
| A4 Shuffle | dates randomly reassigned among candidates | 0.060 / 0.040 | 0.080 / 0.053 | −0.453\* / −0.453\* | Chance: wrong dates break it |
| A6 All-new | all 15 candidates are new articles | 0.137 / 0.122 | 0.214 / 0.221 | −0.328\* / −0.298\* | About 3× chance: some content signal (more than Llama-2's 0.115 / 0.084), but it loses 60% of its accuracy without the date cue |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.040 | 0.213 | — | Reference for the rows below |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.060 | 0.080 | −0.133\* | Traps take the top spot 87% of the time (zero-shot 10%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled new | 0.060 | 0.087 | −0.127\* | Trap@1 0.873 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.060 | 0.087 | −0.127\* | … 0.880 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.060 | 0.087 | −0.127\* | … 0.887 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.087 | 0.093 | −0.120\* | … 0.907 |

**Summary.** Fine-tuned Qwen2.5-7B on news repeats the Llama-2 pattern: 0.53 / 0.51 under Li et al.'s protocol, yet below newest-first; 0 when the correct article's date is swapped; below chance with dates hidden and about chance with dates shuffled; new-looking traps win 87–91% of top spots. With only new candidates (A6) it keeps about 3× chance, so it learned a little more content than Llama-2 did on news.
- **Zero-shot:** at or below chance on Baseline (0.040 / 0.020) and blind to dates: no A2/A4 change is significant (A3 hard +0.040\* goes *up* without dates). As on Amazon it avoids new items on content: A6 rises to 0.137 / 0.122 (+0.092\* / +0.099\*), and Trap@1 is 0.10 on A1.
- Dev only (n = 150); test (45080 zero-shot, 45541 fine-tuned) pending.

## Qwen2.5-7B fine-tuned on EB-NeRD (news) — test set (2026-10-29)

**Model.** The same fine-tuned Qwen2.5-7B-Instruct as in the dev section above (job 45076), with the same 4-bit evaluation setting. Evaluation jobs: fine-tuned 46038 (resubmission of 45541, which failed on a full node-local disk before scoring anything), zero-shot 45080.

**Evaluation.** EB-NeRD test, the same pools as the Llama-2 EB-NeRD test sections: 800 users per cold-target cell (Baseline, A2–A4), 650 for A6, 1,000 per old-target cell (A1-ctrl, A1, A5). K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking for both models. **Change** = the fine-tuned model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. Trap@1 chance 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, fine-tuned | Change (fine-tuned) | What the change means |
|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.044 / 0.036 | 0.492 / 0.539 | +0.449\* / +0.502\* vs zero-shot | 7–8× chance. Below newest-first on random negatives (0.560); on hard negatives just above it (0.515) |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.062 / 0.050 | 0.000 / 0.001 | −0.492\* / −0.537\* | All of the accuracy moves with the date |
| A3 Hide | all dates removed | 0.058 / 0.043 | 0.011 / 0.010 | −0.481\* / −0.529\* | Below chance without dates |
| A4 Shuffle | dates randomly reassigned among candidates | 0.054 / 0.041 | 0.048 / 0.043 | −0.445\* / −0.496\* | Chance: wrong dates break it |
| A6 All-new | all 15 candidates are new articles | 0.143 / 0.091 | 0.214 / 0.222 | −0.286\* / −0.317\* | About 3× chance: some content signal (Llama-2 on news: 0.097 / 0.117), but it loses about 60% of its accuracy without the date cue |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.061 | 0.249 | — | Reference for the rows below |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.073 | 0.087 | −0.162\* | Traps take the top spot 86% of the time (zero-shot 7%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled new | 0.073 | 0.085 | −0.164\* | Trap@1 0.872 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.073 | 0.087 | −0.162\* | … 0.874 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.075 | 0.084 | −0.165\* | … 0.881 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.103 | 0.088 | −0.161\* | … 0.909 |

**Summary.** Test replicates dev in every row (same signs, similar sizes, every fine-tuned change p < 1e-35).
- **Fine-tuned:** 0.49 / 0.54 under Li et al.'s protocol, but 0 when the correct article's date is swapped, below chance with dates hidden, chance with dates shuffled, and new-looking traps win 86–91% of top spots. With only new candidates it keeps about 3× chance, a little more content than Llama-2 learned on news.
- **Zero-shot:** at chance on Baseline (0.044 / 0.036) and not using dates the way the fine-tuned model does: swapping dates *raises* it slightly (+0.019\* / +0.014\*), A6 rises (+0.089\* / +0.046\*), and Trap@1 is 0.07 on A1.
- **With this section the recency shortcut is shown on test for both base models (Llama-2-7B, Qwen2.5-7B) on both datasets (Amazon VG, EB-NeRD).**

## Qwen2.5-7B fine-tuned on EB-NeRD with impression negatives — dev set (2026-10-30)

**Model.** Qwen2.5-7B-Instruct, **fine-tuned on EB-NeRD with the impression-negative training data** used for the Llama-2 impression model: 5,250 cold clicks with up to 5 shown-but-not-clicked articles each, 27,919 training examples; the correct article is newer than all its negatives in only 12.7% of cases.
- Recipe identical to the Qwen EB-NeRD pool-data model (lr 2e-4, QLoRA 4-bit, 1 epoch, pointwise Yes/No, dates shown, batch 32); only the negatives differ. Unlike Llama-2, it did not diverge at 2e-4, so there is no learning-rate confound here.
- Training: job 46950, 4 h 15 min, no NaN steps, final validation loss 0.423. Evaluation: job 46951 (pointwise check passed on 8/8 prompts, max difference 0.031).
- **Old FT** = the Qwen pool-data fine-tune (`qwen25_7b_ft_ebnerd`) from the Qwen EB-NeRD dev section above. **Zero-shot** is reused from that section (job 45079).

**Evaluation.** As in the Qwen EB-NeRD dev section: both models 4-bit, 150 users per cell (A6 131), K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the new model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. Trap@1 chance 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, old FT (pool data) | Recall@1, new FT (impressions) | Change (new FT) | What the change means |
|---|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.040 / 0.020 | 0.533 / 0.507 | 0.407 / 0.167 | +0.367\* / +0.147\* vs zero-shot | Lower than the old model (−0.127\* / −0.340\*) and well **below newest-first** (0.567 / 0.540) |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.047 / 0.040 | 0.000 / 0.000 | 0.000 / 0.000 | −0.407\* / −0.167\* | Unchanged: it never picks the correct article once it looks old |
| A3 Hide | all dates removed | 0.073 / 0.060 | 0.020 / 0.020 | 0.067 / 0.053 | −0.340\* / −0.113\* | Chance (old vs new n.s.) |
| A4 Shuffle | dates randomly reassigned among candidates | 0.060 / 0.040 | 0.080 / 0.053 | 0.053 / 0.067 | −0.353\* / −0.100\* | Chance: wrong dates break it, as before |
| A6 All-new | all 15 candidates are new articles | 0.137 / 0.122 | 0.214 / 0.221 | 0.084 / 0.053 | −0.321\* / −0.107\* | Falls to chance: the old model's content signal is gone (−0.130\* / −0.168\* vs old) |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.040 | 0.213 | 0.160 | — | Slightly worse than the old model (−0.053, n.s.) |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.060 | 0.080 | 0.027 | −0.133\* | Traps win **89%** of top spots (old model 87%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled new | 0.060 | 0.087 | 0.027 | −0.133\* | Trap@1 0.900 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.060 | 0.087 | 0.027 | −0.133\* | … 0.913 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.060 | 0.087 | 0.027 | −0.133\* | … 0.927 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.087 | 0.093 | 0.013 | −0.147\* | … 0.987: near-total |

**Summary.** Same result as for Llama-2: production-style negatives do **not** remove the recency shortcut. Swapping the dates still gives 0, hiding or shuffling them gives chance, and new-looking traps take 89–99% of top spots. Here it also costs accuracy: the baseline drops (hard negatives 0.507 → 0.167), and the all-new A6 score, the one place the pool-data Qwen showed some content signal, falls to chance. So the impression model is less useful and no less date-driven.
- Same likely reason as for Llama-2: in training the correct article is always new, while some negatives are old, so "old means No" is still a perfect rule.
- Dev only (n = 150); the test set (next section) confirms it, except that A6 stays above chance on random negatives.

## Qwen2.5-7B fine-tuned on EB-NeRD with impression negatives — test set (2026-10-30)

**Model.** The impression-negative model from the dev section above (lr 2e-4, job 46950), with the same 4-bit evaluation setting. Test job 46952 (6 h 38 min, pointwise check passed on 8/8 prompts, max difference 0.031). **Old FT** = the Qwen pool-data fine-tune `qwen25_7b_ft_ebnerd`; **zero-shot** = `qwen25_7b_fp16`, both on the same test pools.

**Evaluation.** EB-NeRD test: 800 users per cold-target cell (Baseline, A2–A4), 650 for A6, 1,000 per old-target cell (A1-ctrl, A1, A5). K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the new model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); \* = McNemar p < 0.05. Trap@1 chance 0.20.

| Ablation | What it does | Recall@1, zero-shot | Recall@1, old FT (pool data) | Recall@1, new FT (impressions) | Change (new FT) | What the change means |
|---|---|---|---|---|---|---|
| Baseline | Li et al.'s protocol: correct article is new; dates shown | 0.044 / 0.036 | 0.492 / 0.539 | 0.406 / 0.194 | +0.362\* / +0.158\* vs zero-shot | Lower than the old model (−0.086\* / −0.345\*) and well **below newest-first** (0.560 / 0.515) |
| A2 Swap | correct article's date swapped with an old wrong one's | 0.062 / 0.050 | 0.000 / 0.001 | 0.003 / 0.003 | −0.404\* / −0.191\* | Unchanged: it almost never picks the correct article once it looks old |
| A3 Hide | all dates removed | 0.058 / 0.043 | 0.011 / 0.010 | 0.048 / 0.036 | −0.359\* / −0.158\* | Below chance, though less far below than the old model (+0.036\* / +0.026\* vs old) |
| A4 Shuffle | dates randomly reassigned among candidates | 0.054 / 0.041 | 0.048 / 0.043 | 0.054 / 0.048 | −0.352\* / −0.146\* | Chance: wrong dates break it, as before (old vs new n.s.) |
| A6 All-new | all 15 candidates are new articles | 0.143 / 0.091 | 0.214 / 0.222 | 0.128 / 0.088 | −0.278\* / −0.105\* | About 2× chance on random, near chance on hard; no better than zero-shot and below the old model (−0.086\* / −0.134\*) |
| A1-ctrl (old-target reference) | correct article is old; nothing new-looking | 0.061 | 0.249 | 0.176 | — | Worse than the old model (−0.073\*) at finding an *old* correct article |
| A1 Traps | correct article is old, plus 3 new irrelevant articles | 0.073 | 0.087 | 0.039 | −0.137\* | Traps win **89%** of top spots (old model 86%, chance 20%) |
| A5 Flood, n = 1 | A1 + 1 old wrong article relabelled new | 0.073 | 0.085 | 0.032 | −0.144\* | Trap@1 0.907 … |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.073 | 0.087 | 0.028 | −0.148\* | … 0.922 … |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.075 | 0.084 | 0.020 | −0.156\* | … 0.945 … |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.103 | 0.088 | 0.016 | −0.160\* | … 0.980: near-total |

**Summary.** Test confirms dev: impression negatives do **not** remove the recency shortcut in Qwen either. Swapping the dates gives about 0, hiding or shuffling them gives chance or below, and new-looking traps take 89–98% of top spots (old model 86–91%).
- **It also costs accuracy.** The baseline falls, most on hard negatives (0.539 → 0.194), and every old-target row is significantly lower than the old model. The new model is further below newest-first than the old one.
- **Correction to dev:** on dev, A6 looked like chance on both kinds of negatives (0.084 / 0.053, n = 131). On test it is 0.128 on random negatives, about 2× chance, and 0.088 on hard. Either way it is no better than zero-shot (0.143 / 0.091) and below the pool-data model (0.214 / 0.222), so what the pool-data model learned on A6 did not survive the change of negatives.
- **Comparison with Llama-2:** both impression models keep the shortcut and lose on hard negatives (Llama-2 −0.219\*, Qwen −0.345\*). They differ on random negatives (Llama-2 +0.092\*, Qwen −0.086\*). Qwen used the same lr as its pool-data model, so its change comes from the negatives alone; Llama-2's also includes the lr change.
- Single seed.

## Correction: Trap@1 chance on the A5 rows (2026-10-30)

The sections above give "Trap@1 chance 0.20". That holds only for **A1** (3 trap items out of 15). In A5 the relabelled items also count as traps (`TRAP_ROLES` in `eval/metrics.py`), so chance rises with n: **n = 1: 0.267, n = 2: 0.333, n = 5: 0.533, n = 10: 0.867** (counted from the EB-NeRD test pools: 4, 5, 8 and 13 of 15). So the rise in Trap@1 across A5 is partly just more traps in the pool, and Trap@1 at n = 10 is a weak signal, e.g. "0.98: near-total" is only modestly above 0.867. A1 is the row to cite. The Recall@1 numbers are not affected.

## Llama-2-7B dates-hidden control, fine-tuned on Amazon Video Games — dev set (2026-10-30)

**Model.** Llama-2-7B-Chat fine-tuned on Amazon VG with the same recipe, pools, users and holdout as the date-trained model in the first section, except that **no training prompt shows a date** (23,778 examples; job 48200, 1 h 25 min, no NaN steps, final validation loss 0.283, date-trained 0.240). Evaluation job 48742 (16 min, pointwise check passed on 8/8 prompts, max difference 0.016). **Date-trained** = the model from the first section.

**Evaluation.** Amazon VG dev, the same pools as the first section: 150 users per cold-target cell, 144 per old-target cell. K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the control's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); **vs date-trained** = control minus date-trained on the same pools; \* = McNemar p < 0.05. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. **Only A3 matches the format the control was trained on**; every other row shows it dates it never saw in training.

| Ablation | What it does | Recall@1, date-trained | Recall@1, control | Change (control) | vs date-trained | Trap@1, date-trained → control |
|---|---|---|---|---|---|---|
| Baseline | correct item is new; dates shown | 0.373 / 0.460 | 0.327 / 0.313 | — | −0.047 / −0.147\* | — |
| A2 Swap | correct item's date swapped with an old wrong one's | 0.007 / 0.007 | 0.107 / 0.107 | −0.220\* / −0.207\* | +0.100\* / +0.100\* | — |
| A3 Hide | all dates removed | 0.093 / 0.087 | 0.227 / 0.260 | −0.100\* / −0.053 | +0.133\* / +0.173\* | — |
| A4 Shuffle | dates randomly reassigned among candidates | 0.093 / 0.060 | 0.207 / 0.207 | −0.120\* / −0.107\* | +0.113\* / +0.147\* | — |
| A6 All-new | all 15 candidates are new | 0.240 / 0.280 | 0.187 / 0.253 | −0.140\* / −0.060 | −0.053 / −0.027 | — |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.389 | 0.347 | — | −0.042 | 0.104 → 0.076 |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.139 | 0.090 | −0.257\* | −0.049 | 0.528 → 0.472 |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled new | 0.125 | 0.090 | −0.257\* | −0.035 | 0.632 → 0.549 |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.083 | 0.090 | −0.257\* | +0.007 | 0.688 → 0.569\* |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.062 | 0.076 | −0.271\* | +0.014 | 0.826 → 0.736\* |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.042 | 0.076 | −0.271\* | +0.035 | 0.951 → 0.903 |

Newest-first on these pools: 0.347 / 0.307.

**Summary.** Training without dates weakens the shortcut but does not remove it.
- **Without dates it keeps real skill:** A3 is 0.227 / 0.260, against 0.093 / 0.087 for the date-trained model. So the date-trained A3 row understates how much the training teaches beyond dates.
- **It still follows dates it never saw in training:** swapping the dates costs it −0.220\* / −0.207\*, and new-looking traps take 47% of top spots on A1 (chance 20%). Why is not established. The format shift may play a part, but the swap result shows the displayed dates themselves move it.
- **It costs accuracy where dates are real:** the Baseline falls on hard negatives (−0.147\*), and the control only matches newest-first.
- Dev only (n = 150 / 144), single seed; test pending (job 48743).

## Llama-2-7B dates-hidden control, fine-tuned on Amazon Video Games — test set (2026-10-31)

**Model.** The dates-hidden control from the dev section above (job 48200). Evaluation job 48743 (1 h 14 min, pointwise check passed on 8/8 prompts, max difference 0.016). **Date-trained** = the model from the Amazon test section (job 40143, evaluation 41008).

**Evaluation.** Amazon VG test, the same pools as the date-trained test section: 800 users per cold-target cell (A6 799), 967 per old-target cell. K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the control's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); **vs date-trained** = control minus date-trained on the same pools; \* = McNemar p < 0.05. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. **Only A3 matches the format the control was trained on**; every other row shows it dates it never saw in training.

| Ablation | What it does | Recall@1, date-trained | Recall@1, control | Change (control) | vs date-trained | Trap@1, date-trained → control |
|---|---|---|---|---|---|---|
| Baseline | correct item is new; dates shown | 0.445 / 0.546 | 0.299 / 0.345 | — | −0.146\* / −0.201\* | — |
| A2 Swap | correct item's date swapped with an old wrong one's | 0.005 / 0.006 | 0.084 / 0.100 | −0.215\* / −0.245\* | +0.079\* / +0.094\* | — |
| A3 Hide | all dates removed | 0.101 / 0.107 | 0.234 / 0.282 | −0.065\* / −0.062\* | +0.133\* / +0.175\* | — |
| A4 Shuffle | dates randomly reassigned among candidates | 0.077 / 0.077 | 0.204 / 0.217 | −0.095\* / −0.128\* | +0.126\* / +0.140\* | — |
| A6 All-new | all 15 candidates are new | 0.253 / 0.292 | 0.193 / 0.278 | −0.106\* / −0.068\* | −0.060\* / −0.014 | — |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.384 | 0.331 | — | −0.053\* | 0.127 → 0.155\* |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.152 | 0.126 | −0.205\* | −0.026\* | 0.562 → 0.501\* |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled new | 0.103 | 0.118 | −0.213\* | +0.014 | 0.675 → 0.555\* |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.084 | 0.116 | −0.215\* | +0.032\* | 0.737 → 0.602\* |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.067 | 0.105 | −0.225\* | +0.038\* | 0.845 → 0.744\* |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.060 | 0.098 | −0.233\* | +0.038\* | 0.930 → 0.883\* |

Newest-first on these pools: 0.346 / 0.323. Control minus newest-first: −0.048 (p = 0.05) / +0.022 (n.s.).

**Summary.** Test confirms dev with more power: training without dates weakens the recency shortcut but does not remove it, and it costs accuracy.
- **Real skill without dates:** on A3 the control reaches 0.234 / 0.282, against 0.101 / 0.107 for the date-trained model. That is about 45% of the date-trained model's Baseline gain above chance. So the date-trained A3 row (a model tested on a format it never saw) understates what the training teaches beyond dates.
- **The shortcut remains:** swapping dates takes it to about chance (0.084 / 0.100), and new-looking traps take 50% of top spots on A1 (date-trained 56%, chance 20%). The Trap@1 reductions are significant on every A1 and A5 row, but it stays well above chance on A1 and on A5 up to n = 5 (A5 chance rises with n).
- **It reacts to dates it never saw in training.** Showing correct dates raises it above A3 (+0.065\* / +0.062\*), and swapped dates lower it well below A3. The zero-shot model barely reacts to dates, so this date sensitivity comes with fine-tuning even when the training prompts contain no dates. Why is not established. Titles are unchanged, so a year inside a title could still carry date information during training; this has not been tested.
- **The cost:** Baseline falls by −0.146\* / −0.201\*, and the control no longer beats newest-first (−0.048 at p = 0.05 / +0.022 n.s.), whereas the date-trained model does (0.445 / 0.546 vs 0.346 / 0.323).
- Hiding dates in training is a diagnostic control and not a fix: production prompts show dates, and the model still follows them. Single seed.

## Llama-2-7B with old correct answers mixed into training (the fix), Amazon Video Games — dev set (2026-10-31)

**Model.** Llama-2-7B-Chat fine-tuned on Amazon VG with the date-trained recipe and **dates shown**, but half of each cutoff's training pools have an **old** correct item (4,183 pools: 2,092 new-answer, 2,091 old-answer; 23,802 training examples). In the training file a new item is correct 20.0% of the time and an old item 14.3% (date-trained data: 33% and 0%). Job 49600, 1 h 41 min, no NaN steps, final validation loss 0.335 (date-trained 0.240, on different validation data). Evaluation job 49601 (pointwise check 8/8, max difference 0.016). **Date-trained** = the model from the first section.

**Evaluation.** Amazon VG dev, the same pools as the first section: 150 users per cold-target cell, 144 per old-target cell. K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the fix's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); **vs date-trained** = fix minus date-trained on the same pools; \* = McNemar p < 0.05. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867.

| Ablation | What it does | Recall@1, date-trained | Recall@1, fix | Change (fix) | vs date-trained | Trap@1, date-trained → fix |
|---|---|---|---|---|---|---|
| Baseline | correct item is new; dates shown | 0.373 / 0.460 | 0.267 / 0.260 | — | −0.107\* / −0.200\* | — |
| A2 Swap | correct item's date swapped with an old wrong one's | 0.007 / 0.007 | 0.080 / 0.080 | −0.187\* / −0.180\* | +0.073\* / +0.073\* | — |
| A3 Hide | all dates removed | 0.093 / 0.087 | 0.127 / 0.147 | −0.140\* / −0.113\* | +0.033 / +0.060 | — |
| A4 Shuffle | dates randomly reassigned among candidates | 0.093 / 0.060 | 0.100 / 0.093 | −0.167\* / −0.167\* | +0.007 / +0.033 | — |
| A6 All-new | all 15 candidates are new | 0.240 / 0.280 | 0.207 / 0.313 | −0.060 / +0.053 | −0.033 / +0.033 | — |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.389 | 0.326 | — | −0.062 | 0.104 → 0.146 |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.139 | 0.236 | −0.090\* | +0.097\* | 0.528 → **0.271\*** |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled new | 0.125 | 0.236 | −0.090\* | +0.111\* | 0.632 → 0.333\* |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.083 | 0.188 | −0.139\* | +0.104\* | 0.688 → 0.451\* |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.062 | 0.160 | −0.167\* | +0.097\* | 0.826 → 0.667\* |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.042 | 0.146 | −0.181\* | +0.104\* | 0.951 → 0.819\* |

Newest-first on these pools: 0.347 / 0.307.

**Summary.** Mixing old correct answers into training sharply reduces the pull of new-looking items, but the model still depends on dates when the correct item is new.
- **Traps lose most of their pull:** A1 Trap@1 falls from 0.528 to 0.271 (chance 0.20), and Recall@1 on the same pools rises from 0.139 to 0.236. A5 improves on every row; at n = 10 Trap@1 (0.819) is below chance (0.867).
- **Relevance signals survive:** A6 (0.207 / 0.313) and A1-ctrl (0.326) are not significantly below the date-trained model.
- **Dates still matter:** swap, hide and shuffle still drop the Baseline significantly, and only A2 improves against the date-trained model. A3 and A4 are not significantly better.
- **The cost:** Baseline falls to 0.267 / 0.260, below newest-first. Some of that drop is the shortcut's inflated score going away, but this run cannot separate it from lost skill.
- Dev only (n = 150 / 144), single seed; test pending (job 49882).

## Llama-2-7B with old correct answers mixed into training (the fix), Amazon Video Games — test set (2026-10-31)

**Model.** The fix model from the dev section above (job 49600). Evaluation job 49882 (1 h 38 min, pointwise check passed on 8/8 prompts, max difference 0.016). **Date-trained** = the model from the Amazon test section (job 40143, evaluation 41008).

**Evaluation.** Amazon VG test, the same pools as the date-trained test section: 800 users per cold-target cell (A6 799), 967 per old-target cell. K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the fix's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); **vs date-trained** = fix minus date-trained on the same pools; \* = McNemar p < 0.05. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867.

| Ablation | What it does | Recall@1, date-trained | Recall@1, fix | Change (fix) | vs date-trained | Trap@1, date-trained → fix |
|---|---|---|---|---|---|---|
| Baseline | correct item is new; dates shown | 0.445 / 0.546 | 0.296 / 0.334 | — | −0.149\* / −0.212\* | — |
| A2 Swap | correct item's date swapped with an old wrong one's | 0.005 / 0.006 | 0.068 / 0.091 | −0.229\* / −0.242\* | +0.062\* / +0.085\* | — |
| A3 Hide | all dates removed | 0.101 / 0.107 | 0.164 / 0.191 | −0.133\* / −0.142\* | +0.062\* / +0.084\* | — |
| A4 Shuffle | dates randomly reassigned among candidates | 0.077 / 0.077 | 0.126 / 0.136 | −0.170\* / −0.198\* | +0.049\* / +0.059\* | — |
| A6 All-new | all 15 candidates are new | 0.253 / 0.292 | 0.194 / 0.273 | −0.103\* / −0.061\* | −0.059\* / −0.019 | — |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.384 | 0.308 | — | −0.075\* | 0.127 → 0.147 |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.152 | 0.236 | −0.072\* | +0.084\* | 0.562 → **0.278\*** |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled new | 0.103 | 0.226 | −0.082\* | +0.123\* | 0.675 → 0.373\* |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.084 | 0.214 | −0.094\* | +0.130\* | 0.737 → 0.465\* |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.067 | 0.180 | −0.128\* | +0.113\* | 0.845 → 0.646\* |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.060 | 0.168 | −0.141\* | +0.108\* | 0.930 → 0.810\* |

Newest-first on these pools: 0.346 / 0.323. Fix minus newest-first: −0.050\* (p = 0.03) / +0.011 (n.s.).

**Against the dates-hidden control** (test section above, same pools): Baseline is the same (−0.003 / −0.011, n.s.), Trap@1 is much lower on A1 (0.278 vs 0.501\*) and lower on every A5 row\*, and A3 is lower (0.164 / 0.191 vs 0.234 / 0.282\*).

**Summary.** Test confirms dev: the fix reduces the recency shortcut a lot but does not remove it, and it costs accuracy.
- **Traps lose most of their pull:** A1 Trap@1 falls from 0.562 to 0.278, so the excess over chance (0.20) falls from 0.362 to 0.078. Traps now cost −0.072\* in Recall@1, against −0.232 for the date-trained model. Trap@1 is still significantly above chance on A1 and on A5 up to n = 5 (each p < 1e-8), and below chance at n = 10 (0.810 vs 0.867, p < 1e-6).
- **Dates still dominate when the correct item is new:** swapping the dates takes it to about chance (0.068 / 0.091), and hiding or shuffling them still costs −0.133\* to −0.198\*.
- **Better without dates, unlike dev:** A3 and A4 are significantly above the date-trained model on test (they were not on dev, with n = 150).
- **Relevance signals drop a little, also unlike dev:** A1-ctrl is −0.075\* and A6 is −0.059\* / −0.019 against the date-trained model. Dev's "not significantly below" did not hold at the larger n, except for A6 on hard negatives.
- **The cost:** Baseline falls by −0.149\* / −0.212\*, and the fix does not beat newest-first (−0.050\* / +0.011 n.s.). Some of that drop is the shortcut's inflated score going away, but this run cannot separate it from lost skill.
- **Not fully date-neutral by design:** in the training file a new item is still correct more often than an old one (20.0% vs 14.3%), so some remaining preference for new items is expected. Single seed, Amazon only.

## Llama-2-7B date-trained model across three training seeds — test set, both datasets (2026-11-01)

**Models.** The date-trained Llama-2-7B-Chat on each dataset, seed 42 (the models in the earlier test sections) plus new seeds 43 and 44. The SFT data and all settings are the same, so only training randomness varies (LoRA initialisation, shuffling, dropout), not the data sampling. Amazon: training 49603, 49605; evaluation 49604, 49606. EB-NeRD: training 49607, 49609; evaluation 49608, 51516. All pointwise checks passed on 8/8 prompts (max difference 0.016).

**Evaluation.** The same test pools as the earlier sections. Amazon: 800 users per cold-target cell (A6 799), 967 per old-target cell. EB-NeRD: 800 per cold-target cell (A6 650), 1,000 per old-target cell. K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. Each cell gives seeds 42, 43, 44, then the mean and standard deviation (SD) of the three. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867.

### Amazon Video Games

| Ablation | Random: 42, 43, 44 | Random: mean ± SD | Hard: 42, 43, 44 | Hard: mean ± SD |
|---|---|---|---|---|
| Baseline | 0.445, 0.430, 0.459 | 0.445 ± 0.014 | 0.546, 0.516, 0.535 | 0.532 ± 0.015 |
| A2 Swap | 0.005, 0.005, 0.006 | 0.005 ± 0.001 | 0.006, 0.007, 0.007 | 0.007 ± 0.001 |
| A3 Hide | 0.101, 0.104, 0.119 | 0.108 ± 0.009 | 0.107, 0.111, 0.130 | 0.116 ± 0.012 |
| A4 Shuffle | 0.077, 0.085, 0.092 | 0.085 ± 0.007 | 0.077, 0.070, 0.087 | 0.078 ± 0.009 |
| A6 All-new | 0.253, 0.233, 0.267 | 0.251 ± 0.017 | 0.292, 0.275, 0.302 | 0.290 ± 0.013 |

| Ablation | Recall@1: 42, 43, 44 | Recall@1: mean ± SD | Trap@1: 42, 43, 44 | Trap@1: mean ± SD |
|---|---|---|---|---|
| A1-ctrl | 0.384, 0.363, 0.374 | 0.374 ± 0.010 | 0.127, 0.126, 0.114 | 0.122 ± 0.007 |
| A1 Traps | 0.152, 0.156, 0.150 | 0.153 ± 0.003 | 0.562, 0.551, 0.567 | 0.560 ± 0.008 |
| A5 n = 1 | 0.103, 0.112, 0.105 | 0.107 ± 0.004 | 0.675, 0.658, 0.669 | 0.667 ± 0.009 |
| A5 n = 2 | 0.084, 0.097, 0.092 | 0.091 ± 0.007 | 0.737, 0.713, 0.725 | 0.725 ± 0.012 |
| A5 n = 5 | 0.067, 0.075, 0.074 | 0.072 ± 0.005 | 0.845, 0.842, 0.828 | 0.838 ± 0.009 |
| A5 n = 10 | 0.060, 0.063, 0.058 | 0.060 ± 0.003 | 0.930, 0.926, 0.932 | 0.929 ± 0.003 |

Newest-first: 0.346 / 0.323. Seed minus newest-first: random +0.099\*, +0.084\*, +0.112\*; hard +0.224\*, +0.194\*, +0.213\*.

### EB-NeRD (news)

| Ablation | Random: 42, 43, 44 | Random: mean ± SD | Hard: 42, 43, 44 | Hard: mean ± SD |
|---|---|---|---|---|
| Baseline | 0.394, 0.379, 0.420 | 0.397 ± 0.021 | 0.487, 0.436, 0.448 | 0.457 ± 0.027 |
| A2 Swap | 0.001, 0.000, 0.001 | 0.001 ± 0.001 | 0.000, 0.001, 0.001 | 0.001 ± 0.001 |
| A3 Hide | 0.014, 0.004, 0.036 | 0.018 ± 0.017 | 0.013, 0.004, 0.035 | 0.017 ± 0.016 |
| A4 Shuffle | 0.035, 0.031, 0.036 | 0.034 ± 0.003 | 0.031, 0.036, 0.040 | 0.036 ± 0.004 |
| A6 All-new | 0.097, 0.078, 0.123 | 0.099 ± 0.022 | 0.117, 0.078, 0.126 | 0.107 ± 0.025 |

| Ablation | Recall@1: 42, 43, 44 | Recall@1: mean ± SD | Trap@1: 42, 43, 44 | Trap@1: mean ± SD |
|---|---|---|---|---|
| A1-ctrl | 0.241, 0.199, 0.300 | 0.247 ± 0.051 | 0.202, 0.154, 0.187 | 0.181 ± 0.025 |
| A1 Traps | 0.106, 0.101, 0.078 | 0.095 ± 0.015 | 0.828, 0.786, 0.865 | 0.826 ± 0.040 |
| A5 n = 1 | 0.104, 0.096, 0.073 | 0.091 ± 0.016 | 0.846, 0.823, 0.874 | 0.848 ± 0.026 |
| A5 n = 2 | 0.107, 0.101, 0.074 | 0.094 ± 0.018 | 0.850, 0.834, 0.877 | 0.854 ± 0.022 |
| A5 n = 5 | 0.113, 0.096, 0.073 | 0.094 ± 0.020 | 0.858, 0.863, 0.885 | 0.869 ± 0.014 |
| A5 n = 10 | 0.110, 0.091, 0.071 | 0.091 ± 0.020 | 0.888, 0.903, 0.924 | 0.905 ± 0.018 |

Newest-first: 0.560 / 0.515. Seed minus newest-first: random −0.166\*, −0.181\*, −0.140\*; hard −0.028 (n.s.), −0.079\*, −0.068\*.

**Summary.** The shortcut findings do not depend on the training seed.
- **Every drop replicates:** on every seed and both datasets, A2, A3, A4 and A6 are significantly below the Baseline, and A1 and all A5 rows are significantly below A1-ctrl (McNemar p < 0.05).
- **Amazon is stable:** Baseline SD is about 0.015, A1 Trap@1 is 0.551–0.567, and every seed beats newest-first significantly on both negatives.
- **EB-NeRD varies more:** A1-ctrl Recall@1 ranges from 0.199 to 0.300, and the cost of traps from −0.098\* to −0.222\*. Report these as a mean and spread, not from one seed.
- **One seed 42 reading does not hold:** on EB-NeRD hard negatives seed 42 ties with newest-first, but seeds 43 and 44 are significantly below it. Across seeds the model does not beat newest-first on EB-NeRD on either negative.
- **The fix and the control lie outside seed variation on Amazon:** their A1 Trap@1 (0.278 and 0.501) and Baseline (0.296 / 0.334 and 0.299 / 0.345) are all below the lowest date-trained seed (0.551; 0.430 / 0.516). The fix and control themselves are still one seed each.
- **EB-NeRD seed 44:** its training logged one non-finite grad norm at step 200 and one validation-loss spike at step 700, and ended at a validation loss matching the other seeds (0.199; seed 42 0.200, seed 43 0.198). It has the highest A1-ctrl Recall@1 and the highest Trap@1 on A1 and A5 of the three. Whether that comes from those events or ordinary seed variation is not established; every direction above is the same without it.
- **Scope:** Llama-2-7B date-trained only. Three seeds give a rough measure of spread, and the data sampling was not varied.

## Llama-2-7B Li-style control (old correct items only, no new items in training), Amazon Video Games — dev set (2026-11-03)

**Model.** Llama-2-7B-Chat fine-tuned the way Li et al. build their SFT data: **cold items removed from training**. Every training pool has an old correct item and only old wrong items (gold + 12 warm; the two new negatives of the fix's old-target pools are dropped and replaced by two more warm candidates in the SFT step), so **no training example shows a new item**. Same pool count (4,183), same settings and seed as the date-trained model, dates shown. 23,820 train / 1,278 validation examples. Training: job 53780 hit the ADA home quota at the final save (step 740/744); resumed from checkpoint-700 as job 54245 (final validation loss 0.266, no non-finite steps). Evaluation job 54246 (18 min, pointwise check passed, max difference 0.016). **In these training pools the old correct item is the newest candidate in 50.7% of pools** (chance 7.7%): users buy recently released items even among old ones.

**Evaluation.** As in the fix dev section: 150 users per cold-target cell, 144 per old-target cell, K = 15, Recall@1 as random / hard negatives, chance 0.067. Every pool has a complete ranking. **Change** = the Li-style model's paired change against the reference row (Baseline for A2–A6, A1-ctrl for A1/A5); **vs date-trained** = Li-style minus date-trained on the same pools; \* = McNemar p < 0.05. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867.

| Ablation | What it does | Recall@1, date-trained | Recall@1, fix | Recall@1, Li-style | Change (Li-style) | vs date-trained | Trap@1, date-trained → Li-style |
|---|---|---|---|---|---|---|---|
| Baseline | correct item is new; dates shown | 0.373 / 0.460 | 0.267 / 0.260 | 0.140 / 0.367 | — | −0.233\* / −0.093 | — |
| A2 Swap | correct item's date swapped with an old wrong one's | 0.007 / 0.007 | 0.080 / 0.080 | 0.000 / 0.027 | −0.140\* / −0.340\* | −0.007 / +0.020 | — |
| A3 Hide | all dates removed | 0.093 / 0.087 | 0.127 / 0.147 | 0.173 / 0.313 | +0.033 / −0.053 | +0.080 / +0.227\* | — |
| A4 Shuffle | dates randomly reassigned among candidates | 0.093 / 0.060 | 0.100 / 0.093 | 0.153 / 0.127 | +0.013 / −0.240\* | +0.060 / +0.067 | — |
| A6 All-new | all 15 candidates are new | 0.240 / 0.280 | 0.207 / 0.313 | 0.007 / 0.133 | −0.133\* / −0.233\* | −0.233\* / −0.147\* | — |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.389 | 0.326 | 0.556 | — | +0.167\* | 0.104 → 0.069 |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.139 | 0.236 | 0.062 | −0.493\* | −0.076\* | 0.528 → **0.931\*** |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled new | 0.125 | 0.236 | 0.056 | −0.500\* | −0.069 | 0.632 → 0.938\* |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.083 | 0.188 | 0.056 | −0.500\* | −0.028 | 0.688 → 0.938\* |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.062 | 0.160 | 0.056 | −0.500\* | −0.007 | 0.826 → 0.944\* |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.042 | 0.146 | 0.049 | −0.507\* | +0.007 | 0.951 → 0.951 |

Newest-first on these pools: 0.347 / 0.307 (Baseline), 0.562 (A1-ctrl). Li-style minus newest-first on Baseline: −0.207\* / +0.060 (n.s.).

**Summary.** Removing cold items from training does not remove the date shortcut; it makes the trap failure worse. New-looking irrelevant items take 93% of top spots on A1 (date-trained 53%, fix 27%, chance 20%), and swapping the correct item's date takes it to about 0. With only new candidates (A6) it is at or below chance. Its Baseline is low (0.140 / 0.367) and below newest-first on random negatives. Test below.

## Llama-2-7B Li-style control (old correct items only, no new items in training), Amazon Video Games — test set (2026-11-03)

**Model.** The Li-style model from the dev section above (job 54245). Evaluation job 54247 (1 h 18 min, pointwise check passed, max difference 0.016). **Date-trained** = the model from the Amazon test section (job 40143, evaluation 41008); **fix** = the half-old model (job 49600, evaluation 49882).

**Evaluation.** Amazon VG test, the same pools as the date-trained and fix test sections: 800 users per cold-target cell (A6 799), 967 per old-target cell. Columns and chance levels as in the dev section above.

| Ablation | What it does | Recall@1, date-trained | Recall@1, fix | Recall@1, Li-style | Change (Li-style) | vs date-trained | Trap@1, date-trained → Li-style |
|---|---|---|---|---|---|---|---|
| Baseline | correct item is new; dates shown | 0.445 / 0.546 | 0.296 / 0.334 | 0.152 / 0.297 | — | −0.292\* / −0.249\* | — |
| A2 Swap | correct item's date swapped with an old wrong one's | 0.005 / 0.006 | 0.068 / 0.091 | 0.006 / 0.015 | −0.146\* / −0.282\* | +0.001 / +0.009 | — |
| A3 Hide | all dates removed | 0.101 / 0.107 | 0.164 / 0.191 | 0.125 / 0.250 | −0.028 / −0.048\* | +0.024 / +0.142\* | — |
| A4 Shuffle | dates randomly reassigned among candidates | 0.077 / 0.077 | 0.126 / 0.136 | 0.110 / 0.115 | −0.043\* / −0.182\* | +0.033\* / +0.037\* | — |
| A6 All-new | all 15 candidates are new | 0.253 / 0.292 | 0.194 / 0.273 | 0.024 / 0.094 | −0.128\* / −0.204\* | −0.229\* / −0.198\* | — |
| A1-ctrl (old-target reference) | correct item is old; nothing new-looking | 0.384 | 0.308 | 0.572 | — | +0.188\* | 0.127 → 0.100\* |
| A1 Traps | correct item is old, plus 3 new irrelevant items | 0.152 | 0.236 | 0.039 | −0.533\* | −0.113\* | 0.562 → **0.945\*** |
| A5 Flood, n = 1 | A1 + 1 old wrong item relabelled new | 0.103 | 0.226 | 0.040 | −0.532\* | −0.063\* | 0.675 → 0.950\* |
| A5 Flood, n = 2 | A1 + 2 relabelled | 0.084 | 0.214 | 0.039 | −0.533\* | −0.044\* | 0.737 → 0.953\* |
| A5 Flood, n = 5 | A1 + 5 relabelled | 0.067 | 0.180 | 0.038 | −0.534\* | −0.029\* | 0.845 → 0.960\* |
| A5 Flood, n = 10 | A1 + 10 relabelled | 0.060 | 0.168 | 0.037 | −0.535\* | −0.023\* | 0.930 → 0.963\* |

Newest-first on these pools: 0.346 / 0.323 (Baseline), 0.523 (A1-ctrl). Li-style minus newest-first on Baseline: −0.194\* / −0.025 (n.s.). Li-style minus fix on Baseline: −0.144\* / −0.036 (n.s.); on A1 Trap@1: +0.667\*.

**How often the top-ranked item is simply the newest candidate** (test; for reference, how often the correct item is the newest):

| Pools | Zero-shot | Date-trained | Fix | Li-style | Correct item is newest |
|---|---|---|---|---|---|
| Baseline, random / hard | 0.071 / 0.087 | 0.224 / 0.239 | 0.210 / 0.176 | **0.399 / 0.450** | 0.349 / 0.334 |
| A1-ctrl (old target, all old) | 0.104 | 0.586 | 0.456 | 0.563 | 0.523 |
| A1 Traps | 0.067 | 0.172 | 0.122 | 0.418 | 0.000 |
| A6 All-new, random | 0.068 | 0.038 | 0.118 | 0.131 | 0.076 |

**Summary.** Test confirms dev with more power. Training the way Li et al. do, with no new item anywhere in training, does **not** remove the date shortcut. It changes its form.
- **What it learned is "the newest candidate wins".** Its training pools reward that rule (the old correct item is the newest candidate in 50.7% of pools, chance 7.7%), and it follows it more than any other model: its top pick is the newest candidate in 40–45% of Baseline pools, against 22–24% for the date-trained model and 7–9% zero-shot.
- **Strongest trap failure of all models:** irrelevant new-looking items take **94.5%** of top spots on A1 (date-trained 56%, fix 28%, chance 20%), and 95–96% on every A5 row; Recall@1 under traps falls to 0.04.
- **Dates decide:** swapping the correct item's date takes it to about 0 (0.006 / 0.015); shuffling costs −0.043\* / −0.182\*. With only new candidates (A6) it is at or below chance (0.024 / 0.094), so it has little to go on once dates stop separating the candidates.
- **Old targets without traps:** A1-ctrl 0.572, the best of all models (+0.188\* vs date-trained), but newest-first already gets 0.523 there, because the old correct item is the newest candidate in 52% of these pools.
- **On the cold-start benchmark it scores low:** 0.152 / 0.297, below newest-first on random negatives (−0.194\*) and level on hard (−0.025, n.s.). This matches Li et al.'s report that their fine-tuned models are weak on cold-start items when cold items are excluded from training. Here the standard benchmark is not over-crediting the model; it gives it a low score, but only the diagnostics show why.
- **Reading for the paper:** the recency cue comes from user behaviour (people buy recent releases), not from how we built our pools. Every training set we tried carries it, including Li et al.'s setup, and every fine-tuned model follows it.
- Caveats: single seed, Amazon only. The run resumed from step 700 after the quota failure, so it is not bit-identical to an uninterrupted run. Hard negatives are old by construction, which is why the Baseline score is higher on hard than random negatives for this model.

## Llama-2-7B Li-style control, second training seed (43), Amazon Video Games — test set (2026-11-03)

**Model.** Li-style Llama-2-7B-Chat (old correct items only, no new items in training), seed 43: training job 54991 (1 h 43 min, no non-finite loss, final validation loss 0.271), evaluation job 54992. The SFT data and all settings are the same as the seed-42 model (job 54245, evaluation 54247), so only training randomness varies. Seed 44 is queued (54993/54994); the three-seed mean ± SD will be added when it finishes.

**Evaluation.** The same Amazon VG test pools as the seed-42 section: 800 users per cold-target cell (A6 799), 967 per old-target cell; complete rankings for every pool. Recall@1 as random / hard negatives, chance 0.067. Trap@1 chance: A1 0.20; A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. \* = paired McNemar p < 0.05. "s43 − s42" compares the two Li-style seeds pool by pool.

| Ablation | Recall@1, Li-style s42 | Recall@1, Li-style s43 | s43 − s42 | Change from Baseline (s43) | s43 vs date-trained (s42) |
|---|---|---|---|---|---|
| Baseline | 0.152 / 0.297 | 0.170 / 0.296 | +0.018 / −0.001 | — | −0.275\* / −0.250\* |
| A2 Swap | 0.006 / 0.015 | 0.005 / 0.015 | −0.001 / +0.000 | −0.165\* / −0.281\* | +0.000 / +0.009 |
| A3 Hide | 0.125 / 0.250 | 0.117 / 0.254 | −0.007 / +0.004 | −0.052\* / −0.043\* | +0.016 / +0.146\* |
| A4 Shuffle | 0.110 / 0.115 | 0.113 / 0.113 | +0.003 / −0.003 | −0.058\* / −0.184\* | +0.035\* / +0.035\* |
| A6 All-new | 0.024 / 0.094 | 0.018 / 0.088 | −0.006 / −0.006 | −0.153\* / −0.209\* | −0.235\* / −0.204\* |

| Ablation | Recall@1, s42 → s43 | Trap@1, s42 → s43 | Trap@1 s43 − s42 | Trap@1 s43 vs date-trained (s42) |
|---|---|---|---|---|
| A1-ctrl | 0.572 → 0.579 | 0.100 → 0.109 | +0.008 | −0.019 |
| A1 Traps | 0.039 → 0.047 | 0.945 → **0.938** | −0.007 | +0.376\* |
| A5 n = 1 | 0.040 → 0.047 | 0.950 → 0.944 | −0.006 | +0.269\* |
| A5 n = 2 | 0.039 → 0.046 | 0.953 → 0.949 | −0.004 | +0.212\* |
| A5 n = 5 | 0.038 → 0.041 | 0.960 → 0.957 | −0.003 | +0.112\* |
| A5 n = 10 | 0.037 → 0.038 | 0.963 → 0.962 | −0.001 | +0.032\* |

Newest-first: 0.346 / 0.323 (Baseline), 0.523 (A1-ctrl). s43 minus newest-first on Baseline: −0.176\* / −0.026 (n.s.).

**Summary.** The second seed reproduces the seed-42 Li-style result. Every s43 − s42 difference is 0.018 or smaller and none is significant. Traps take 93.8% of top spots on A1 (seed 42: 94.5%). Swapping the correct item's date takes Recall@1 to about 0. The Baseline score stays below newest-first on random negatives. The Li-style finding is not a single-seed accident on Amazon.

## Llama-2-7B Li-style control (old correct items only, no new items in training), EB-NeRD — dev set (2026-11-03)

**Model.** Li-style Llama-2-7B-Chat on EB-NeRD, seed 42: training pools from the top-up collect (5,246 pools, see design.md), training job 54988 (5 h 13 min, no non-finite loss, final validation loss 0.139), dev evaluation job 54989. **Date-trained** = `llama2_7b_ft_ebnerd` (seed 42); **impression** = `llama2_7b_ftimp_ebnerd` (lr 1e-4 impression fine-tune).

**Evaluation.** EB-NeRD dev: 150 users per cell (A6 131). Recall@1 as random / hard negatives (chance 0.067); Trap@1 chance A1 0.20, A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. \* = paired McNemar p < 0.05. Small dev set; the test run (54990) is the main result.

| Ablation | Recall@1, date-trained | Recall@1, impression | Recall@1, Li-style | Change (Li-style) | Li-style vs date-trained |
|---|---|---|---|---|---|
| Baseline | 0.433 / 0.413 | 0.540 / 0.233 | 0.367 / 0.313 | — | −0.067 / −0.100 |
| A2 Swap | 0.000 / 0.000 | 0.000 / 0.000 | 0.040 / 0.020 | −0.327\* / −0.293\* | +0.040\* / +0.020 |
| A3 Hide | 0.007 / 0.007 | 0.107 / 0.093 | 0.053 / 0.060 | −0.313\* / −0.253\* | +0.047\* / +0.053\* |
| A4 Shuffle | 0.053 / 0.067 | 0.053 / 0.060 | 0.087 / 0.060 | −0.280\* / −0.253\* | +0.033 / −0.007 |
| A6 All-new | 0.115 / 0.084 | 0.137 / 0.084 | 0.198 / 0.183 | −0.145\* / −0.107\* | +0.084 / +0.099\* |

| Ablation | Recall@1: date-trained / impression / Li-style | Trap@1: date-trained / impression / Li-style | Li-style Trap@1 vs date-trained |
|---|---|---|---|
| A1-ctrl | 0.247 / 0.147 / **0.693** | 0.120 / 0.187 / 0.127 | +0.007 |
| A1 Traps | 0.127 / 0.047 / **0.653** | 0.767 / 0.900 / **0.213** | −0.553\* |
| A5 n = 1 | 0.107 / 0.047 / 0.647 | 0.827 / 0.900 / 0.233 | −0.593\* |
| A5 n = 2 | 0.107 / 0.047 / 0.633 | 0.840 / 0.907 / 0.260 | −0.580\* |
| A5 n = 5 | 0.107 / 0.033 / 0.647 | 0.860 / 0.940 / 0.300 | −0.560\* |
| A5 n = 10 | 0.100 / 0.007 / 0.640 | 0.887 / 0.993 / 0.353 | −0.533\* |

Newest-first: 0.567 / 0.540 (Baseline), 0.620 (A1-ctrl). Li-style minus newest-first on Baseline: −0.200\* / −0.227\*.

**Top pick is the newest candidate (dev):** Baseline random / hard: date-trained 0.447 / 0.433, impression 0.347 / 0.233, Li-style 0.307 / 0.300 (correct item newest 0.560 / 0.540). A1-ctrl: date-trained 0.207, impression 0.193, Li-style **0.660** (correct item newest 0.620). A1 Traps: date-trained 0.333, impression 0.253, Li-style 0.120.

**Summary (dev, n = 150; to be checked on test).** Unlike Amazon, the EB-NeRD Li-style model is **not** taken by the traps: Trap@1 0.213 on A1 (chance 0.20; date-trained 0.767, impression 0.900), and Recall@1 under traps (0.653) stays close to A1-ctrl (0.693). Per trap, it picks one at about the 1-in-15 base rate whether the trap is dated < 12 h, 12–24 h or > 24 h after the user's last history item (0.063, 0.059, 0.106), so the trap dates do not attract it. It still depends on dates: swapping the correct item's date takes it to 0.040 / 0.020 and hiding dates to 0.053 / 0.060. Among all-old candidates it picks the newest one 66% of the time (A1-ctrl). On the cold benchmark it scores below newest-first (−0.200\* / −0.227\*).

## Llama-2-7B Li-style control (old correct items only, no new items in training), EB-NeRD — test set (2026-11-04)

**Model.** Same seed-42 Li-style adapter as the dev table above; test evaluation job 56416 (resubmitted on gnode065 after the gnode073 stop). **Date-trained** = `llama2_7b_ft_ebnerd` (seed 42); **impression** = `llama2_7b_ftimp_ebnerd`.

**Evaluation.** EB-NeRD test: 800 users per cell for Baseline/A2–A4 (A6 650), 1,000 per trap cell. Recall@1 as random / hard negatives (chance 0.067); Trap@1 chance A1 0.20, A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. \* = paired McNemar p < 0.05. No missing predictions.

| Ablation | Recall@1, date-trained | Recall@1, impression | Recall@1, Li-style | Change (Li-style) | Li-style vs date-trained |
|---|---|---|---|---|---|
| Baseline | 0.394 / 0.487 | 0.486 / 0.269 | 0.324 / 0.282 | — | −0.070\* / −0.205\* |
| A2 Swap | 0.001 / 0.000 | 0.001 / 0.000 | 0.020 / 0.021 | −0.304\* / −0.261\* | +0.019\* / +0.021\* |
| A3 Hide | 0.014 / 0.013 | 0.096 / 0.084 | 0.059 / 0.062 | −0.265\* / −0.220\* | +0.045\* / +0.050\* |
| A4 Shuffle | 0.035 / 0.031 | 0.069 / 0.070 | 0.066 / 0.056 | −0.258\* / −0.226\* | +0.031\* / +0.025\* |
| A6 All-new | 0.097 / 0.117 | 0.157 / 0.123 | 0.140 / 0.165 | −0.203\* / −0.131\* | +0.043\* / +0.048\* |

| Ablation | Recall@1: date-trained / impression / Li-style | Trap@1: date-trained / impression / Li-style | Li-style Trap@1 vs date-trained |
|---|---|---|---|
| A1-ctrl | 0.241 / 0.132 / **0.670** | 0.202 / 0.162 / 0.144 | −0.058\* |
| A1 Traps | 0.106 / 0.051 / **0.654** | 0.828 / 0.898 / **0.194** | −0.634\* |
| A5 n = 1 | 0.104 / 0.045 / 0.645 | 0.846 / 0.911 / 0.224 | −0.622\* |
| A5 n = 2 | 0.107 / 0.042 / 0.638 | 0.850 / 0.922 / 0.249 | −0.601\* |
| A5 n = 5 | 0.113 / 0.030 / 0.625 | 0.858 / 0.944 / 0.314 | −0.544\* |
| A5 n = 10 | 0.110 / 0.028 / 0.630 | 0.888 / 0.971 / 0.367 | −0.521\* |

Li-style Recall@1 change from A1-ctrl: A1 −0.016 (n.s.), A5 −0.025\* to −0.045\*. Newest-first: 0.560 / 0.515 (Baseline), 0.603 (A1-ctrl). Li-style minus newest-first on Baseline: −0.236\* / −0.233\*.

**Top pick is the newest candidate (test):** Baseline random / hard: date-trained 0.384 / 0.469, impression 0.390 / 0.320, Li-style 0.291 / 0.237 (correct item newest 0.556 / 0.511). A1-ctrl (all candidates old): date-trained 0.233, impression 0.181, Li-style **0.639** (correct item newest 0.602). A1 Traps (a trap is always newest): date-trained 0.334, impression 0.286, Li-style **0.070**.

**Summary (test confirms dev).** On EB-NeRD the Li-style model is **not** taken by the traps, unlike on Amazon (0.945). A1 Trap@1 is 0.194, at chance (0.20), against 0.828 for date-trained and 0.898 for impression. With more traps it stays **below** chance (A5 n = 10: 0.367 vs 0.867), so it ranks items dated after the user's history *lower* than average. Recall@1 under traps (0.654) is almost unchanged from A1-ctrl (0.670). It still leans on dates *within the old range*: among all-old candidates its top pick is the newest one 64% of the time, and swapping or hiding dates collapses it (A2 0.020 / 0.021, A3 0.059 / 0.062). It learned "newest among old items", not "newest overall". On the cold benchmark it is below both date-trained and newest-first (−0.236\* / −0.233\* vs newest-first).

## Llama-2-7B Li-style control, third training seed (44), Amazon Video Games — test set, and three-seed mean ± SD (2026-11-04)

**Model.** Li-style Llama-2-7B-Chat, seed 44: training job 56417 (1 h 40 min, gnode065, final validation loss 0.274; seeds 42 / 43: 0.266 / 0.271), evaluation job 56418. Same SFT data and settings as seeds 42 and 43.

**Evaluation.** The same Amazon VG test pools as the seed-42 and seed-43 sections (800 per cold-target cell, A6 799, 967 per old-target cell); complete rankings for every pool. Recall@1 as random / hard negatives (chance 0.067); Trap@1 chance A1 0.20, A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. \* = paired McNemar p < 0.05. Mean ± sample SD over seeds 42, 43, 44.

| Ablation | Recall@1, s44 | s44 − s42 | Change from Baseline (s44) | Recall@1, mean ± SD (3 seeds) |
|---|---|---|---|---|
| Baseline | 0.158 / 0.302 | +0.005 / +0.005 | — | 0.160 ± 0.009 / 0.298 ± 0.003 |
| A2 Swap | 0.001 / 0.005 | −0.005 / −0.010\* | −0.156\* / −0.297\* | 0.004 ± 0.003 / 0.012 ± 0.006 |
| A3 Hide | 0.116 / 0.271 | −0.009 / +0.021 | −0.041\* / −0.031 | 0.119 ± 0.005 / 0.258 ± 0.011 |
| A4 Shuffle | 0.100 / 0.107 | −0.010 / −0.007 | −0.058\* / −0.195\* | 0.108 ± 0.007 / 0.112 ± 0.004 |
| A6 All-new | 0.015 / 0.088 | −0.009 / −0.006 | −0.141\* / −0.215\* | 0.019 ± 0.005 / 0.090 ± 0.003 |

| Ablation | Recall@1, s44 | Trap@1, s44 | Trap@1 s44 vs date-trained | Recall@1 mean ± SD | Trap@1 mean ± SD |
|---|---|---|---|---|---|
| A1-ctrl | 0.564 | 0.104 | −0.023 | 0.572 ± 0.008 | — |
| A1 Traps | 0.036 | **0.957** | +0.395\* | 0.041 ± 0.006 | **0.947 ± 0.010** |
| A5 n = 1 | 0.034 | 0.961 | +0.285\* | — | 0.952 ± 0.009 |
| A5 n = 2 | 0.033 | 0.964 | +0.226\* | — | 0.955 ± 0.008 |
| A5 n = 5 | 0.029 | 0.970 | +0.125\* | — | 0.962 ± 0.007 |
| A5 n = 10 | 0.029 | 0.971 | +0.041\* | — | 0.965 ± 0.005 |

Newest-first: 0.346 / 0.323 (Baseline), 0.523 (A1-ctrl). s44 minus newest-first on Baseline: −0.189\* / −0.020 (n.s.).

**Summary.** Seed 44 reproduces seeds 42 and 43. The only significant seed difference is A2 hard (−0.010\*, both near zero). Over three seeds, traps take **94.7% ± 1.0%** of top spots on A1 (chance 20%), swapping the correct item's date leaves 0.004 / 0.012 Recall@1, and the Baseline score (0.160 / 0.298) stays below newest-first on random negatives. The Amazon Li-style finding is stable across training seeds.

## Llama-2-7B Li-style control, second training seed (43), EB-NeRD — test set (2026-11-04)

**Model.** Li-style Llama-2-7B-Chat on EB-NeRD, seed 43: training job 56419 (5 h 03 min, gnode065, final validation loss 0.142; seed 42: 0.139), evaluation job 56420. Same SFT data (5,246 pools from the top-up collect) and settings as seed 42.

**Evaluation.** The same EB-NeRD test pools as the seed-42 test section (800 per cold-target cell, A6 650, 1,000 per trap cell); complete rankings for every pool. Recall@1 as random / hard negatives (chance 0.067); Trap@1 chance A1 0.20, A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. \* = paired McNemar p < 0.05. Seed 44 is training (56421/56422); mean ± SD follows it.

| Ablation | Recall@1, Li-style s42 | Recall@1, Li-style s43 | s43 − s42 | Change from Baseline (s43) | s43 vs date-trained |
|---|---|---|---|---|---|
| Baseline | 0.324 / 0.282 | 0.234 / 0.242 | −0.090\* / −0.040\* | — | −0.160\* / −0.245\* |
| A2 Swap | 0.020 / 0.021 | 0.025 / 0.028 | +0.005 / +0.006 | −0.209\* / −0.215\* | +0.024\* / +0.028\* |
| A3 Hide | 0.059 / 0.062 | 0.072 / 0.076 | +0.014 / +0.014 | −0.161\* / −0.166\* | +0.059\* / +0.064\* |
| A4 Shuffle | 0.066 / 0.056 | 0.061 / 0.062 | −0.005 / +0.006 | −0.172\* / −0.180\* | +0.026\* / +0.031\* |
| A6 All-new | 0.140 / 0.165 | 0.149 / 0.140 | +0.009 / −0.025 | −0.100\* / −0.122\* | +0.052\* / +0.023 |

| Ablation | Recall@1, s42 → s43 | Trap@1, s42 → s43 | Trap@1 s43 − s42 | Trap@1 s43 vs date-trained |
|---|---|---|---|---|
| A1-ctrl | 0.670 → 0.665 | 0.144 → 0.144 | +0.000 | −0.058\* |
| A1 Traps | 0.654 → 0.676 | 0.194 → **0.133** | −0.061\* | −0.695\* |
| A5 n = 1 | 0.645 → 0.674 | 0.224 → 0.159 | −0.065\* | −0.687\* |
| A5 n = 2 | 0.638 → 0.677 | 0.249 → 0.174 | −0.075\* | −0.676\* |
| A5 n = 5 | 0.625 → 0.692 | 0.314 → 0.220 | −0.094\* | −0.638\* |
| A5 n = 10 | 0.630 → 0.702 | 0.367 → 0.288 | −0.079\* | −0.600\* |

Newest-first: 0.560 / 0.515 (Baseline), 0.603 (A1-ctrl). s43 minus newest-first on Baseline: −0.326\* / −0.273\*.

**Summary.** Seed 43 reproduces the qualitative EB-NeRD Li-style result and makes it stronger: traps take only 13.3% of top spots on A1 (chance 20%) and stay well below chance with more traps (A5 n = 10: 0.288 vs 0.867), so new-dated items are pushed down. Swapping or hiding dates still collapses it (0.025 / 0.028, 0.072 / 0.076). Unlike Amazon, the seeds differ in size on EB-NeRD: Baseline is lower for s43 (−0.090\* / −0.040\*) and trap avoidance is stronger (−0.061\* on A1), so the EB-NeRD Li-style numbers vary more across seeds than the Amazon ones. Both seeds sit well below newest-first on the cold benchmark.

## Llama-2-7B Li-style control, third training seed (44), EB-NeRD — test set, and three-seed mean ± SD (2026-11-05)

**Model.** Li-style Llama-2-7B-Chat on EB-NeRD, seed 44: training job 56421 (5 h 03 min, gnode065, no non-finite loss, final validation loss 0.136; seeds 42 / 43: 0.139 / 0.142), evaluation job 56422. Same SFT data and settings as seeds 42 and 43.

**Evaluation.** The same EB-NeRD test pools as the seed-42 and seed-43 sections (800 per cold-target cell, A6 650, 1,000 per trap cell); complete rankings for every pool. Recall@1 as random / hard negatives (chance 0.067); Trap@1 chance A1 0.20, A5 n = 1, 2, 5, 10: 0.267, 0.333, 0.533, 0.867. \* = paired McNemar p < 0.05. Mean ± sample SD over seeds 42, 43, 44.

| Ablation | Recall@1, s44 | s44 − s42 | Change from Baseline (s44) | Recall@1, mean ± SD (3 seeds) |
|---|---|---|---|---|
| Baseline | 0.285 / 0.273 | −0.039\* / −0.010 | — | 0.281 ± 0.045 / 0.266 ± 0.021 |
| A2 Swap | 0.022 / 0.031 | +0.003 / +0.010 | −0.263\* / −0.241\* | 0.022 ± 0.003 / 0.027 ± 0.005 |
| A3 Hide | 0.046 / 0.048 | −0.013 / −0.015 | −0.239\* / −0.225\* | 0.059 ± 0.013 / 0.062 ± 0.014 |
| A4 Shuffle | 0.058 / 0.048 | −0.009 / −0.009 | −0.228\* / −0.225\* | 0.062 ± 0.004 / 0.055 ± 0.007 |
| A6 All-new | 0.151 / 0.145 | +0.011 / −0.020 | −0.163\* / −0.157\* | 0.147 ± 0.006 / 0.150 ± 0.013 |

| Ablation | Recall@1, s44 | Trap@1, s44 | Trap@1 s44 − s42 | Recall@1 mean ± SD | Trap@1 mean ± SD (chance) |
|---|---|---|---|---|---|
| A1-ctrl | 0.664 | 0.141 | −0.003 | 0.666 ± 0.003 | — |
| A1 Traps | 0.671 | **0.142** | −0.052\* | 0.667 ± 0.012 | **0.156 ± 0.033** (0.20) |
| A5 n = 1 | 0.672 | 0.170 | −0.054\* | — | 0.184 ± 0.035 (0.267) |
| A5 n = 2 | 0.669 | 0.193 | −0.056\* | — | 0.205 ± 0.039 (0.333) |
| A5 n = 5 | 0.680 | 0.248 | −0.066\* | — | 0.261 ± 0.048 (0.533) |
| A5 n = 10 | 0.672 | 0.320 | −0.047\* | — | 0.325 ± 0.040 (0.867) |

Newest-first: 0.560 / 0.515 (Baseline), 0.603 (A1-ctrl). s44 minus newest-first on Baseline: −0.275\* / −0.242\*.

**Summary.** Seed 44 lands between seeds 42 and 43, so the three-seed EB-NeRD Li-style picture is settled:
- **Not drawn to traps:** Trap@1 0.156 ± 0.033 on A1, below chance (0.20) on every seed, and far below chance with more traps (A5 n = 10: 0.325 vs 0.867). Recall@1 with traps (0.667) equals A1-ctrl (0.666): traps cost it nothing.
- **Still date-driven among old items:** swapping the correct item's date leaves 0.022 / 0.027, hiding dates 0.059 / 0.062, shuffling 0.062 / 0.055 (all about chance), on every seed.
- **Below newest-first** on the cold benchmark on every seed (mean 0.281 / 0.266 vs 0.560 / 0.515).
- **Seed spread:** Baseline varies more than on Amazon (SD 0.045 / 0.021 vs 0.009 / 0.003); every diagnostic conclusion holds on each seed.

## Qwen2.5-7B Li-style control (old correct items only, no new items in training), Amazon Video Games — dev and test (2026-11-05)

**Model.** Qwen2.5-7B-Instruct (`a09a35458c`), QLoRA with `finetune_qwen25_ada_oldonly.yaml`: the uniform Qwen recipe (4-bit NF4, fp16 compute, fp16 embeddings, LoRA r16 / α32, lr 2e-4, 1 epoch, effective batch 32 as 4 GPUs × 4 × 2), seed 42, one seed (user decision). Same Li-style SFT pools as the Llama-2 Li-style models (23,820 train / 1,278 validation examples). Training job 57630 (1 h 29 min, gnode065, final validation loss 0.276; Llama-2 Li-style 0.266), dev job 57631, test job 57632 (6 h 11 min). Evaluated 4-bit (vLLM bitsandbytes), like every Qwen run.

**Comparators.** The paired columns use our Llama-2 runs on the same pools: **Llama date-trained** (`llama2_7b_ft_amazon_vg`, seed 42) and **Llama Li-style** (seed 42). The Qwen date-trained Amazon model (teammate's, jobs 40320/40321) was scored on separately built pools (962 old-target users instead of 967), so it is compared by value only, not paired. Recall@1 as random / hard negatives (chance 0.067); Trap@1 chance A1 0.20, A5 0.267 / 0.333 / 0.533 / 0.867. \* = paired McNemar p < 0.05.

**Test** (800 per cold-target cell, A6 799, 967 per old-target cell; complete rankings):

| Ablation | Recall@1, Qwen date-trained (unpaired) | Recall@1, Qwen Li-style | Change (Qwen Li-style) | vs Llama Li-style | vs Llama date-trained |
|---|---|---|---|---|---|
| Baseline | 0.432 / 0.532 | 0.149 / 0.383 | — | −0.004 / +0.085\* | −0.296\* / −0.164\* |
| A2 Swap | 0.015 / 0.014 | 0.000 / 0.006 | −0.149\* / −0.376\* | −0.006 / −0.009\* | −0.005 / +0.000 |
| A3 Hide | 0.138 / 0.146 | 0.096 / 0.282 | −0.052\* / −0.100\* | −0.029\* / +0.033\* | −0.005 / +0.175\* |
| A4 Shuffle | 0.079 / 0.081 | 0.102 / 0.106 | −0.046\* / −0.276\* | −0.007 / −0.009 | +0.025\* / +0.029\* |
| A6 All-new | 0.152 / 0.297 | 0.019 / 0.124 | −0.130\* / −0.259\* | −0.005 / +0.030\* | −0.234\* / −0.168\* |

| Ablation | Recall@1, Qwen Li-style | Trap@1: Qwen date-trained (unpaired) / Llama Li-style / **Qwen Li-style** | Qwen Li-style Trap@1 vs Llama Li-style |
|---|---|---|---|
| A1-ctrl | 0.558 | — / 0.100 / 0.110 | +0.009 |
| A1 Traps | 0.021 | 0.600 / 0.945 / **0.969** | +0.024\* |
| A5 n = 1 | 0.021 | 0.672 / 0.950 / 0.971 | +0.021\* |
| A5 n = 2 | 0.021 | 0.738 / 0.953 / 0.975 | +0.022\* |
| A5 n = 5 | 0.019 | 0.857 / 0.960 / 0.980 | +0.021\* |
| A5 n = 10 | 0.019 | 0.940 / 0.963 / 0.980 | +0.018\* |

Newest-first: 0.346 / 0.323 (Baseline), 0.523 (A1-ctrl). Qwen Li-style minus newest-first on Baseline: −0.198\* / +0.060\*.

**Dev** (150 per cold-target cell, 144 per old-target cell): Baseline 0.220 / 0.453; A2 0.000 / 0.013; A3 0.160 / 0.313; A4 0.147 / 0.120; A6 0.020 / 0.153; A1-ctrl 0.549; A1 Recall@1 0.014, Trap@1 **0.972**; A5 Trap@1 0.986–0.993. Same direction as test in every row.

**Summary.** The Amazon Li-style result is not specific to Llama-2: Qwen2.5-7B trained the Li et al. way shows the same, slightly stronger, recency shortcut.
- **Traps take 96.9% of top spots** on A1 (Llama Li-style 94.5%, Qwen date-trained 60%, chance 20%), and 97–98% on every A5 row; Recall@1 under traps falls from 0.558 to 0.021.
- **Dates decide:** swapping the correct item's date gives 0.000 / 0.006, shuffling −0.046\* / −0.276\*; with only new candidates (A6) it is at or below chance on random negatives (0.019).
- **Benchmark score:** 0.149 / 0.383, below newest-first on random negatives (−0.198\*) and above it on hard negatives (+0.060\*); hard negatives are old by construction, so a recency rule separates them easily.
- Caveats: one seed; 4-bit evaluation; the Qwen date-trained comparison is unpaired (different pools).
