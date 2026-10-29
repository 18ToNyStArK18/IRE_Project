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
