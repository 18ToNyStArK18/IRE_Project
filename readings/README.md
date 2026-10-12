# Readings

| File | Paper | Role in our project |
|---|---|---|
| `li2026_llm_reasoning_cold_start_netflix.pdf` | Li, Wang, Wang, Li, Ghosh, Cocos. *LLM Reasoning for Cold-Start Item Recommendation.* WWW '26. arXiv:2511.18261 | **Baseline protocol we reproduce** |
| `fang2025_recency_bias_llm_reranking.pdf` | Fang, Tao, Chen, Chang, Sakai. *Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking.* SIGIR-AP '25. arXiv:2509.11353 | **Closest prior work** |
| `dong2026_diagnosing_cold_start_retrieval.pdf` | Dong, Qin, Shah, Wang. *Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation.* arXiv:2606.29947 | Evidence / tail-prior decomposition we extend; Amazon preprocessing code |

## Li et al. (2026): the Netflix baseline

- **Cold-start definition (§2.1):** an item launched after the latest interaction in *all* user histories, and after the base LLM's training cutoff. Cold items appear only as ground-truth targets (§3.1.1).
- **Candidate pool (§2.1):** the top 40 titles from Netflix's production ranker, plus 10 sampled cold-start titles that include the target, for 50 candidates. The paper does not say how the other 9 cold titles are sampled.
- **Item attributes in the prompt:** title, launch date, genre (§2.2.1).
- **Test set (§3.1.2):** thousands of interactions from **a single day**, chosen so that the target is a cold-start item.
- **Metrics:** Recall@1 on *Discovery* (primary; plays of content the user has not seen) and on *AnyPlay* (secondary).
- **Model:** Qwen-2.5-32B-Instruct.
- **Methods (§3.2.1, §2.2):**
  - *Fast-Reason*: Base-Reason with twice the history length, shorter example patterns and one-shot prompting. It is the 0% reference row in Table 1.
  - *Direct-Rec*: rank directly, with no reasoning.
  - *Base-Reason*: summarize the user's interests, then reason about the candidates. Uses two few-shot examples.
  - *Structural Reasoning*: build preference-factor paths (actor / genre / director), score each candidate against each path, then take a weighted sum. **The prompt explicitly asks for weights based on "factor prominence, action type, and recency".**
  - *Soft Self-Consistency*: the LLM builds several reasoning paths and summarizes them softly instead of taking a majority vote.
- **Fine-tuning:**
  - SFT: QLoRA 4-bit, 1 epoch, on 7,252 successful reasoning paths.
  - GRPO: reward +1 for a correct prediction, −0.1 for a wrong one, −1 for a parse failure.
  - SFT+GRPO.
- **Results:**
  - **Only relative (%) numbers are reported, never absolute Recall@1.** So their gains cannot be checked against the date-only reference of 0.10.
  - Cold-start, no fine-tuning: Soft Self-Consistency is best on Discovery (+6.4% over Fast-Reason), and Base-Reason is best on AnyPlay.
  - Cold-start, fine-tuning: GRPO helps Discovery most (+18.4% with Base-Reason) and SFT helps AnyPlay most. **SFT+GRPO is *not* best in cold-start.**
  - The "8% over Netflix production" claim is **warm-start**, Discovery (Table 3).

## Fang et al. (2025): closest prior work

- **Task:** passage reranking on TREC DL21/DL22 (53 + 76 queries), starting from a BM25 top-100. It is not recommendation and not cold-start.
- **Models:** GPT-3.5-turbo, GPT-4, GPT-4o, Llama-3 8B/70B, Qwen-2.5 7B/72B. Prompt: RankZephyr. Decoding: temperature 0.
- **Listwise test:**
  - Rerank once, then prefix every passage with "Published on: YYYY/MM/DD."
  - Dates **run against the first ranking**: rank 1 gets 1926/01/01 and rank 100 gets 2025/01/01.
  - Rerank again with a sliding window of size 10, and measure how far the ranking moves.
- **Pairwise test:**
  - Uses pairs with the *same* human relevance label.
  - The passage the model preferred gets 1980/01/01 and the other gets 2025/01/01. Preferences reverse up to 25% of the time on average.
- **Metrics:** rank-shift measures (mean absolute rank shift, largest rank shift) and the change in the Top-10's mean publication year (up to +4.78 years).
- **Finding:** every model promotes passages that look new. Larger models reduce the effect but do not remove it.

**How our project differs:**
1. **Domain:** we study recommendation, not query–passage relevance.
2. **Benchmark confound:** we ask whether temporal-split cold-start benchmarks *give credit for* this bias. Fang et al. measure the bias, not the evaluation artifact.
3. **Realistic dates:** our dates are real launch times or plausible swaps of them, not a synthetic 1926–2025 ramp.
4. **Accuracy against ground truth:** we score against the item the user actually chose next (Recall@1, Trap@1) and decompose it into evidence and recency. Fang et al. measure rank shifts.
5. **Domain contrast:** news, where recency is a valid signal, against video games, where it is not.

Fang et al.'s pairwise design (equal relevance, preferred item made old) is the closest analogue to our A2 date swap.
