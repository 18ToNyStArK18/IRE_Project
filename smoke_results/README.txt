Smoke test + Direct-Rec prompt experiment (Amazon VG, dev, random negatives)

Smoke test (2 pools per model): logs/cr-smoke-*.out (job logs) and logs/llm-<jobid>/ (per-task logs).
  19485: failed in 8 s on node gnode077 (GPU not visible) -> node excluded afterwards
  19488: llama2_7b, qwen_7b ok; llama2_13b failed (setuptools missing), mistral failed
         (sliding window vs prefix caching), phi2 failed (OOM)
  19492: retry -> mistral ok; llama2_13b does not fit 4096 context on one GPU; phi2 assertion
  19506: phi2 ok after the yes/no rewrite (amazon_vg/.../phi2/pointwise/dev, 2 pools)
  ada-shrish 19927/20396: llama2_13b on 2 GPUs (tensor parallel) ok
  checks/v.log, h.log: vLLM vs HF yes/no scores on 6 prompts (same ranking; diff <= 0.03)
NOTE: the smoke outputs for direct (2 pools) were overwritten in the same directories by the
20-pool Direct-Rec prompt experiment below; their per-task logs are kept.

Direct-Rec prompt experiment (20 pools x 3 shuffles), amazon_vg/baseline/random/<model>/<variant>/dev:
  direct            = original prompt then (numbered IDs + "FINAL: [" prefill); 90-100% positional copying
  direct_free       = numbered IDs, no prefill
  direct_codes      = two-letter codes + prefill
  direct_codes_free = two-letter codes, no prefill   <- frozen as the study's Direct-Rec prompt
raw.jsonl = every model output with its parse; preds.jsonl = Borda-combined ranking per pool.
