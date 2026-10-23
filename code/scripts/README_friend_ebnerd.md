# Running the EB-NeRD pilot on one GPU (Qwen2.5-32B)

You are running **30 inference tasks** of an IRE course project on **one free 32 GB GPU**: Qwen2.5-32B-Instruct (4-bit AWQ) reranks candidate news articles for 150 users, under 4 conditions. Everything is pinned (model revision, package versions, prompts), so the results line up with the runs on the IIIT ADA cluster.

**Needs:** Linux, an NVIDIA GPU with ≥ 32 GB that nobody else is using, CUDA driver ≥ 12.4, ~30 GB of free disk (model 19.3 GB, environment ~8 GB, data ~0.5 GB), internet for the setup. **Time:** ~15 min setup + ~20–40 min model download + **~10 h** of GPU time (probably less on an RTX 5000 Ada).

The scripts only use the GPU you name (`GPU=0`) and refuse to start if it already has memory in use. They do not touch any existing conda / Python environment.

## 1. Unpack and set up (once)

```bash
unzip ebnerd_qwen_bundle.zip            # creates IRE_Project/
cd IRE_Project/code
bash scripts/setup_local_env.sh         # new Python 3.12 env in IRE_Project/.venv (~10 min)
source ../.venv/bin/activate
export HF_HOME=$PWD/../hf_cache         # where the model goes (or any folder with 20 GB free)
python scripts/download_models.py qwen25_32b    # 19.3 GB, pinned revision
```

## 2. Smoke test (~15 min): does it run on this GPU?

```bash
cd IRE_Project/code
nvidia-smi                              # check which GPU is free; use its number below
CONFIG=ebnerd MODEL=qwen25_32b STAGE=smoke-k15 GPU=0 TP=1 bash scripts/run_local.sh
```

It should end with `stage smoke-k15 finished with status 0` and print 7 lines starting `{"task": ...}`. **Send those 7 lines** (they hold the timings) before starting step 3, so the full run can be checked first.

If it fails, send the last ~40 lines of the log it names (`../outputs/logs/ebnerd_qwen25_32b_smoke-k15_local-.../gpu0.log`).

## 3. The pilot (~10 h): 30 tasks

Run it in the background so it survives closing the terminal:

```bash
cd IRE_Project/code
CONFIG=ebnerd MODEL=qwen25_32b STAGE=pilot-dev GPU=0 TP=1 nohup bash scripts/run_local.sh > pilot.log 2>&1 &
```

Check progress any time:

```bash
tail -3 pilot.log                                                    # overall
grep -c '"task"' ../outputs/logs/ebnerd_qwen25_32b_pilot-dev_local-*/gpu0.log   # tasks done (of 30)
tail -c 300 ../outputs/logs/ebnerd_qwen25_32b_pilot-dev_local-*/gpu0.log       # current task's progress bar
nvidia-smi                                                           # GPU busy?
```

If it stops (reboot, crash, someone needs the GPU), just run the same command again: finished tasks are skipped and finished model outputs are reused from the cache.

To stop it: `pkill -f "coldrank.llm.run --config ebnerd"`.

## 4. Send the results back

When `pilot.log` ends with `stage pilot-dev finished with status 0`:

```bash
cd IRE_Project
zip -r ebnerd_qwen_results.zip outputs/ebnerd outputs/logs outputs/tasks outputs/cache code/pilot.log
```

Send `ebnerd_qwen_results.zip` (a few hundred MB at most).

## What is being run (for reference)

| Condition (EB-NeRD dev) | Methods | Tasks |
|---|---|---|
| Baseline, random and hard negatives | Direct-Rec, pointwise, Base, Fast, Structural, Structural without recency, SSC | 14 |
| A3: dates hidden, random and hard | Direct-Rec, pointwise, SSC, Structural | 8 |
| A1: old correct article + 3 new skipped articles (traps) | Direct-Rec, pointwise, SSC, Structural | 4 |
| A1-ctrl: A1 without the traps | Direct-Rec, pointwise, SSC, Structural | 4 |
