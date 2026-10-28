"""LLM engines: vLLM (cluster) and HF transformers (fallback / local CPU tests).

Both expose
    generate(prompts, max_new_tokens)  -> list[str]          (greedy decoding)
    yes_no(prompts)                    -> list[float]        log p(Yes) - log p(No)
    count_tokens(prompt)               -> int
where `prompts` are fully formatted strings (coldrank.llm.chat). Pointwise scores
are exact in both engines: the full next-token distribution, not a top-k list.
"Yes"/"No" may each map to several single-token spellings ("Yes", " Yes"); their
probabilities are summed (logsumexp).

CachedEngine wraps either engine with an on-disk JSONL cache keyed by
(model, revision, quantization, call, prompt, params), so jobs are resumable and
shared prompts are computed once.
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path

from coldrank.common import stable_hash

YES_SPELLINGS = ("Yes", " Yes")
EXACT_TOPK = 500  # depth of the full-vocabulary logprob list used to verify fast pointwise scores
NO_SPELLINGS = ("No", " No")


def answer_token_ids(tokenizer, spellings) -> list[int]:
    """Token ids of the spellings that encode to exactly one token."""
    ids = set()
    for s in spellings:
        t = tokenizer.encode(s, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    if not ids:
        raise RuntimeError(f"no single-token spelling among {spellings} for this tokenizer")
    return sorted(ids)


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class HFEngine:
    def __init__(self, mcfg: dict, device: str | None = None, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = mcfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = dict(revision=mcfg["revision"], trust_remote_code=mcfg.get("trust_remote_code", False))
        self.tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], **kw)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dtype = getattr(torch, mcfg.get("dtype", "float16")) if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], torch_dtype=dtype, **kw).to(self.device)
        self.model.eval()
        self.bs = batch_size
        self.yes_ids = answer_token_ids(self.tok, YES_SPELLINGS)
        self.no_ids = answer_token_ids(self.tok, NO_SPELLINGS)

    def count_tokens(self, prompt: str) -> int:
        return len(self.tok(prompt)["input_ids"])

    def generate(self, prompts: list[str], max_new_tokens: list[int]) -> list[str]:
        out = []
        for s in range(0, len(prompts), self.bs):
            chunk, mx = prompts[s : s + self.bs], max(max_new_tokens[s : s + self.bs])
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=self.tok.pad_token_id)
            for k, row in enumerate(gen):
                new = row[enc["input_ids"].shape[1]:][: max_new_tokens[s + k]]
                out.append(self.tok.decode(new, skip_special_tokens=True))
        return out

    def yes_no(self, prompts: list[str]) -> list[float]:
        out = []
        for s in range(0, len(prompts), self.bs):
            enc = self.tok(prompts[s : s + self.bs], return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()  # left padding -> last position is real
            lp = self.torch.log_softmax(logits, -1)
            for row in lp:
                out.append(logsumexp([row[i].item() for i in self.yes_ids]) - logsumexp([row[i].item() for i in self.no_ids]))
        return out


HFEngine.yes_no_exact = HFEngine.yes_no  # the HF path is already exact


class VLLMEngine:
    def __init__(self, mcfg: dict, gpu_memory_utilization: float = 0.90):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.cfg = mcfg
        # max_logprobs: the exact pointwise reference needs deep top-k lists (Llama-2-7B puts
        # "No" far below the top 20; its Yes-No log-odds are about +5)
        kw = {"gpu_memory_utilization": gpu_memory_utilization, "max_logprobs": EXACT_TOPK}
        kw.update(mcfg.get("vllm") or {})  # optional per-model engine options (models.yaml `vllm:`)
        model, revision = mcfg["hf_id"], mcfg["revision"]
        if mcfg.get("quantization") == "bitsandbytes" and not Path(model).is_dir():
            # vLLM's bitsandbytes loader lists the repo's files online even with HF_HUB_OFFLINE=1;
            # give it the local snapshot instead
            from huggingface_hub import snapshot_download

            model, revision = snapshot_download(model, revision=revision, local_files_only=True), None
        self.llm = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            quantization=mcfg.get("quantization"),
            dtype=mcfg.get("dtype", "float16"),  # default: 2080 Ti (sm_75) has no bf16
            max_model_len=mcfg["max_model_len"],
            trust_remote_code=mcfg.get("trust_remote_code", False),
            enable_prefix_caching=True,
            seed=0,
            **kw,
        )
        self.tok = self.llm.get_tokenizer()
        self.yes_ids = answer_token_ids(self.tok, YES_SPELLINGS)
        self.no_ids = answer_token_ids(self.tok, NO_SPELLINGS)

    def count_tokens(self, prompt: str) -> int:
        return len(self.tok.encode(prompt))

    def generate(self, prompts: list[str], max_new_tokens: list[int]) -> list[str]:
        params = [self.SamplingParams(temperature=0.0, max_tokens=int(m)) for m in max_new_tokens]
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        return [o.outputs[0].text for o in outs]

    def yes_no(self, prompts: list[str]) -> list[float]:
        """Fast path: generate ONE token restricted to the Yes/No token ids and read their
        logprobs. Restricting the vocabulary renormalizes over those ids only, which
        leaves log p(Yes) - log p(No) unchanged, and prefix caching reuses the shared
        history across a pool's 50 candidates. Verified against yes_no_exact()
        (run.py --verify-pointwise) before a pointwise run is trusted."""
        ids = self.yes_ids + self.no_ids
        sp = self.SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(ids), allowed_token_ids=ids)
        outs = self.llm.generate(prompts, sp, use_tqdm=True)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]  # {token_id: Logprob} for the single generated position
            missing = [t for t in ids if t not in lp]
            if missing:
                raise RuntimeError(f"vLLM returned no logprob for token ids {missing}; "
                                   "this vLLM version does not apply allowed_token_ids before logprobs")
            res.append(logsumexp([lp[t].logprob for t in self.yes_ids]) - logsumexp([lp[t].logprob for t in self.no_ids]))
        return res

    def yes_no_exact(self, prompts: list[str]) -> list[float]:
        """Reference for yes_no(): one unrestricted greedy step returning the top-EXACT_TOPK
        logprobs over the FULL vocabulary, i.e. the model's exact log p(t | prompt). Returns NaN
        for a prompt where a Yes/No spelling falls outside them. (Prompt logprobs would be the
        other way, but in vLLM 0.8.5 V0 they crash when prefix caching is on.)"""
        sp = self.SamplingParams(max_tokens=1, temperature=0.0, logprobs=EXACT_TOPK)
        outs = self.llm.generate(prompts, sp, use_tqdm=True)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            if any(t not in lp for t in self.yes_ids + self.no_ids):
                res.append(float("nan"))
                continue
            res.append(logsumexp([lp[t].logprob for t in self.yes_ids]) - logsumexp([lp[t].logprob for t in self.no_ids]))
        return res


class CachedEngine:
    """On-disk cache in front of an engine. One JSONL file per model."""

    def __init__(self, engine, cache_dir: Path, model_key: str, cache_tag: str | None = None):
        self.engine = engine
        self.cfg = engine.cfg
        # cache_tag gives each parallel worker its own file: concurrent appends to one
        # file (especially on a network filesystem) can interleave and lose entries.
        self.path = Path(cache_dir) / (f"{model_key}__{cache_tag}.jsonl" if cache_tag else f"{model_key}.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # a merged fine-tuned model is identified by base + adapter hash (ident_hf_id), not its local path
        self.ident = (model_key, engine.cfg.get("ident_hf_id", engine.cfg["hf_id"]), engine.cfg["revision"],
                      engine.cfg.get("quantization"))
        self.mem: dict[str, object] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated last line from a killed job
                    self.mem[r["k"]] = r["v"]

    def _key(self, call: str, prompt: str, params) -> str:
        return stable_hash(self.ident, call, prompt, params)

    def _store(self, items: list[tuple[str, object]]) -> None:
        with self._lock, open(self.path, "a") as f:
            for k, v in items:
                self.mem[k] = v
                f.write(json.dumps({"k": k, "v": v}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def count_tokens(self, prompt: str) -> int:
        return self.engine.count_tokens(prompt)

    def generate(self, prompts: list[str], max_new_tokens: list[int]) -> list[str]:
        keys = [self._key("gen", p, int(m)) for p, m in zip(prompts, max_new_tokens)]
        todo = [i for i, k in enumerate(keys) if k not in self.mem]
        if todo:
            res = self.engine.generate([prompts[i] for i in todo], [max_new_tokens[i] for i in todo])
            self._store([(keys[i], r) for i, r in zip(todo, res)])
        return [self.mem[k] for k in keys]

    def yes_no(self, prompts: list[str]) -> list[float]:
        keys = [self._key("yesno", p, None) for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.mem]
        if todo:
            res = self.engine.yes_no([prompts[i] for i in todo])
            self._store([(keys[i], r) for i, r in zip(todo, res)])
        return [self.mem[k] for k in keys]


def make_engine(mcfg: dict, backend: str, cache_dir: Path, model_key: str, cache_tag: str | None = None,
                **kw) -> CachedEngine:
    if backend == "vllm":
        eng = VLLMEngine(mcfg, **kw)
    elif backend == "hf":
        eng = HFEngine(mcfg, **kw)
    else:
        raise ValueError(backend)
    return CachedEngine(eng, cache_dir, model_key, cache_tag)
