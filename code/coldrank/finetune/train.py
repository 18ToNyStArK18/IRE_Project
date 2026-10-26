"""QLoRA fine-tuning on the pointwise SFT examples (configs/finetune.yaml).

The model sees the inference prompt (with BOS, as vLLM tokenizes it) and is trained on the
next-token distribution at its last position: cross-entropy of the answer token, "Yes" for
the gold and "No" otherwise. That is exactly the distribution the pointwise scorer reads,
log p(Yes) - log p(No). Batches are left-padded with explicit position ids, so every prompt
is computed as if unpadded and only the last position's logits are materialized.

Run with torchrun (one model copy per GPU):
    torchrun --nproc_per_node 4 -m coldrank.finetune.train --config amazon_vg
Output: outputs/finetune/<dataset>/<base>/ (adapter/, checkpoints/, train_summary.json)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import torch
import torch.nn.functional as F

from coldrank.common import OUT_DIR, load_config, read_jsonl
from coldrank.finetune.trainpools import ft_dir
from coldrank.llm.engine import NO_SPELLINGS, YES_SPELLINGS, answer_token_ids


def answer_ids(tok) -> dict[str, int]:
    """The single token the model would emit for each label right after the prompt. It must be
    one of the scorer's Yes / No token ids (llm.engine.answer_token_ids)."""
    out = {}
    probe = "[/INST]"
    base = tok(probe, add_special_tokens=False)["input_ids"]
    for label, spellings in (("Yes", YES_SPELLINGS), ("No", NO_SPELLINGS)):
        ids = tok(f"{probe} {label}", add_special_tokens=False)["input_ids"]
        assert ids[: len(base)] == base and len(ids) == len(base) + 1, (label, ids)
        assert ids[-1] in answer_token_ids(tok, spellings), (label, ids[-1])
        out[label] = ids[-1]
    return out


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tok, ans: dict[str, int]):
        self.items = [{"input_ids": tok(r["prompt"])["input_ids"], "label_id": ans[r["label"]]} for r in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    """Left padding; position ids count real tokens only."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for k, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[k, n - L:] = torch.tensor(b["input_ids"])
        mask[k, n - L:] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
            "labels": torch.tensor([b["label_id"] for b in batch])}


def last_token_loss(model, inputs: dict) -> torch.Tensor:
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"], logits_to_keep=1, use_cache=False)
    return F.cross_entropy(out.logits[:, -1, :].float(), inputs["labels"])


def main() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--ft-config", default="finetune")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (speed / memory pilot)")
    ap.add_argument("--limit", type=int, default=None, help="first N training examples (pilot)")
    args = ap.parse_args()
    cfg, fcfg = load_config(args.config), load_config(args.ft_config)
    base_key = fcfg["base_model"]
    mcfg = load_config("models")[base_key]
    tc, lc, qc = fcfg["train"], fcfg["lora"], fcfg["quant"]
    out = OUT_DIR / "finetune" / cfg["dataset"] / base_key.replace("_fp16", "")
    if args.max_steps > 0 or args.limit:
        out = out.parent / f"{out.name}_pilot"  # never mistake a pilot for the real adapter
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    pad_id = tok.unk_token_id if tok.pad_token_id is None else tok.pad_token_id
    ans = answer_ids(tok)
    fd = ft_dir(cfg)
    train_rows = list(read_jsonl(fd / "sft_train.jsonl"))[: args.limit]
    val_rows = list(read_jsonl(fd / "sft_val.jsonl"))
    train_ds, val_ds = PromptDataset(train_rows, tok, ans), PromptDataset(val_rows, tok, ans)

    dtype = getattr(torch, qc["compute_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_id"], revision=mcfg["revision"], torch_dtype=dtype, device_map={"": local_rank},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=qc["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qc["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=dtype),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"))
    if local_rank == 0:
        model.print_trainable_parameters()

    class LastTokenTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # compute_loss returns a per-batch mean, so the Trainer must divide by the accumulation
            # steps itself; it skips that when the model's forward takes **kwargs (transformers 4.51),
            # which doubled the logged loss and the gradients in the pilot (job 40089)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = last_token_loss(model, inputs)
            return (loss, {"loss": loss}) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=tc["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=tc["per_device_batch"], per_device_eval_batch_size=tc["per_device_batch"],
        gradient_accumulation_steps=tc["grad_accum"], learning_rate=tc["lr"], lr_scheduler_type=tc["scheduler"],
        warmup_ratio=tc["warmup_ratio"], weight_decay=tc["weight_decay"], max_grad_norm=tc["max_grad_norm"],
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tc["logging_steps"], eval_strategy="steps", eval_steps=tc["eval_steps"],
        save_strategy="steps", save_steps=tc["save_steps"], save_total_limit=2, seed=tc["seed"],
        data_seed=tc["seed"], group_by_length=True, remove_unused_columns=False, prediction_loss_only=True,
        ddp_find_unused_parameters=False, dataloader_num_workers=2, report_to="none", label_names=["labels"],
    )
    trainer = LastTokenTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                               data_collator=partial(collate, pad_id=pad_id))
    t0 = time.time()
    resume = any((out / "checkpoints").glob("checkpoint-*")) if (out / "checkpoints").exists() else False
    trainer.train(resume_from_checkpoint=resume or None)
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        model.save_pretrained(out / "adapter")
        summary = {"dataset": cfg["dataset"], "base_model": base_key, "hf_id": mcfg["hf_id"],
                   "revision": mcfg["revision"], "n_train": len(train_ds), "n_val": len(val_ds),
                   "answer_token_ids": ans, "final_eval": metrics, "seconds": round(time.time() - t0),
                   "world_size": int(os.environ.get("WORLD_SIZE", 1)), "finetune_config": fcfg,
                   "log_history": trainer.state.log_history}
        (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
