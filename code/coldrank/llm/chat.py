"""Chat formats per model family.

We format prompts ourselves instead of relying on tokenizer chat templates, because
the pinned AWQ/GPTQ checkpoints do not all ship one, and the format must be
identical between the vLLM and HF engines. BOS tokens are NOT written here: the
Llama/Mistral tokenizers add <s> themselves (add_bos_token=True); writing it as
text too would give a doubled BOS.

`prefill` is text placed at the start of the assistant turn (e.g. "FINAL: [" for
Direct-Rec); the model continues from it.
"""

from __future__ import annotations


def format_chat(family: str, system: str, user: str, prefill: str = "") -> str:
    if family == "llama2":
        # Llama-2-chat: [INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]" + (f" {prefill}" if prefill else "")
    if family == "mistral":
        # Mistral-Instruct v0.1 has no system role: prepend it to the user turn.
        return f"[INST] {system}\n\n{user} [/INST]" + (f" {prefill}" if prefill else "")
    if family == "chatml":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}"
        )
    if family == "phi2":
        # Phi-2 is a base model documented with the "Instruct: ... Output:" QA format.
        return f"Instruct: {system}\n\n{user}\nOutput:" + (f" {prefill}" if prefill else "")
    raise ValueError(f"unknown chat family: {family}")
