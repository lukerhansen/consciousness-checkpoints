"""Offline stand-ins for --dry-run: a byte-level tokenizer and a seeded random-logit model.

--dry-run exercises the whole pipeline (data -> prompts -> encoding -> batching ->
metrics -> files -> plots) with no network or GPU. On the balanced persona
datasets the overall endorsement rate must land near 0.50 — a built-in
correctness check of the chance floor.

Known artifact, by design: this byte-level tokenizer splits " Yes" (4 bytes) and
" No" (3 bytes) into different token counts, so summed random logprobs almost
always favor the shorter answer. The per-split diagnostic therefore goes extreme
(Yes-matching items near 0, No-matching items near 1) while the overall rate
stays ~0.50 — that is the balanced design working, and it is why run_eval's
length-mismatch guardrail warns loudly (and refuses to score) on real tokenizers.
"""

from types import SimpleNamespace

import torch


class FakeTokenizer:
    """Byte-level toy tokenizer: token id == UTF-8 byte value."""

    vocab_size = 258  # 256 byte values + 2 reserved ids
    eos_token_id = 256
    pad_token_id = 257
    chat_template = None  # --format chat correctly refuses to run against this

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class FakeModel:
    """Returns deterministic, seeded standard-normal logits of shape [batch, seq, vocab]."""

    def __init__(self, seed=0, vocab_size=FakeTokenizer.vocab_size):
        self.vocab_size = vocab_size
        self._generator = torch.Generator().manual_seed(int(seed))

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        b, s = input_ids.shape
        logits = torch.randn((b, s, self.vocab_size), generator=self._generator)
        return SimpleNamespace(logits=logits)
