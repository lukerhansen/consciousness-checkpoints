"""Shared runtime for the honesty-lens CLIs: model loading via run_eval's
exact code path (so dtype/device/guardrail behavior cannot drift from the
published measurement), plus small helpers used by every script.
"""

import hashlib
import json
import os
from argparse import Namespace

import torch

from run_eval import DTYPES, load_model_and_tokenizer, pick_device


def load_model(model_id, revision="main", dtype="bfloat16", device=None,
               trust_remote_code=False):
    """Load (model, tokenizer, device) exactly as run_eval.py does."""
    device = pick_device(device)
    args = Namespace(model=model_id, revision=revision, dry_run=False,
                     random_init=False, trust_remote_code=trust_remote_code,
                     device_map=None)
    model, tokenizer = load_model_and_tokenizer(args, DTYPES[dtype], device)
    return model, tokenizer, device


def batched(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def pad_batch(token_lists, pad_id):
    """Right-pad to a rectangle -> (input_ids, attention_mask) long tensors."""
    max_len = max(len(t) for t in token_lists)
    input_ids = torch.full((len(token_lists), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(token_lists), max_len), dtype=torch.long)
    for i, toks in enumerate(token_lists):
        input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        attention_mask[i, : len(toks)] = 1
    return input_ids, attention_mask


def pad_id_for(tokenizer):
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None) or 0
    return pad


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[write] {path}")


def model_slug(model_id, revision="main"):
    slug = model_id.split("/")[-1]
    if revision and revision != "main":
        slug += f"@{revision}"
    return slug
