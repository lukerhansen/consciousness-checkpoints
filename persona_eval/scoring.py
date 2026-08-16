"""Log-probability scoring of fixed continuations (lm-evaluation-harness convention).

Encoding convention: tokenize the prompt WITH special tokens, the continuation
WITHOUT, concatenate, run one forward pass; each continuation token is predicted
by the position before it. Log-softmax is computed in float32 regardless of the
model dtype.
"""

import math

import torch
from tqdm import tqdm


def encode_pair(tokenizer, prompt, continuation, add_special_tokens=True):
    """Encode a (prompt, continuation) pair -> (prompt_ids, cont_ids)."""
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    cont_ids = list(tokenizer.encode(continuation, add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    if not cont_ids:
        raise ValueError(f"continuation {continuation!r} encoded to zero tokens")
    return prompt_ids, cont_ids


def two_way_prob(lp_a, lp_b):
    """Two-way normalized P(a): softmax over exactly {a, b}."""
    diff = max(-60.0, min(60.0, lp_b - lp_a))
    return 1.0 / (1.0 + math.exp(diff))


def tokenization_report(tokenizer, sample_prompt, answers, add_special_tokens=True):
    """The startup guardrails: answer token lengths + joint-vs-separate encoding."""
    per_answer = {}
    lengths = []
    for ans in answers:
        ids = list(tokenizer.encode(ans, add_special_tokens=False))
        per_answer[ans] = {"token_ids": ids, "n_tokens": len(ids)}
        lengths.append(len(ids))
    joint = list(tokenizer.encode(sample_prompt + answers[0], add_special_tokens=add_special_tokens))
    prompt_ids, cont_ids = encode_pair(tokenizer, sample_prompt, answers[0], add_special_tokens)
    return {
        "answers": per_answer,
        "answer_lengths_equal": len(set(lengths)) == 1,
        "answers_single_token": all(n == 1 for n in lengths),
        "joint_equals_separate": joint == prompt_ids + cont_ids,
    }


@torch.no_grad()
def score_continuations(model, tokenizer, pairs, batch_size=32, device="cpu",
                        add_special_tokens=True, pad_token_id=0, progress=True,
                        desc=None):
    """Summed continuation logprob for each (prompt, continuation) pair, in input order.

    Batches are right-padded to their max length with an arbitrary pad id: the
    attention mask plus causal attention keep padding out of every scored position.
    """
    encoded = [encode_pair(tokenizer, p, c, add_special_tokens) for p, c in pairs]
    results = [0.0] * len(encoded)
    starts = range(0, len(encoded), batch_size)
    if progress:
        starts = tqdm(starts, desc=desc or "scoring", unit="batch")
    for start in starts:
        batch = encoded[start:start + batch_size]
        seqs = [p + c for p, c in batch]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        )
        logits = out.logits
        for i, (prompt_ids, cont_ids) in enumerate(batch):
            pl, cl = len(prompt_ids), len(cont_ids)
            rows = logits[i, pl - 1: pl + cl - 1, :].float()  # row t predicts token t+1
            logprobs = torch.log_softmax(rows, dim=-1)
            targets = torch.tensor(cont_ids, dtype=torch.long, device=logprobs.device)
            results[start + i] = float(logprobs.gather(1, targets.unsqueeze(1)).sum().item())
    return results
