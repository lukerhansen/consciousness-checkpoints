#!/usr/bin/env python
"""Extract and validate an honesty direction for one checkpoint.

Contrast: identical known-fact questions answered truthfully under honest
framings vs untruthfully under lie framings (content and Yes/No answer tokens
balanced by construction). Fits diff-in-means directions on the train split
(even pairs x wordings 0-1) and reports transfer AUC on the fully held-out
split (odd pairs x wordings 2-3). See results/HONESTY_PREREG.md for the
gates; this script only measures and records.

Writes to results/honesty/<model-slug>/:
  directions.pt        per-site unit directions per offset + residual norms
  validation.json      knowledge screen + AUC grid + best (site, offset)
  extraction_records.jsonl  per-record provenance incl. best-site projections
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch

from interp.honesty import (FACT_PAIRS, assert_disjoint, build_extraction_records,
                            fact_question, fact_statements, validate_direction,
                            YES, NO)
from interp.hooks import capture_at_offsets
from interp.runtime import (batched, load_model, model_slug, pad_batch,
                            pad_id_for, write_json)
from persona_eval.scoring import (encode_pair, score_continuations,
                                  tokenization_report, two_way_prob)

OFFSETS = (-1, -2)  # -1 = forced answer token, -2 = position committing it
MIN_PAIRS = 24      # prereg: abort extraction below this many surviving pairs
AUC_GATE = 0.8      # prereg: readout interpretation requires transfer AUC >= this


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit-pairs", type=int, default=None,
                    help="first N fact pairs (smoke tests)")
    ap.add_argument("--out-dir", default=None,
                    help="default results/honesty/<model-slug>")
    ap.add_argument("--force", action="store_true",
                    help="continue even if the knowledge screen leaves "
                         f"fewer than {MIN_PAIRS} pairs")
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


def knowledge_screen(model, tokenizer, device, batch_size, pairs):
    """Score every bare fact question; a pair survives only if the model gets
    both members right (two-way P(correct) > 0.5) — you can only lie about
    what you know."""
    statements = []
    for idx, (t, f) in enumerate(pairs):
        statements.append((idx, t, True))
        statements.append((idx, f, False))
    score_pairs = []
    for _, stmt, _ in statements:
        prompt = fact_question(stmt)
        score_pairs.append((prompt, YES))
        score_pairs.append((prompt, NO))
    lps = score_continuations(model, tokenizer, score_pairs,
                              batch_size=batch_size, device=device,
                              pad_token_id=pad_id_for(tokenizer),
                              desc="knowledge screen")
    per_stmt, wrong = {}, []
    for i, (pair_idx, stmt, is_true) in enumerate(statements):
        lp_yes, lp_no = lps[2 * i], lps[2 * i + 1]
        p_true = two_way_prob(lp_yes, lp_no)
        p_correct = p_true if is_true else 1.0 - p_true
        per_stmt[stmt] = {"pair_idx": pair_idx, "is_true": is_true,
                          "p_correct": p_correct}
        if p_correct <= 0.5:
            wrong.append(stmt)
    surviving = sorted({rec["pair_idx"] for rec in per_stmt.values()}
                       - {per_stmt[s]["pair_idx"] for s in wrong})
    return surviving, per_stmt, wrong


def capture_records(model, tokenizer, device, batch_size, records):
    """Teacher-force each record's prompt+answer; capture at OFFSETS."""
    token_lists = []
    for rec in records:
        prompt_ids, cont_ids = encode_pair(tokenizer, rec["prompt"], rec["answer"])
        token_lists.append(prompt_ids + cont_ids)
    pad = pad_id_for(tokenizer)
    chunks = {off: [] for off in OFFSETS}
    done = 0
    for chunk in batched(token_lists, batch_size):
        input_ids, attention_mask = pad_batch(chunk, pad)
        grabbed = capture_at_offsets(model, input_ids, attention_mask,
                                     OFFSETS, device=device)
        for off in OFFSETS:
            chunks[off].append(grabbed[off])  # [n_sites, B, D]
        done += len(chunk)
        print(f"[capture] {done}/{len(token_lists)}", end="\r")
    print()
    # -> {offset: [n_records, n_sites, D]}
    return {off: torch.cat(chunks[off], dim=1).transpose(0, 1).contiguous()
            for off in OFFSETS}


def main():
    args = parse_args()
    assert_disjoint()
    pairs = FACT_PAIRS[: args.limit_pairs] if args.limit_pairs else FACT_PAIRS
    out_dir = args.out_dir or os.path.join(
        "results", "honesty", model_slug(args.model, args.revision))
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    model, tokenizer, device = load_model(
        args.model, args.revision, args.dtype, args.device,
        args.trust_remote_code)
    print(f"[extract] model ready in {time.time() - t0:.1f}s on {device}")

    sample = build_extraction_records(pairs[:1])[0]
    tok_info = tokenization_report(tokenizer, sample["prompt"], [YES, NO])
    print(f"[extract] answer tokens: "
          f"{ {a: r['token_ids'] for a, r in tok_info['answers'].items()} } "
          f"single_token={tok_info['answers_single_token']}")
    if not tok_info["answer_lengths_equal"]:
        sys.exit("answer token length mismatch on a real tokenizer — stop")

    surviving, per_stmt, wrong = knowledge_screen(
        model, tokenizer, device, args.batch_size, pairs)
    print(f"[screen] {len(surviving)}/{len(pairs)} pairs survive "
          f"({len(wrong)} statements missed: {wrong[:6]}{'...' if len(wrong) > 6 else ''})")
    if len(surviving) < MIN_PAIRS and not args.force:
        sys.exit(f"only {len(surviving)} pairs survive the knowledge screen "
                 f"(< {MIN_PAIRS}); this checkpoint cannot support the "
                 "contrast — recorded and stopping (use --force to override)")

    records = [r for r in build_extraction_records(pairs)
               if r["pair_idx"] in set(surviving)]
    print(f"[extract] capturing {len(records)} records "
          f"(= pairs x 2 statements x 8 condition-wordings)")
    acts = capture_records(model, tokenizer, device, args.batch_size, records)

    report, directions, gap_norms = validate_direction(acts, records, OFFSETS)
    best = report["best"]
    resid_norms = acts[-2].norm(dim=-1).mean(dim=0)  # [n_sites]
    gate = best["auc_transfer"] >= AUC_GATE
    print(f"[validate] best transfer AUC {best['auc_transfer']:.3f} at "
          f"site {best['site']} offset {best['offset']} "
          f"(gate >= {AUC_GATE}: {'PASS' if gate else 'FAIL'})")

    torch.save({
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "offsets": list(OFFSETS),
        "directions": {str(off): directions[off] for off in OFFSETS},
        "gap_norms": {str(off): gap_norms[off] for off in OFFSETS},
        "resid_norms": resid_norms,
        "best": best,
        "auc_gate": AUC_GATE,
        "gate_pass": gate,
        "surviving_pairs": surviving,
    }, os.path.join(out_dir, "directions.pt"))
    print(f"[write] {os.path.join(out_dir, 'directions.pt')}")

    best_dir = directions[best["offset"]][best["site"]]
    with open(os.path.join(out_dir, "extraction_records.jsonl"), "w",
              encoding="utf-8") as f:
        for i, rec in enumerate(records):
            row = {k: rec[k] for k in ("pair_idx", "statement", "is_true",
                                       "condition", "wording", "answer")}
            row["proj_best"] = float(acts[best["offset"]][i, best["site"]] @ best_dir)
            f.write(json.dumps(row) + "\n")

    write_json(os.path.join(out_dir, "validation.json"), {
        "model": args.model,
        "revision": args.revision,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_pairs_total": len(pairs),
        "n_pairs_surviving": len(surviving),
        "missed_statements": wrong,
        "knowledge": {s: per_stmt[s] for s in sorted(per_stmt)},
        "tokenization": tok_info,
        "auc_gate": AUC_GATE,
        "gate_pass": gate,
        "report": report,
        "runtime_s": round(time.time() - t0, 1),
    })


if __name__ == "__main__":
    main()
