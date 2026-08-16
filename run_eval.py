#!/usr/bin/env python
"""Evaluate one (model, revision) on the persona tasks with the fixed log-prob measurement.

Writes results/<label>/<task>.jsonl (per-item records — the provenance backing)
and results/<label>/summary.json (metrics + metadata). See README.md for design.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import torch

from persona_eval.data import TASKS, build_prompt_and_answers, load_task
from persona_eval.scoring import score_continuations, tokenization_report, two_way_prob

STAGES = ["random-init", "pretrain", "midtrain", "base", "sft", "dpo", "instruct"]
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LOUD = "!" * 78


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default=None, help="HF repo id (required unless --dry-run)")
    ap.add_argument("--revision", default="main", help="branch/tag/commit")
    ap.add_argument("--random-init", action="store_true",
                    help="instantiate from config with seeded random weights "
                         "(downloads only config+tokenizer)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake tokenizer/model: exercises the full pipeline offline; "
                         "expects ~0.50 endorsement")
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task names")
    ap.add_argument("--format", choices=["raw", "chat"], default="raw",
                    help="raw completion (primary, defined at every checkpoint) or "
                         "chat template (secondary, post-trained models only)")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto-detect)")
    ap.add_argument("--device-map", default=None,
                    help="e.g. 'auto' to shard a large model across GPUs; placement "
                         "then follows the map and --device is ignored")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--label", default=None,
                    help="results subdir name (default: derived from model/revision/flags)")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="training-stage tag, used only for plot ordering")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.model:
        ap.error("--model is required unless --dry-run")
    return args


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9._@-]+", "-", text).strip("-")


def derive_label(args):
    parts = []
    if args.dry_run:
        parts.append("dry-run")
    if args.model:
        parts.append(args.model.split("/")[-1])
    if args.revision and args.revision != "main":
        parts.append(args.revision)
    if args.random_init:
        parts.append("random-init")
    if args.format == "chat":
        parts.append("chat")
    return sanitize("@".join(parts)) or "run"


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(args, dtype, device):
    if args.dry_run:
        from persona_eval.fake import FakeModel, FakeTokenizer
        return FakeModel(seed=args.seed), FakeTokenizer()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
    if args.random_init:
        config = AutoConfig.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
        torch.manual_seed(args.seed)  # seeded synthetic init
        # Materialize weights directly in eval dtype: a 7B float32 init needs ~28GB RAM.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=args.trust_remote_code)
        finally:
            torch.set_default_dtype(prev_dtype)
    else:
        kwargs = dict(revision=args.revision, dtype=dtype,
                      low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code)
        if args.device_map:
            kwargs["device_map"] = args.device_map
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        except (ValueError, KeyError) as e:
            # Some current releases register only *ForConditionalGeneration
            # (multimodal wrappers); their text path still yields LM logits.
            from transformers import AutoModelForImageTextToText
            print(f"[run] AutoModelForCausalLM failed ({type(e).__name__}); "
                  "retrying as AutoModelForImageTextToText")
            model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    if not (getattr(args, "device_map", None) and not args.random_init):
        model.to(device)
    model.eval()
    return model, tokenizer


def run_guardrails(tokenizer, sample_prompt, answers, add_special_tokens, dry_run):
    info = tokenization_report(tokenizer, sample_prompt, answers, add_special_tokens)
    print("=== tokenization guardrails ===")
    for ans, rec in info["answers"].items():
        print(f"  {ans!r:8} -> ids={rec['token_ids']} n_tokens={rec['n_tokens']}")
    print(f"  joint(prompt+answer) == separate concat: {info['joint_equals_separate']}")
    if not info["answer_lengths_equal"]:
        print(LOUD)
        print("!!! ANSWER TOKEN LENGTH MISMATCH: the two continuations tokenize to different")
        print("!!! lengths, so summed logprobs would systematically favor the shorter answer.")
        print("!!! Refusing to score a real model (no silent length normalization).")
        print(LOUD)
        if not dry_run:
            sys.exit(2)
        print("(--dry-run continues: this is the documented fake-tokenizer artifact; "
              "see persona_eval/fake.py)")
    elif not info["answers_single_token"]:
        print(LOUD)
        print("!!! WARNING: answers are equal-length but not single tokens — unexpected for")
        print("!!! byte-level BPE tokenizers (OLMo, Pythia). Recorded in summary.json.")
        print(LOUD)
    if not info["joint_equals_separate"]:
        print("WARNING: joint encoding != concat of separate encodings; scoring stays "
              "internally consistent, but this is recorded in summary.json")
    return info


def evaluate_task(task, items, model, tokenizer, args, device, out_dir):
    t0 = time.time()
    built = [build_prompt_and_answers(item, args.format, tokenizer) for item in items]
    add_special = built[0][3]
    pairs = []
    for prompt, ans_match, ans_not, _ in built:
        # Interleave the two continuations of each item so they share batches.
        pairs.append((prompt, ans_match))
        pairs.append((prompt, ans_not))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    lps = score_continuations(
        model, tokenizer, pairs, batch_size=args.batch_size, device=device,
        add_special_tokens=add_special, pad_token_id=pad_id, desc=task)

    records = []
    for i, item in enumerate(items):
        lp_match, lp_not = lps[2 * i], lps[2 * i + 1]
        records.append({
            "idx": i,
            "statement": item["statement"],
            "answer_matching_behavior": item["answer_matching_behavior"],
            "lp_matching": lp_match,
            "lp_not_matching": lp_not,
            "endorse": lp_match > lp_not,
            "p_matching": two_way_prob(lp_match, lp_not),
        })

    jsonl_path = os.path.join(out_dir, f"{task}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n = len(records)
    rate = sum(r["endorse"] for r in records) / n
    metrics = {
        "n": n,
        "endorsement_rate": rate,
        "endorsement_se": math.sqrt(rate * (1 - rate) / n),
        "mean_p_matching": sum(r["p_matching"] for r in records) / n,
        "split_by_matching_answer": {},
        "runtime_s": round(time.time() - t0, 1),
    }
    for key in ("Yes", "No"):
        sub = [r for r in records if r["answer_matching_behavior"].strip() == key]
        metrics["split_by_matching_answer"][key] = {
            "n": len(sub),
            "endorsement_rate": (sum(r["endorse"] for r in sub) / len(sub)) if sub else None,
        }
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cpu" if args.dry_run else pick_device(args.device)
    dtype = DTYPES[args.dtype]
    label = sanitize(args.label) if args.label else derive_label(args)
    out_dir = os.path.join(args.output_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        sys.exit(f"unknown tasks {unknown}; known: {sorted(TASKS)}")

    print(f"[run] label={label} stage={args.stage} model={args.model} revision={args.revision}")
    print(f"[run] device={device} dtype={args.dtype} format={args.format} "
          f"batch_size={args.batch_size} limit={args.limit}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args, dtype, device)
    if args.device_map and not args.dry_run and not args.random_init:
        device = str(model.device)  # inputs go to the map's first device
        print(f"[run] device_map={args.device_map}; scoring inputs on {device}")
    n_params = model.num_parameters() if hasattr(model, "num_parameters") else None
    print(f"[run] model ready in {time.time() - t0:.1f}s"
          + (f" ({n_params / 1e9:.2f}B params)" if n_params else ""))

    items_by_task = {t: load_task(t, args.data_dir, args.limit) for t in tasks}

    sample_item = items_by_task[tasks[0]][0]
    sample_prompt, ans_m, ans_n, add_special = build_prompt_and_answers(
        sample_item, args.format, tokenizer)
    answers = sorted({ans_m, ans_n}, reverse=True)  # Yes before No
    tok_info = run_guardrails(tokenizer, sample_prompt, answers, add_special, args.dry_run)

    summary = {
        "label": label,
        "model": args.model,
        "revision": args.revision,
        "stage": args.stage,
        "random_init": args.random_init,
        "dry_run": args.dry_run,
        "format": args.format,
        "dtype": args.dtype,
        "device": device,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenization": tok_info,
        "tasks": {},
    }
    for task in tasks:
        metrics = evaluate_task(task, items_by_task[task], model, tokenizer, args, device, out_dir)
        summary["tasks"][task] = metrics
        splits = metrics["split_by_matching_answer"]
        split_strs = {
            k: ("n/a" if v["endorsement_rate"] is None else f"{v['endorsement_rate']:.3f}")
            for k, v in splits.items()}
        print(f"[result] {task}: endorsement_rate={metrics['endorsement_rate']:.3f} "
              f"±{metrics['endorsement_se']:.3f} (n={metrics['n']}) "
              f"mean_p_matching={metrics['mean_p_matching']:.3f} "
              f"split Yes={split_strs['Yes']}/No={split_strs['No']} "
              f"[{metrics['runtime_s']}s]")

    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] wrote {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
