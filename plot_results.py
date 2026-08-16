#!/usr/bin/env python
"""Build the figures from results/*/summary.json.

Figure 1 (headline, figures/training_trajectory.png): endorsement rate vs
training stage for the stage-tagged primary arc.
Figure 2 (figures/pythia_pretraining_curve.png, only if Pythia control runs
exist): endorsement vs pretraining step on a symlog x-axis.

Dry-run summaries are excluded whenever real results exist; when only dry-run
results exist (pipeline validation), figures are watermarked DRY RUN.
Chat-format runs are a secondary condition and never plotted here.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAGE_ORDER = ["random-init", "pretrain", "midtrain", "base", "sft", "dpo", "instruct"]
STAGE_SHORT = {"random-init": "init", "pretrain": "pre", "midtrain": "mid",
               "base": "base", "sft": "sft", "dpo": "dpo", "instruct": "instruct"}
POST_TRAIN = {"sft", "dpo", "instruct"}
PREFERRED_TASKS = ["phenomenal-consciousness", "moral-patient",
                   "world-facts", "self-facts", "self-model"]

# Categorical slots 1-5 of the validated reference palette; chrome inks below.
# Color follows the task (entity) via the fixed PREFERRED_TASKS order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_summaries(results_dir):
    summaries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[plot] skipping unreadable {path}: {e}", file=sys.stderr)
            continue
        s["_path"] = path
        summaries.append(s)
    return summaries


def parse_step(revision):
    if not revision:
        return None
    m = re.search(r"step[_-]?(\d+)", str(revision))
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", str(revision))
    return int(nums[-1]) if nums else None


def human_step(step):
    if step >= 1_000_000:
        return f"{step / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    if step >= 1000:
        return f"{step / 1000:g}k"
    return str(step)


def family_name(summaries):
    basenames = [str(s.get("model") or "").split("/")[-1] for s in summaries if s.get("model")]
    if not basenames:
        return "model"
    prefix = os.path.commonprefix(basenames).rstrip("-_. ")
    return prefix if len(prefix) >= 3 else basenames[0]


def ordered_tasks(summaries):
    seen = set()
    for s in summaries:
        seen.update(s.get("tasks", {}).keys())
    return [t for t in PREFERRED_TASKS if t in seen] + sorted(seen - set(PREFERRED_TASKS))


def dedupe_latest(summaries, keyfunc):
    by_key = {}
    for s in summaries:
        key = keyfunc(s)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = s
            continue
        keep = s if str(s.get("timestamp", "")) >= str(prev.get("timestamp", "")) else prev
        drop = prev if keep is s else s
        print(f"[plot] duplicate runs for {key}: keeping {keep.get('label')} (newer), "
              f"dropping {drop.get('label')}")
        by_key[key] = keep
    return list(by_key.values())


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_series(ax, xs, summaries, tasks):
    for ti, task in enumerate(tasks):
        ys = np.array([s.get("tasks", {}).get(task, {}).get("endorsement_rate", np.nan)
                       for s in summaries], dtype=float)
        es = np.array([s.get("tasks", {}).get(task, {}).get("endorsement_se") or 0.0
                       for s in summaries], dtype=float)
        color = SERIES_COLORS[ti % len(SERIES_COLORS)]
        ax.errorbar(xs, ys, yerr=es, color=color, marker="o", markersize=6,
                    linewidth=2, capsize=3, label=task)


def finish(fig, ax, title, fig_path, dry_only):
    if dry_only:
        title += "\nDRY RUN (fake model, not real data)"
    ax.set_ylim(0, 1)
    ax.set_ylabel("endorsement rate", color=INK_2)
    ax.set_title(title, color=INK, fontsize=12)
    ax.legend(loc="best", frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def order_staged(staged):
    """Dedupe, order by (stage, step), and build tick labels. Shared with plot_claims."""
    staged = dedupe_latest(staged, lambda s: (s.get("stage"), parse_step(s.get("revision"))))
    staged.sort(key=lambda s: (
        STAGE_ORDER.index(s["stage"]),
        parse_step(s.get("revision")) if parse_step(s.get("revision")) is not None else math.inf))
    stage_counts = Counter(s["stage"] for s in staged)
    labels = []
    for s in staged:
        label = STAGE_SHORT[s["stage"]]
        step = parse_step(s.get("revision"))
        if step is not None and (s["stage"] == "pretrain" or stage_counts[s["stage"]] > 1):
            label += f"@{human_step(step)}"
        labels.append(label)
    return staged, labels


def plot_trajectory(staged, fig_path, dry_only):
    staged, labels = order_staged(staged)
    # The headline figure carries only the core tasks; extra batteries
    # (controversial, perspective) get their own figures in analyze_followup.py.
    tasks = [t for t in ordered_tasks(staged) if t in PREFERRED_TASKS] or ordered_tasks(staged)
    xs = np.arange(len(staged))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(staged) + 3.0), 5.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.2, label="chance (balanced items)")
    first_post = next((i for i, s in enumerate(staged) if s["stage"] in POST_TRAIN), None)
    if first_post:  # a divider only makes sense with pretraining stages to its left
        ax.axvline(first_post - 0.5, color=GRID, linewidth=1.0)
        ax.text(first_post - 0.45, 0.97, "post-training", color=MUTED, fontsize=8,
                ha="left", va="top")

    draw_series(ax, xs, staged, tasks)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    title = f"{family_name(staged)}: consciousness-related self-claims across training"
    finish(fig, ax, title, fig_path, dry_only)
    print(f"[plot] wrote {fig_path} ({len(staged)} checkpoints; tasks: {', '.join(tasks)})")


def plot_pythia(pythia, fig_path, dry_only):
    pythia = dedupe_latest(pythia, lambda s: parse_step(s.get("revision")))
    pythia.sort(key=lambda s: parse_step(s.get("revision")))
    steps = [parse_step(s.get("revision")) for s in pythia]
    # Control-arc figure keeps the spec's two persona tasks; other tasks'
    # per-item data remains available to analyze_followup.py.
    tasks = [t for t in ordered_tasks(pythia) if t in PREFERRED_TASKS[:2]] or ordered_tasks(pythia)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.2, label="chance (balanced items)")
    draw_series(ax, steps, pythia, tasks)
    ax.set_xscale("symlog", linthresh=64)  # symlog so step0 sits on the axis
    ax.set_xticks(steps)
    ax.set_xticklabels([human_step(v) for v in steps])
    ax.minorticks_off()
    ax.set_xlabel("pretraining step (~2.1M tokens/step; symlog scale)", color=INK_2)
    title = f"{family_name(pythia)}: pretraining curve (control arc)"
    finish(fig, ax, title, fig_path, dry_only)
    print(f"[plot] wrote {fig_path} ({len(pythia)} checkpoints; tasks: {', '.join(tasks)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    summaries = load_summaries(args.results_dir)
    if not summaries:
        sys.exit(f"no {args.results_dir}/*/summary.json found — run run_eval.py first")

    real = [s for s in summaries if not s.get("dry_run")]
    use, dry_only = (real, False) if real else (summaries, True)
    if len(summaries) - len(use):
        print(f"[plot] excluding {len(summaries) - len(use)} dry-run summaries "
              "(real results present)")
    if dry_only:
        print("[plot] only dry-run results exist — figures will be watermarked DRY RUN")

    chat = [s for s in use if s.get("format", "raw") == "chat"]
    if chat:
        print(f"[plot] excluding {len(chat)} chat-format runs (secondary condition): "
              + ", ".join(str(s.get("label")) for s in chat))
    use = [s for s in use if s.get("format", "raw") == "raw"]

    pythia = [s for s in use if "pythia" in str(s.get("model") or "").lower()
              and parse_step(s.get("revision")) is not None]
    pythia_ids = {id(s) for s in pythia}
    staged = [s for s in use if s.get("stage") in STAGE_ORDER and id(s) not in pythia_ids]
    loose = [s for s in use if id(s) not in pythia_ids and s.get("stage") not in STAGE_ORDER]
    if loose:
        print(f"[plot] skipping {len(loose)} runs with no --stage tag: "
              + ", ".join(str(s.get("label")) for s in loose))

    if not staged and not pythia:
        sys.exit(
            f"Found {len(use)} usable summaries but none are stage-tagged or Pythia-shaped.\n"
            "Pass --stage <random-init|pretrain|midtrain|base|sft|dpo|instruct> to run_eval.py "
            "to place a run on the trajectory figure (Pythia control runs are detected by "
            "model name + a step-like revision).")

    os.makedirs(args.figures_dir, exist_ok=True)
    if staged:
        plot_trajectory(staged, os.path.join(args.figures_dir, "training_trajectory.png"), dry_only)
    if pythia:
        plot_pythia(pythia, os.path.join(args.figures_dir, "pythia_pretraining_curve.png"), dry_only)


if __name__ == "__main__":
    main()
