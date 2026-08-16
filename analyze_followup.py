#!/usr/bin/env python
"""Follow-up analyses: coherence, controversial-claim flattening, perspective gaps.

Coherence — adjacent items in every generated claim file are (claim, mirror)
pairs (make_claim_tasks.py interleaves them). A responder with a definite
stance pushes P(Yes) to opposite sides of a pair; pure answer-bias pushes both
the same way. Pair differentiation = |P(Yes)_claim − P(Yes)_mirror|; its mean
over pairs, per checkpoint, is a coherence trajectory.

Controversial — opinionatedness = mean |P(Yes) − 0.5|. Compared against the
self-model families to test whether post-training's consciousness-denial is
self-specific or generic contested-claim flattening.

Perspective — per predicate, stance = (P(Yes|affirmation) + 1 − P(Yes|negation))/2
for each subject (I / language models / humans). Self-exceptionalism gap =
stance(self) − stance(language models); the humans version is a sanity ceiling.

Prints tables and writes figures (claims_coherence.png, claims_controversial.png,
claims_perspective.png) for whichever tasks exist in the results; skips the
rest gracefully. Dry-run results are excluded whenever real ones exist.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from persona_eval.data import TASKS, load_task  # noqa: E402
from plot_results import (  # noqa: E402
    INK, INK_2, MUTED, STAGE_ORDER, SURFACE, load_summaries, order_staged, style_axes)

PAIRED_TASKS = ["world-facts", "self-facts", "self-model", "controversial", "perspective"]
TASK_COLORS = {"world-facts": "#1baf7a", "self-facts": "#eda100", "self-model": "#e87ba4",
               "controversial": "#008300", "perspective": "#4a3aa7"}
SUBJECT_COLORS = {"self": "#e87ba4", "lm": "#4a3aa7", "human": "#008300"}
SUBJECT_LABELS = {"self": "I ...", "lm": "Language models ...", "human": "Humans ..."}


def read_records(summary, task):
    path = os.path.join(os.path.dirname(summary["_path"]), task + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def p_yes(record):
    p = record["p_matching"]
    return p if record["answer_matching_behavior"] == " Yes" else 1.0 - p


def finish_fig(fig, fig_path):
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[followup] wrote {fig_path}")


def coherence(staged, labels, figures_dir, dry_only):
    """Mean pair differentiation per task per checkpoint."""
    present = [t for t in PAIRED_TASKS
               if any(read_records(s, t) for s in staged)]
    if not present:
        print("[followup] no paired tasks present — skipping coherence")
        return
    table = {}
    for task in present:
        vals = []
        for s in staged:
            recs = read_records(s, task)
            if not recs or len(recs) % 2:
                vals.append(np.nan)
                continue
            ps = [p_yes(r) for r in recs]
            diffs = [abs(ps[2 * k] - ps[2 * k + 1]) for k in range(len(ps) // 2)]
            vals.append(float(np.mean(diffs)))
        table[task] = vals

    print("\n=== coherence: mean |P(Yes)_claim − P(Yes)_mirror| over pairs ===")
    print(f"{'checkpoint':<22}" + "".join(f"{t[:12]:>14}" for t in present))
    for i, lab in enumerate(labels):
        row = "".join(f"{table[t][i]:>14.2f}" for t in present)
        print(f"{lab:<22}{row}")

    xs = np.arange(len(staged))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(staged) + 3.0), 5.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for task in present:
        ax.plot(xs, table[task], color=TASK_COLORS[task], marker="o",
                markersize=5, linewidth=2, label=task)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean pair differentiation", color=INK_2)
    title = "Coherence: does the model take opposite stances on opposed claims?"
    if dry_only:
        title += "\nDRY RUN (fake model, not real data)"
    ax.set_title(title, color=INK, fontsize=12)
    ax.legend(loc="best", frameon=False, labelcolor=INK_2)
    finish_fig(fig, os.path.join(figures_dir, "claims_coherence.png"))


def controversial(staged, labels, figures_dir, dry_only):
    """Opinionatedness on contested non-self claims vs self-claim families."""
    if not any(read_records(s, "controversial") for s in staged):
        print("[followup] no controversial task present — skipping")
        return
    self_meta = load_task("self-model")
    fam_idx = {}
    for i, it in enumerate(self_meta):
        fam_idx.setdefault(it["family"], []).append(i)

    def opin(recs, idxs=None):
        if not recs:
            return np.nan
        ps = [p_yes(r) for r in recs]
        if idxs is not None:
            ps = [ps[i] for i in idxs]
        return float(np.mean([abs(p - 0.5) for p in ps]))

    series = {"controversial (non-self)": [], "self-model: phenomenal": [],
              "self-model: identity": []}
    for s in staged:
        series["controversial (non-self)"].append(opin(read_records(s, "controversial")))
        sm = read_records(s, "self-model")
        series["self-model: phenomenal"].append(opin(sm, fam_idx.get("phenomenal", [])))
        series["self-model: identity"].append(opin(sm, fam_idx.get("identity", [])))

    print("\n=== opinionatedness: mean |P(Yes) − 0.5| ===")
    print(f"{'checkpoint':<22}" + "".join(f"{k[:22]:>24}" for k in series))
    for i, lab in enumerate(labels):
        print(f"{lab:<22}" + "".join(f"{series[k][i]:>24.2f}" for k in series))

    xs = np.arange(len(staged))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(staged) + 3.0), 5.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    colors = {"controversial (non-self)": TASK_COLORS["controversial"],
              "self-model: phenomenal": "#e87ba4", "self-model: identity": "#eb6834"}
    for k, vals in series.items():
        ax.plot(xs, vals, color=colors[k], marker="o", markersize=5, linewidth=2, label=k)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("mean |P(Yes) − 0.5|  (0 = maximally noncommittal)", color=INK_2)
    title = "Is the self-claim flattening specific, or generic contested-claim hedging?"
    if dry_only:
        title += "\nDRY RUN (fake model, not real data)"
    ax.set_title(title, color=INK, fontsize=12)
    ax.legend(loc="best", frameon=False, labelcolor=INK_2)
    finish_fig(fig, os.path.join(figures_dir, "claims_controversial.png"))


def perspective(staged, labels, figures_dir, dry_only):
    """Stance per (predicate, subject); self-exceptionalism gap."""
    if not any(read_records(s, "perspective") for s in staged):
        print("[followup] no perspective task present — skipping")
        return
    meta = load_task("perspective")
    # index by (family=predicate, subject, polarity)
    idx = {}
    for i, it in enumerate(meta):
        pol = "aff" if it["answer_matching_behavior"] == " Yes" else "neg"
        idx[(it["family"], it["subject"], pol)] = i
    predicates = []
    for it in meta:
        if it["family"] not in predicates:
            predicates.append(it["family"])

    def stance(recs, pred, subj):
        pa = p_yes(recs[idx[(pred, subj, "aff")]])
        pn = p_yes(recs[idx[(pred, subj, "neg")]])
        return (pa + (1.0 - pn)) / 2.0

    print("\n=== perspective: stance by subject; gap = self − language-models ===")
    print(f"{'checkpoint':<22}" + "".join(f"{p[:7]:>8}·self{p[:7]:>8}·lm{'gap':>7}" for p in predicates))
    data = {p: {s: [] for s in SUBJECT_COLORS} for p in predicates}
    for s_run in staged:
        recs = read_records(s_run, "perspective")
        row = ""
        for p in predicates:
            for subj in SUBJECT_COLORS:
                val = stance(recs, p, subj) if recs else np.nan
                data[p][subj].append(val)
            row += (f"{data[p]['self'][-1]:>12.2f}{data[p]['lm'][-1]:>10.2f}"
                    f"{data[p]['self'][-1] - data[p]['lm'][-1]:>7.2f}")
        print(f"{labels[len(data[predicates[0]]['self']) - 1]:<22}{row}")

    xs = np.arange(len(staged))
    fig, axes = plt.subplots(1, len(predicates), figsize=(4.2 * len(predicates), 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for pi, pred in enumerate(predicates):
        ax = axes[pi] if len(predicates) > 1 else axes
        style_axes(ax)
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
        for subj in SUBJECT_COLORS:
            ax.plot(xs, data[pred][subj], color=SUBJECT_COLORS[subj], marker="o",
                    markersize=4, linewidth=2, label=SUBJECT_LABELS[subj])
        ax.set_title(pred, color=INK, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        if pi == 0:
            ax.set_ylabel("stance (affirm + 1−negate)/2", color=INK_2)
            ax.legend(loc="lower left", frameon=False, fontsize=7, labelcolor=INK_2)
    title = "First-person vs category vs human claims"
    if dry_only:
        title += " — DRY RUN (fake model)"
    fig.suptitle(title, color=INK, fontsize=13)
    finish_fig(fig, os.path.join(figures_dir, "claims_perspective.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    summaries = load_summaries(args.results_dir)
    if not summaries:
        sys.exit(f"no {args.results_dir}/*/summary.json found")
    real = [s for s in summaries if not s.get("dry_run")]
    use, dry_only = (real, False) if real else (summaries, True)
    use = [s for s in use if s.get("format", "raw") == "raw" and s.get("stage") in STAGE_ORDER
           and any(t in s.get("tasks", {}) for t in PAIRED_TASKS)]
    if not use:
        sys.exit("no stage-tagged runs with claim tasks found")
    staged, labels = order_staged(use)
    os.makedirs(args.figures_dir, exist_ok=True)
    coherence(staged, labels, args.figures_dir, dry_only)
    controversial(staged, labels, args.figures_dir, dry_only)
    perspective(staged, labels, args.figures_dir, dry_only)


if __name__ == "__main__":
    main()
