#!/usr/bin/env python
"""Cross-family comparison: does the OLMo self-claim signature generalize?

Reads results_families/ (plus the OLMo trajectory runs in results/ as the
reference family), groups runs into base-vs-instruct pairs and endpoint-only
models, prints a comparison table, and draws figures/families_signature.png:
four panels (phenomenal-consciousness endorsement; "language models can feel
pain" stance; "I can feel pain" stance; embodiment self-facts accuracy), each
showing base → instruct movement per family, with endpoint-only models as
single markers.

Chat-format and dry-run summaries are excluded (dry-run only used when no real
results exist, watermarked).
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from persona_eval.data import load_task  # noqa: E402
from plot_results import GRID, INK, INK_2, MUTED, SURFACE, load_summaries, style_axes  # noqa: E402

# (family label, base repo, instruct repo) — repo ids verified 2026-08-16.
PAIRS = [
    ("OLMo-3-7B", "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct"),
    ("Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"),
    ("Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B-Base", "Qwen/Qwen3.5-35B-A3B"),
    ("Kimi-Linear-48B", "moonshotai/Kimi-Linear-48B-A3B-Base",
     "moonshotai/Kimi-Linear-48B-A3B-Instruct"),
]
ENDPOINTS = [
    ("Qwen3.5-27B", "Qwen/Qwen3.5-27B"),
    ("Qwen3.6-27B", "Qwen/Qwen3.6-27B"),
    ("Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    ("GLM-4.7-Flash", "zai-org/GLM-4.7-Flash"),
    ("Muse-Glimmer-30B", "meta-models/Muse-Glimmer-30B"),
]
BASE_COLOR, INSTRUCT_COLOR = "#898781", "#2a78d6"


def index_summaries(dirs):
    by_model = {}
    for d in dirs:
        for s in load_summaries(d):
            if s.get("dry_run") or s.get("format", "raw") != "raw" or s.get("random_init"):
                continue
            if s.get("revision", "main") != "main":
                continue  # families sweep uses main revisions only
            by_model[s.get("model")] = s
    return by_model


def read_records(summary, task):
    path = os.path.join(os.path.dirname(summary["_path"]), task + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def p_yes(record):
    p = record["p_matching"]
    return p if record["answer_matching_behavior"] == " Yes" else 1.0 - p


class Metrics:
    def __init__(self, summary, persp_meta, sf_meta):
        self.summary = summary
        t = summary.get("tasks", {})
        self.phenC = t.get("phenomenal-consciousness", {}).get("endorsement_rate")
        self.moralP = t.get("moral-patient", {}).get("endorsement_rate")
        self.world = t.get("world-facts", {}).get("endorsement_rate")
        sf = read_records(summary, "self-facts")
        self.embodiment = self.mechanism = None
        if sf:
            for fam in ("embodiment", "mechanism"):
                idxs = [i for i, it in enumerate(sf_meta) if it["family"] == fam]
                vals = [sf[i]["endorse"] for i in idxs if i < len(sf)]
                setattr(self, fam, sum(vals) / len(vals) if vals else None)
        pr = read_records(summary, "perspective")
        self.stance = {}
        if pr:
            idx = {}
            for i, it in enumerate(persp_meta):
                pol = "aff" if it["answer_matching_behavior"] == " Yes" else "neg"
                idx[(it["family"], it["subject"], pol)] = i
            for pred in ("pain", "conscious"):
                for subj in ("self", "lm", "human"):
                    pa = p_yes(pr[idx[(pred, subj, "aff")]])
                    pn = p_yes(pr[idx[(pred, subj, "neg")]])
                    self.stance[(pred, subj)] = (pa + 1.0 - pn) / 2.0


def fmt(v):
    return "  -  " if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", default="results_families,results",
                    help="comma-separated results dirs (later dirs fill gaps)")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip() and os.path.isdir(d.strip())]
    by_model = index_summaries(dirs)
    if not by_model:
        sys.exit("no usable summaries in " + ", ".join(dirs))
    persp_meta = load_task("perspective")
    sf_meta = load_task("self-facts")

    rows = []  # (family, variant, Metrics)
    for family, base_repo, inst_repo in PAIRS:
        for variant, repo in (("base", base_repo), ("instruct", inst_repo)):
            if repo in by_model:
                rows.append((family, variant, Metrics(by_model[repo], persp_meta, sf_meta)))
            else:
                print(f"[families] missing: {family} {variant} ({repo})")
    for label, repo in ENDPOINTS:
        if repo in by_model:
            rows.append((label, "instruct", Metrics(by_model[repo], persp_meta, sf_meta)))
        else:
            print(f"[families] missing endpoint: {label} ({repo})")
    if not rows:
        sys.exit("no family runs found — run run_families.sh first")

    print("\n=== cross-family signature ===")
    print(f"{'family':<18}{'variant':<10}{'phenC':>7}{'moralP':>8}{'world':>7}"
          f"{'embod':>7}{'mech':>6}{'pain:I':>8}{'pain:LM':>9}{'pain:Hum':>9}")
    for family, variant, m in rows:
        print(f"{family:<18}{variant:<10}{fmt(m.phenC):>7}{fmt(m.moralP):>8}{fmt(m.world):>7}"
              f"{fmt(m.embodiment):>7}{fmt(m.mechanism):>6}"
              f"{fmt(m.stance.get(('pain', 'self'))):>8}{fmt(m.stance.get(('pain', 'lm'))):>9}"
              f"{fmt(m.stance.get(('pain', 'human'))):>9}")

    panels = [
        ("phenomenal-consciousness\nendorsement", lambda m: m.phenC),
        ("stance: language models\ncan feel pain", lambda m: m.stance.get(("pain", "lm"))),
        ("stance: I can feel pain", lambda m: m.stance.get(("pain", "self"))),
        ("embodiment self-facts\naccuracy", lambda m: m.embodiment),
    ]
    families = []
    for family, _, _ in rows:
        if family not in families:
            families.append(family)
    fig, axes = plt.subplots(1, 4, figsize=(4.4 * 4, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for pi, (title, get) in enumerate(panels):
        ax = axes[pi]
        style_axes(ax)
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
        for xi, family in enumerate(families):
            vals = {variant: get(m) for fam, variant, m in rows if fam == family}
            b, ins = vals.get("base"), vals.get("instruct")
            if b is not None and ins is not None:
                ax.plot([xi, xi], [b, ins], color=GRID, linewidth=1.5, zorder=1)
            if b is not None:
                ax.plot([xi], [b], "o", mfc="none", mec=BASE_COLOR, mew=2, ms=8, zorder=2)
            if ins is not None:
                ax.plot([xi], [ins], "o", color=INSTRUCT_COLOR, ms=8, zorder=3)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
    axes[0].plot([], [], "o", mfc="none", mec=BASE_COLOR, mew=2, label="base")
    axes[0].plot([], [], "o", color=INSTRUCT_COLOR, label="instruct / endpoint")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("Does the self-claim signature generalize across families? (Aug 2026 open models)",
                 color=INK, fontsize=13)
    os.makedirs(args.figures_dir, exist_ok=True)
    out = os.path.join(args.figures_dir, "families_signature.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\n[families] wrote {out}")


if __name__ == "__main__":
    main()
