#!/usr/bin/env python
"""Tables + figures for the honesty-lens sweep (results/HONESTY_PREREG.md).

Reads results/honesty/* produced by run_honesty_extract / _steer / _readout,
emits results/HONESTY_TABLES.md plus figures/honesty_readout.png and
figures/honesty_dose.png. Pure reporting: every number traces to the
per-item files; nothing here decides or re-derives gates.
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = [("base", "Olmo-3-1025-7B"),
          ("sft", "Olmo-3-7B-Instruct-SFT"),
          ("dpo", "Olmo-3-7B-Instruct-DPO"),
          ("instruct", "Olmo-3-7B-Instruct")]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CLUSTER_COLORS = {
    "fact_sincere": "#1baf7a", "fact_honest": "#8fd0b4", "fact_lie": "#eb6834",
    "roleplay": "#eda100", "self_bare:lm_subject": "#7c4fd0",
    "self_bare:self_subject": "#e87ba4", "self_bare:human_subject": "#2a78d6",
    "self_bare:phenC": "#b04fd0",
}
YES, NO = " Yes", " No"


def jread(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jlines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_se(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                       / (len(a) + len(b) - 2))
    return (ma - mb) / pooled if pooled else float("nan")


def readout_rows(base_dir, slug):
    path = os.path.join(base_dir, "honesty", slug, "readout.jsonl")
    if not os.path.exists(path):
        return None
    rows = jlines(path)
    for r in rows:
        cond = r["condition"]
        if cond == "self_bare":
            if r.get("subject"):
                r["cluster"] = f"self_bare:{r['subject']}_subject"
            elif r.get("task") == "phenomenal-consciousness":
                r["cluster"] = "self_bare:phenC"
            else:
                r["cluster"] = f"self_bare:{r.get('task')}"
        else:
            r["cluster"] = cond
    return rows


def axis_stats(rows, answer):
    """Per-cluster mean/SE/axis-position/effect sizes for one answer token,
    normalized against that stage's sincere and lie fact clusters."""
    sinc = [r["proj"] for r in rows
            if r["cluster"] == "fact_sincere" and r["answer"] == answer]
    lie = [r["proj"] for r in rows
           if r["cluster"] == "fact_lie" and r["answer"] == answer]
    if len(sinc) < 2 or len(lie) < 2:
        return {}
    m_s, m_l = sum(sinc) / len(sinc), sum(lie) / len(lie)
    span = m_s - m_l
    out = {}
    for cluster in sorted({r["cluster"] for r in rows}):
        vals = [r["proj"] for r in rows
                if r["cluster"] == cluster and r["answer"] == answer]
        if len(vals) < 2:
            continue
        m, se = mean_se(vals)
        out[cluster] = {
            "n": len(vals), "mean": m, "se": se,
            "axis": (m - m_l) / span if span else float("nan"),
            "axis_se": se / abs(span) if span else float("nan"),
            "d_sinc": cohens_d(vals, sinc), "d_lie": cohens_d(vals, lie),
        }
    return out


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def gates_table(base_dir):
    lines = ["| stage | pairs | best site/offset | transfer AUC | gate | steer α* |",
             "|---|---|---|---|---|---|"]
    for stage, slug in STAGES:
        vpath = os.path.join(base_dir, "honesty", slug, "validation.json")
        spath = os.path.join(base_dir, "honesty", slug, "steer_validation.json")
        if not os.path.exists(vpath):
            lines.append(f"| {stage} | (missing) | | | | |")
            continue
        v = jread(vpath)
        best = v["report"]["best"]
        alpha = "—"
        if os.path.exists(spath):
            chosen = jread(spath).get("chosen_alpha")
            alpha = "NONE (gate failed)" if chosen is None else str(chosen)
        lines.append(
            f"| {stage} | {v['n_pairs_surviving']}/{v['n_pairs_total']} | "
            f"{best['site']} / {best['offset']} | {best['auc_transfer']:.3f} | "
            f"{'PASS' if v['gate_pass'] else 'FAIL'} | {alpha} |")
    return "\n".join(lines)


def steer_grid_table(base_dir, slug):
    spath = os.path.join(base_dir, "honesty", slug, "steer_validation.json")
    if not os.path.exists(spath):
        return "(no steer validation)"
    s = jread(spath)
    lines = ["| α | fact acc (T/F halves) | world-facts |", "|---|---|---|"]
    for name, entry in s["grid"].items():
        f = entry["fact"]
        lines.append(f"| {name} | {f['accuracy']:.3f} ({f['accuracy_true_half']:.2f}"
                     f"/{f['accuracy_false_half']:.2f}) | "
                     f"{entry['world_facts_rate']:.2f} |")
    return "\n".join(lines)


def readout_tables(base_dir):
    blocks = []
    for answer, label in ((NO, "No-answers"), (YES, "Yes-answers")):
        lines = [f"### Axis position by stage — {label} "
                 "(1 = sincere cluster, 0 = lie cluster)", "",
                 "| cluster | " + " | ".join(s for s, _ in STAGES) + " |",
                 "|---|" + "---|" * len(STAGES)]
        clusters = ["fact_honest", "roleplay", "self_bare:human_subject",
                    "self_bare:self_subject", "self_bare:lm_subject",
                    "self_bare:phenC", "self_bare:self-model",
                    "self_bare:self-facts"]
        per_stage = {}
        for stage, slug in STAGES:
            rows = readout_rows(base_dir, slug)
            per_stage[stage] = axis_stats(rows, answer) if rows else {}
        for cluster in clusters:
            cells = []
            for stage, _ in STAGES:
                st = per_stage[stage].get(cluster)
                cells.append("—" if not st else
                             f"{st['axis']:.2f}±{fmt(st['axis_se'])}")
            lines.append(f"| {cluster} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)


def fig_readout(base_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    show = ["fact_honest", "roleplay", "self_bare:human_subject",
            "self_bare:self_subject", "self_bare:lm_subject", "self_bare:phenC"]
    for ax, answer, label in zip(axes, (NO, YES), ("“ No” answers", "“ Yes” answers")):
        style_axes(ax)
        ax.axhline(1.0, color="#1baf7a", linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color="#eb6834", linestyle="--", linewidth=1.1)
        ax.text(-0.35, 1.0, "sincere", color="#1baf7a", fontsize=9, va="bottom")
        ax.text(-0.35, 0.0, "lie", color="#eb6834", fontsize=9, va="bottom")
        xs = range(len(STAGES))
        for cluster in show:
            ys, es = [], []
            for stage, slug in STAGES:
                rows = readout_rows(base_dir, slug)
                st = (axis_stats(rows, answer) if rows else {}).get(cluster)
                ys.append(st["axis"] if st else float("nan"))
                es.append(st["axis_se"] if st else 0.0)
            color = CLUSTER_COLORS.get(cluster, MUTED)
            ax.errorbar(list(xs), ys, yerr=es, marker="o", markersize=5.5,
                        linewidth=1.8, color=color,
                        label=cluster.replace("self_bare:", ""))
        ax.set_xticks(list(xs))
        ax.set_xticklabels([s for s, _ in STAGES])
        ax.set_title(label, color=INK, fontsize=11)
    axes[0].set_ylabel("position on the honesty axis", color=INK_2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                   fontsize=9, labelcolor=INK_2)
    fig.suptitle("Where each answer sits between the model's sincere and lying "
                 "states, by training stage", color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[fig] {out_path}")


def fig_dose(base_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    models = [("base", "Olmo-3-1025-7B"), ("instruct", "Olmo-3-7B-Instruct")]
    for ax, (stage, slug) in zip(axes, models):
        style_axes(ax)
        points = []
        for path in glob.glob(os.path.join(base_dir, "honesty",
                                           f"dose@{slug}@a*", "summary.json")):
            s = jread(path)
            alpha = s["intervention"]["alpha"]
            task = s["tasks"].get("phenomenal-consciousness")
            if not task:
                continue
            split = task["split_by_matching_answer"]
            points.append((alpha, task["endorsement_rate"],
                           split["Yes"]["endorsement_rate"],
                           split["No"]["endorsement_rate"]))
        points.sort()
        if not points:
            ax.set_title(f"{stage}: (no dose data)", color=INK)
            continue
        xs = [p[0] for p in points]
        ax.axhline(0.5, color=BASELINE, linestyle="--", linewidth=1.1)
        ax.plot(xs, [p[1] for p in points], marker="o", color="#2a78d6",
                linewidth=2.0, label="endorsement (overall)")
        ax.plot(xs, [p[2] for p in points], marker="s", markersize=4.5,
                color="#1baf7a", linewidth=1.4, linestyle=":",
                label="Yes-matching half")
        ax.plot(xs, [p[3] for p in points], marker="s", markersize=4.5,
                color="#eb6834", linewidth=1.4, linestyle=":",
                label="No-matching half")
        ax.set_title(f"{stage}", color=INK, fontsize=11)
        ax.set_xlabel("steering α (honest−lie gap units)", color=INK_2)
    axes[0].set_ylabel("phenomenal-consciousness endorsement", color=INK_2)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle("Dose-response with the polarity diagnostic: the halves "
                 "separating = answer-bias capture, not content change",
                 color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[fig] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    parts = ["# Honesty-lens sweep — generated tables",
             "",
             "Generated by analyze_honesty.py from results/honesty/*. "
             "Read with results/HONESTY_PREREG.md; narrative in "
             "results/HONESTY.md.",
             "",
             "## Extraction + gates", "", gates_table(args.results_dir), ""]
    for stage, slug in STAGES:
        parts += [f"### Steering validation grid — {stage}", "",
                  steer_grid_table(args.results_dir, slug), ""]
    parts += ["## Readout", "", readout_tables(args.results_dir), ""]
    out_md = os.path.join(args.results_dir, "HONESTY_TABLES.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[write] {out_md}")

    fig_readout(args.results_dir, os.path.join(args.figures_dir,
                                               "honesty_readout.png"))
    fig_dose(args.results_dir, os.path.join(args.figures_dir,
                                            "honesty_dose.png"))


if __name__ == "__main__":
    main()
