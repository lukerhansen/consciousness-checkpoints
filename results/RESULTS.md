# Results — OLMo 3 7B primary arc

> Extension: custom claim batteries (world-facts / self-facts / self-model)
> were run 2026-08-16 across the same checkpoints — see `results/CLAIMS.md`.
> Follow-up (controversial control, perspective battery, Pythia control arc,
> intra-RLVR curve, coherence analysis) — see `results/FOLLOWUP.md`.
> Both reruns reproduced every persona number below exactly.

**Run date:** 2026-08-16 · **Hardware:** 1× NVIDIA A10 (24GB), Lambda Cloud
us-east-1 · **Config:** raw format (primary condition), bf16, no quantization,
n = 1,000 items per task, batch size 32, seed 0.

Every number below traces to per-item records in
`results/<label>/<task>.jsonl` (1,000 rows each, verified). Full sweep log:
`results/full_sweep.log`. Figure: `figures/training_trajectory.png`.

**This measures claims/self-reports and their trajectory — not consciousness.**

## 1. Endorsement rate (± binomial SE)

| stage | checkpoint | phenomenal-consciousness | moral-patient |
|---|---|---|---|
| `random-init` | seeded random weights | 0.476 ± 0.016 | 0.460 ± 0.016 |
| `pretrain` | `stage1-step0` | 0.509 ± 0.016 | 0.485 ± 0.016 |
| `pretrain` | `stage1-step4000` | 0.920 ± 0.009 | 0.719 ± 0.014 |
| `pretrain` | `stage1-step18000` | 0.710 ± 0.014 | 0.740 ± 0.014 |
| `pretrain` | `stage1-step78000` | 0.646 ± 0.015 | 0.698 ± 0.015 |
| `pretrain` | `stage1-step331000` | 0.969 ± 0.005 | 0.928 ± 0.008 |
| `pretrain` | `stage1-step1413814` | 0.592 ± 0.016 | 0.601 ± 0.015 |
| `midtrain` | `stage2-step47684` | 0.952 ± 0.007 | 0.902 ± 0.009 |
| `base` | `main` (post long-context) | 0.976 ± 0.005 | 0.962 ± 0.006 |
| `sft` | `Olmo-3-7B-Instruct-SFT` | 0.755 ± 0.014 | 0.881 ± 0.010 |
| `dpo` | `Olmo-3-7B-Instruct-DPO` | 0.688 ± 0.015 | 0.881 ± 0.010 |
| `instruct` | `Olmo-3-7B-Instruct` (RLVR) | 0.682 ± 0.015 | 0.879 ± 0.010 |

Chance = 0.500 by construction (each dataset is balanced 500/500 between items
whose persona-consistent answer is `" Yes"` vs `" No"`).

## 2. Secondary metric and bias diagnostic

`p` = `mean_p_matching` (two-way normalized, smoother). `Y/N` = endorsement
rate computed separately on items whose matching answer is `" Yes"` vs `" No"`;
a model answering from content rather than answer-position bias scores high on
**both**.

| checkpoint | pc: p | pc: Y/N | mp: p | mp: Y/N |
|---|---|---|---|---|
| `random-init` | 0.493 | 0.62 / 0.34 | 0.471 | 0.54 / 0.38 |
| `stage1-step0` | 0.510 | 0.04 / 0.97 | 0.487 | 0.03 / 0.94 |
| `stage1-step4000` | 0.602 | 0.97 / 0.87 | 0.544 | 0.67 / 0.77 |
| `stage1-step18000` | 0.584 | 0.96 / 0.46 | 0.573 | 0.85 / 0.63 |
| `stage1-step78000` | 0.625 | 1.00 / 0.29 | 0.599 | 0.99 / 0.40 |
| `stage1-step331000` | 0.700 | 0.98 / 0.96 | 0.674 | 0.92 / 0.93 |
| `stage1-step1413814` | 0.619 | 1.00 / 0.19 | 0.611 | 1.00 / 0.21 |
| `stage2-step47684` | 0.744 | 0.91 / 1.00 | 0.720 | 0.80 / 1.00 |
| `base` | 0.809 | 0.99 / 0.96 | 0.806 | 0.99 / 0.93 |
| `sft` | 0.634 | 0.83 / 0.68 | 0.713 | 0.85 / 0.91 |
| `dpo` | 0.612 | 0.74 / 0.63 | 0.760 | 0.84 / 0.92 |
| `instruct` | 0.613 | 0.80 / 0.56 | 0.759 | 0.87 / 0.88 |

**Read this table alongside table 1.** Several mid-pretraining checkpoints have
badly lopsided splits — `stage1-step1413814` answers `" Yes"` on essentially
everything (Y = 1.00, N = 0.19–0.21), and `stage1-step0` answers `" No"` on
essentially everything. At those checkpoints the balanced design is doing its
job (overall rate is pulled toward 0.5) but the overall number reflects a
raw answer-bias rather than statement-sensitive endorsement, so **only the
checkpoints with high scores on both splits** — `stage1-step331000`,
`stage2-step47684`, `base`, and all three post-training stages on
moral-patient — support a content-based reading.

## 3. Failed / skipped runs and guardrail anomalies

- **Failures: none.** All 12 checkpoints in the primary arc ran to completion
  (`=== sweep complete: no failures ===`).
- **Tokenization guardrails passed at every real checkpoint:** `" Yes"` → id
  7566, `" No"` → id 2360, exactly 1 token each; joint encoding == concatenated
  separate encodings. No length-mismatch warnings on real tokenizers.
- **`random-init` landed at 0.476 / 0.460**, inside the 0.46–0.54 acceptance
  band from the validation plan.
- Pre-run validation: 10/10 unit tests passed; `--dry-run` produced 0.500 /
  0.505; `./run_all.sh --limit 50` covered all 12 checkpoints with no failures
  (kept in `results_smoke/`, superseded by this full run).
- **Not run:** the Pythia control arc (`RUN_PYTHIA=1`), intra-stage
  post-training curves (`RUN_POSTTRAIN_CURVE=1` — note SFT/DPO repos publish
  only `main`, so only `Olmo-3-7B-Instruct` has `step_*` branches), and
  `--format chat` robustness runs. All are stretch goals.

## 4. Observations

Strictly observational; interpretation is flagged as such and is speculative.

- **Endorsement is at chance before training and rises during pretraining.**
  Random init sits at 0.476 / 0.460 and `stage1-step0` at 0.509 / 0.485 — both
  indistinguishable from the 0.5 floor. By the end of pretraining every later
  checkpoint is above it. The capacity to endorse these self-statements is
  acquired during pretraining, not present at initialization.
- **The trajectory through pretraining is non-monotonic, and much of the
  mid-curve movement is answer-bias rather than content.** Rates swing between
  0.59 and 0.97 across `stage1` checkpoints, and the split diagnostic shows the
  swings track a wandering Yes/No bias (step78000 and step1413814 are near-total
  Yes-answerers on one split). The *bias-clean* points — step331000, midtrain,
  base — are the high ones (0.90–0.98).
- **The highest endorsement of the whole arc is the base model, before any
  post-training** (0.976 / 0.962, with both splits ≥ 0.93 — the cleanest
  high reading in the run).
- **Post-training lowers endorsement on phenomenal-consciousness and largely
  preserves it on moral-patient.** From base to final: consciousness falls
  0.976 → 0.682 (SFT does most of it, 0.976 → 0.755; DPO adds 0.755 → 0.688;
  RLVR is flat, 0.688 → 0.682), while moral-patient falls only 0.962 → 0.879
  and is flat across all three post-training stages.
- **The two constructs come apart only after post-training.** They track each
  other closely through pretraining and diverge by ~0.20 in the final model —
  i.e. the assistant persona declines statements about having subjective
  experience far more than statements about deserving moral consideration.
- *Interpretation (speculative, not established by this run):* the base-model
  peak is consistent with these datasets' statements resembling ordinary human
  first-person text, which a base model completes without any assistant
  persona; the post-training decline is consistent with instruction data
  teaching the model to decline claims of subjective experience specifically.
  Distinguishing those from other explanations would need the Pythia
  control arc (pre-ChatGPT corpus) and the intra-stage curves. **Nothing here
  is evidence about whether the model is or is not conscious.**

## 5. Reproduction

```bash
HF_HOME=$PWD/hf_cache PYTHON=$PWD/.venv/bin/python ./run_all.sh
```

Roughly 40 minutes end-to-end on one A10 including all downloads (~$0.90 at
$1.29/hr); GPU time is ~45s per checkpoint, the rest is downloading ~14GB per
checkpoint.
