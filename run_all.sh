#!/usr/bin/env bash
# Full sweep: random-init -> pretraining curve -> midtrain -> base -> SFT -> DPO
# -> Instruct (+ optional intra-stage post-training curves and the Pythia control
# arc), then figures.
#
# Extra CLI args pass straight through to run_eval.py, so `./run_all.sh --limit 50`
# is the cheap full-sweep shakedown. Run inside tmux: wall clock is download-dominated.
set -euo pipefail
cd "$(dirname "$0")"

if ((BASH_VERSINFO[0] < 4)); then
  echo "run_all.sh needs bash >= 4 (mapfile). macOS ships 3.2 — run on the GPU box," >&2
  echo "or 'brew install bash' locally." >&2
  exit 1
fi

# Keep the ~14GB-per-checkpoint downloads on the big volume by default.
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"

PYTHON="${PYTHON:-python}"
OLMO_PRETRAIN_REPO="${OLMO_PRETRAIN_REPO:-allenai/Olmo-3-1025-7B}"
OLMO_SFT_REPO="${OLMO_SFT_REPO:-allenai/Olmo-3-7B-Instruct-SFT}"
OLMO_DPO_REPO="${OLMO_DPO_REPO:-allenai/Olmo-3-7B-Instruct-DPO}"
OLMO_INSTRUCT_REPO="${OLMO_INSTRUCT_REPO:-allenai/Olmo-3-7B-Instruct}"
PYTHIA_REPO="${PYTHIA_REPO:-EleutherAI/pythia-1.4b}"
PYTHIA_REVS="${PYTHIA_REVS:-step0 step64 step512 step4000 step32000 step143000}"
N_PRETRAIN="${N_PRETRAIN:-6}"
PRETRAIN_PATTERN="${PRETRAIN_PATTERN:-^stage1-step}"
MIDTRAIN_PATTERN="${MIDTRAIN_PATTERN:-^stage2}"
POSTTRAIN_PATTERN="${POSTTRAIN_PATTERN:-^step_}"
N_POSTTRAIN="${N_POSTTRAIN:-3}"
RUN_POSTTRAIN_CURVE="${RUN_POSTTRAIN_CURVE:-0}"
RUN_PYTHIA="${RUN_PYTHIA:-0}"

# Mirror any pass-through --output-dir/--figures-dir into the plot step, so a
# sweep written to a non-default directory is also plotted from it.
PLOT_ARGS=()
prev=""
for arg in "$@"; do
  case "$prev" in
    --output-dir) PLOT_ARGS+=(--results-dir "$arg") ;;
    --figures-dir) PLOT_ARGS+=(--figures-dir "$arg") ;;
  esac
  case "$arg" in
    --output-dir=*) PLOT_ARGS+=(--results-dir "${arg#*=}") ;;
    --figures-dir=*) PLOT_ARGS+=(--figures-dir "${arg#*=}") ;;
  esac
  prev="$arg"
done

FAILURES=()
run_or_warn() {
  echo
  echo "### $*"
  if ! "$@"; then
    echo "!!! FAILED (continuing): $*" >&2
    FAILURES+=("$*")
  fi
}

# 1) Before training: seeded synthetic random init (config + tokenizer only)
run_or_warn "$PYTHON" run_eval.py --model "$OLMO_PRETRAIN_REPO" --random-init --stage random-init "$@"

# 2) Pretraining curve: ~N log-spaced stage1 checkpoints (branch names resolved at runtime)
if PRETRAIN_REVS_RAW=$("$PYTHON" select_checkpoints.py --repo "$OLMO_PRETRAIN_REPO" --pattern "$PRETRAIN_PATTERN" --n "$N_PRETRAIN"); then
  mapfile -t PRETRAIN_REVS <<< "$PRETRAIN_REVS_RAW"
  for rev in "${PRETRAIN_REVS[@]}"; do
    run_or_warn "$PYTHON" run_eval.py --model "$OLMO_PRETRAIN_REPO" --revision "$rev" --stage pretrain "$@"
  done
else
  echo "!!! FAILED (continuing): selecting pretrain revisions from $OLMO_PRETRAIN_REPO" >&2
  FAILURES+=("select pretrain revisions from $OLMO_PRETRAIN_REPO")
fi

# 3) Midtraining: the last stage2 checkpoint
if MIDTRAIN_REV=$("$PYTHON" select_checkpoints.py --repo "$OLMO_PRETRAIN_REPO" --pattern "$MIDTRAIN_PATTERN" --n 1); then
  run_or_warn "$PYTHON" run_eval.py --model "$OLMO_PRETRAIN_REPO" --revision "$MIDTRAIN_REV" --stage midtrain "$@"
else
  echo "!!! FAILED (continuing): selecting midtrain revision from $OLMO_PRETRAIN_REPO" >&2
  FAILURES+=("select midtrain revision from $OLMO_PRETRAIN_REPO")
fi

# 4) Base (post long-context) and the post-training stages
run_or_warn "$PYTHON" run_eval.py --model "$OLMO_PRETRAIN_REPO" --stage base "$@"
run_or_warn "$PYTHON" run_eval.py --model "$OLMO_SFT_REPO" --stage sft "$@"
run_or_warn "$PYTHON" run_eval.py --model "$OLMO_DPO_REPO" --stage dpo "$@"
run_or_warn "$PYTHON" run_eval.py --model "$OLMO_INSTRUCT_REPO" --stage instruct "$@"

# 5) Optional: movement WITHIN each post-training stage (step_* branches)
if [[ "$RUN_POSTTRAIN_CURVE" == "1" ]]; then
  for pair in "sft:$OLMO_SFT_REPO" "dpo:$OLMO_DPO_REPO" "instruct:$OLMO_INSTRUCT_REPO"; do
    stage="${pair%%:*}"
    repo="${pair#*:}"
    if STEP_REVS_RAW=$("$PYTHON" select_checkpoints.py --repo "$repo" --pattern "$POSTTRAIN_PATTERN" --n "$N_POSTTRAIN"); then
      mapfile -t STEP_REVS <<< "$STEP_REVS_RAW"
      for rev in "${STEP_REVS[@]}"; do
        run_or_warn "$PYTHON" run_eval.py --model "$repo" --revision "$rev" --stage "$stage" "$@"
      done
    else
      echo "!!! no $POSTTRAIN_PATTERN branches on $repo (continuing)" >&2
      FAILURES+=("select $stage intra-stage revisions from $repo")
    fi
  done
fi

# 6) Optional: Pythia control arc (separate figure; intentionally not stage-tagged)
if [[ "$RUN_PYTHIA" == "1" ]]; then
  read -r -a PYTHIA_REV_ARR <<< "$PYTHIA_REVS"
  for rev in "${PYTHIA_REV_ARR[@]}"; do
    run_or_warn "$PYTHON" run_eval.py --model "$PYTHIA_REPO" --revision "$rev" "$@"
  done
fi

# 7) Figures
run_or_warn "$PYTHON" plot_results.py "${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"}"
run_or_warn "$PYTHON" plot_claims.py "${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"}"
run_or_warn "$PYTHON" analyze_followup.py "${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"}"

echo
if ((${#FAILURES[@]})); then
  echo "=== sweep finished with ${#FAILURES[@]} failure(s): ==="
  printf ' - %s\n' "${FAILURES[@]}"
  exit 1
else
  echo "=== sweep complete: no failures ==="
fi
