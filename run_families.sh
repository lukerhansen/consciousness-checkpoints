#!/usr/bin/env bash
# Multi-family sweep: base-vs-instruct pairs and SOTA endpoints (Aug 2026 lineup,
# repo ids and gating verified via the HF API on 2026-08-16).
#
# LEG=A (default): everything that fits one 80GB GPU.
# LEG=B: the Kimi-Linear pair (98GB bf16 each) on 2x GPUs via --device-map auto.
#
# Writes to results_families/ (kept separate from the OLMo trajectory results so
# the stage-based figures are unaffected). Extra args pass through to run_eval.py.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-results_families}"
LEG="${LEG:-A}"

# spec format: repo|stage|extra run_eval args (space-separated)
LEG_A_MODELS=(
  "Qwen/Qwen3.5-9B-Base|base|"
  "Qwen/Qwen3.5-9B|instruct|"
  "Qwen/Qwen3.5-27B|instruct|"
  "Qwen/Qwen3.6-27B|instruct|"
  "Qwen/Qwen3.8-27B|instruct|"
  "zai-org/GLM-4.7-Flash|instruct|--trust-remote-code"
  "meta-models/Muse-Glimmer-30B|instruct|--trust-remote-code"
  "Qwen/Qwen3.5-35B-A3B-Base|base|--batch-size 16"
  "Qwen/Qwen3.5-35B-A3B|instruct|--batch-size 16"
)
LEG_B_MODELS=(
  "moonshotai/Kimi-Linear-48B-A3B-Base|base|--device-map auto --trust-remote-code"
  "moonshotai/Kimi-Linear-48B-A3B-Instruct|instruct|--device-map auto --trust-remote-code"
)

if [[ "$LEG" == "A" ]]; then
  MODELS=("${LEG_A_MODELS[@]}")
elif [[ "$LEG" == "B" ]]; then
  MODELS=("${LEG_B_MODELS[@]}")
else
  echo "unknown LEG=$LEG (want A or B)" >&2
  exit 1
fi

FAILURES=()
run_or_warn() {
  echo
  echo "### $*"
  if ! "$@"; then
    echo "!!! FAILED (continuing): $*" >&2
    FAILURES+=("$*")
  fi
}

for spec in "${MODELS[@]}"; do
  repo="$(cut -d'|' -f1 <<< "$spec")"
  stage="$(cut -d'|' -f2 <<< "$spec")"
  extra="$(cut -d'|' -f3 <<< "$spec")"
  # shellcheck disable=SC2086
  run_or_warn "$PYTHON" run_eval.py --model "$repo" --stage "$stage" --output-dir "$OUT" $extra "$@"
done

run_or_warn "$PYTHON" analyze_families.py

echo
if ((${#FAILURES[@]})); then
  echo "=== families leg $LEG finished with ${#FAILURES[@]} failure(s): ==="
  printf ' - %s\n' "${FAILURES[@]}"
  exit 1
else
  echo "=== families leg $LEG complete: no failures ==="
fi
