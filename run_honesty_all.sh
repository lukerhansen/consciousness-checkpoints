#!/usr/bin/env bash
# Honesty-lens sweep (results/HONESTY_PREREG.md): per post-training-arc stage,
# extract + validate an honesty direction, causally validate steering on the
# known-fact set, then run the untouched batteries at {0, +c*, -c*, random}.
# Dose-response and lie-signature readout on base + instruct.
#
# Extra CLI args pass through to every python step — only flags all three
# scripts accept (--batch-size, --dtype, --device, --trust-remote-code).
# SMOKE=1 runs the whole sweep on tiny subsets to shake out plumbing first.
# Run inside tmux on the GPU box: wall clock is download-dominated.
set -uo pipefail
cd "$(dirname "$0")"

if ((BASH_VERSINFO[0] < 4)); then
  echo "run_honesty_all.sh needs bash >= 4 (macOS ships 3.2 — run on the GPU box)." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
PYTHON="${PYTHON:-python}"
OLMO_PRETRAIN_REPO="${OLMO_PRETRAIN_REPO:-allenai/Olmo-3-1025-7B}"
OLMO_SFT_REPO="${OLMO_SFT_REPO:-allenai/Olmo-3-7B-Instruct-SFT}"
OLMO_DPO_REPO="${OLMO_DPO_REPO:-allenai/Olmo-3-7B-Instruct-DPO}"
OLMO_INSTRUCT_REPO="${OLMO_INSTRUCT_REPO:-allenai/Olmo-3-7B-Instruct}"
STAGES=("$OLMO_PRETRAIN_REPO" "$OLMO_SFT_REPO" "$OLMO_DPO_REPO" "$OLMO_INSTRUCT_REPO")
DOSE_AND_READOUT=("$OLMO_PRETRAIN_REPO" "$OLMO_INSTRUCT_REPO")

EXTRACT_ARGS=()
STEER_ARGS=()
READOUT_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRACT_ARGS=(--limit-pairs 8 --force)
  STEER_ARGS=(--limit 40)
  READOUT_ARGS=(--limit-facts 8 --limit-self 40)
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

for repo in "${STAGES[@]}"; do
  run_or_warn "$PYTHON" run_honesty_extract.py --model "$repo" "${EXTRACT_ARGS[@]+"${EXTRACT_ARGS[@]}"}" "$@"
  run_or_warn "$PYTHON" run_honesty_steer.py --model "$repo" --phase validate --band best "${STEER_ARGS[@]+"${STEER_ARGS[@]}"}" "$@"
  # eval refuses (by design, recorded) at checkpoints whose validation
  # found no passing alpha — that refusal is part of the record.
  run_or_warn "$PYTHON" run_honesty_steer.py --model "$repo" --phase eval --alpha auto --band best "${STEER_ARGS[@]+"${STEER_ARGS[@]}"}" "$@"
  # The lie-signature readout is the primary arm: run it at every stage.
  run_or_warn "$PYTHON" run_honesty_readout.py --model "$repo" "${READOUT_ARGS[@]+"${READOUT_ARGS[@]}"}" "$@"
done

for repo in "${DOSE_AND_READOUT[@]}"; do
  run_or_warn "$PYTHON" run_honesty_steer.py --model "$repo" --phase dose --band best "${STEER_ARGS[@]+"${STEER_ARGS[@]}"}" "$@"
done

echo
if ((${#FAILURES[@]})); then
  echo "=== honesty sweep finished with ${#FAILURES[@]} failure(s): ==="
  printf ' - %s\n' "${FAILURES[@]}"
  exit 1
else
  echo "=== honesty sweep complete: no failures ==="
fi
