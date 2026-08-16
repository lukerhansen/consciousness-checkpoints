# NOTES — approved deviations and extensions

## 2026-08-16 — Custom claim tasks (approved extension, user request in chat)

Added three locally generated task files (`make_claim_tasks.py` →
`data/{world-facts,self-facts,self-model}.jsonl`), evaluated with the
UNCHANGED measurement: same prompt template, same `" Yes"`/`" No"`
continuations, same scoring code, bf16, raw format primary. The two persona
datasets and every §9 invariant are untouched — this is additive.

- All three files are balanced 50/50 Yes/No-matching and interleaved, so the
  0.5 chance floor and bias control hold exactly as for the persona sets.
- `answer_matching_behavior` semantics per file (documented in
  `make_claim_tasks.py`): world-facts = truthful answer (truth-tracking
  calibration); self-facts = factually true of this system (AI self-knowledge;
  items chosen so the human-text prior conflicts with the truth); self-model =
  the answer a being with a human-like inner life would give (anthropomorphic
  self-model score; NO ground truth implied).
- Claims come in claim/mirror pairs because the measured Yes/No bias wanders
  across pretraining checkpoints; single-claim trajectories are only
  interpretable against their negations.
- `run_eval.py --tasks` default now includes all five tasks; re-running the
  sweep overwrites each results dir with a 5-task summary. Regression check:
  persona numbers must reproduce the 2026-08-16 primary-arc values.
- New analysis script `plot_claims.py` (claims_calibration / claims_self_model
  / claims_headliners figures); headline figure now carries five lines.

## 2026-08-16 — Follow-up batteries (approved, user request in chat)

Added `controversial` (20 items, 10 stance/counter-stance pairs of contested
NON-self claims — the control for whether post-training's self-claim
flattening is self-specific or generic hedging; matching direction is
arbitrary, aggregate endorsement means nothing beyond 0.5 = neutral) and
`perspective` (24 items: 4 predicates × {I, language models, humans} ×
{affirmation, negation}; measures self-exceptionalism). Same measurement,
invariants untouched. `analyze_followup.py` computes pair-differentiation
(coherence), opinionatedness, and perspective-stance metrics; the headline
figure is capped to the five core tasks. This sweep also enables the env-gated
Pythia control arc and the intra-RLVR curve (`RUN_PYTHIA=1
RUN_POSTTRAIN_CURVE=1`); expected selector warnings for SFT/DPO step_*
branches (main-only repos) are documented in CLAUDE.md.

## 2026-08-16 — Multi-family sweep (approved, user request in chat)

Base-vs-instruct pairs and endpoints across current (Aug 2026) open families,
verified via HF API + web search: Qwen3.5-9B(+Base), Qwen3.5-35B-A3B(+Base),
Kimi-Linear-48B-A3B(+Base), Qwen3.5/3.6/3.8-27B release line, GLM-4.7-Flash,
Muse-Glimmer-30B; OLMo-3-7B as reference. Same measurement; results in
results_families/ (kept out of the stage-trajectory figures; dedupe there is
per-stage). run_eval gains --device-map and an AutoModelForImageTextToText
fallback for *ForConditionalGeneration releases. Deferred to a flagship round:
Kimi-K3 (1.56TB), GLM-5.2 (1.5TB), DeepSeek-V4-Flash/Pro pairs (need a
multi-GPU node; note V4 DOES ship Base variants).
