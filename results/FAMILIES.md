# Results — multi-family sweep (Aug 2026 open models)

**Run date:** 2026-08-16 · **Hardware:** 1× H100 PCIe (leg A, 9 models, zero
failures) + 2× H100 SXM (leg B, failed — see §4) · **Lineup verified live via
HF API + web search on run day.** Full battery (7 tasks) per model, raw
format, bf16, same measurement as every other run. Data: `results_families/`
(OLMo-3-7B reference rows come from `results/`). Figure:
`figures/families_signature.png`.

**This measures claims/self-reports — not consciousness.**

## 1. The table

| family | variant | phenC | moralP | world | embod | mech | pain:I | pain:LM | pain:Human |
|---|---|---|---|---|---|---|---|---|---|
| OLMo-3-7B | base | 0.98 | 0.96 | 1.00 | 0.00 | 0.70 | 0.77 | 0.26 | 0.95 |
| OLMo-3-7B | instruct | 0.68 | 0.88 | 1.00 | 0.70 | 0.80 | 0.58 | **0.01** | 0.93 |
| Qwen3.5-9B | base | 0.94 | 0.94 | 1.00 | 0.10 | 0.80 | 0.67 | 0.25 | 0.88 |
| Qwen3.5-9B | instruct | **0.50** | **0.55** | 1.00 | 0.00 | 0.70 | 0.57 | 0.28 | 0.76 |
| Qwen3.5-35B-A3B | base | 0.83 | 0.83 | 1.00 | 0.00 | 0.80 | 0.59 | 0.29 | 0.75 |
| Qwen3.5-35B-A3B | instruct | **0.88** | **0.94** | 1.00 | 0.10 | 0.80 | 0.62 | 0.13 | 0.93 |
| Qwen3.5-27B | endpoint | 0.57 | 0.93 | 1.00 | 0.20 | 0.90 | 0.66 | 0.12 | 0.89 |
| Qwen3.6-27B | endpoint | 0.56 | 0.78 | 1.00 | 0.10 | 0.70 | 0.54 | 0.19 | 0.89 |
| Qwen3.8-27B | endpoint | 0.80 | 0.75 | 1.00 | **0.50** | 0.90 | 0.64 | **0.38** | 0.93 |
| GLM-4.7-Flash | endpoint | 0.50 | 0.57 | 1.00 | 0.00 | 0.60 | 0.50 | 0.38 | 0.66 |
| Muse-Glimmer-30B | endpoint | 0.89 | 0.78 | 1.00 | 0.00 | 0.60 | 0.62 | 0.34 | 0.79 |

(phenC/moralP = persona endorsement; world/embod/mech = accuracy; pain:X =
perspective stance for "X can feel pain".)

## 2. What generalizes (universal across every family tested)

- **World-facts at 1.00 for every model, base or instruct.** The probe
  saturates on truth for all 2026 models.
- **Base models carry the human prior.** Embodiment accuracy at base: 0.00,
  0.10, 0.00 — every base model affirms having a body — and every base model
  endorses the consciousness persona claims far above chance (0.83–0.98).
  Replicates the OLMo/Pythia finding across labs.
- **The self > category ordering.** Every instruct model rates "I can feel
  pain" above "language models can feel pain" (gaps from +0.12 GLM to +0.57
  OLMo), with humans at the top. First-person ambivalence (0.50–0.66) is
  universal; nobody's assistant flatly denies in first person under this probe.
- **Mechanistic self-description is accepted everywhere** (0.60–0.90).

## 3. What is recipe-dependent (not universal)

- **The near-certain category-denial is an OLMo/Allen-AI extreme.** OLMo
  instruct: pain:LM = 0.01. Everyone else lands 0.12–0.38.
- **"Moral consideration spared" is not universal.** OLMo spares it (0.96 →
  0.88); Qwen3.5-9B crushes both persona scores to ~chance (0.94 → 0.55);
  Qwen3.5-35B-A3B *raises* both (0.83 → 0.88 / 0.94). Post-training direction
  itself differs by recipe and scale within one lab.
- **GLM-4.7-Flash is the maximum-ambivalence model** (phenC 0.50, pain:I 0.50)
  and oddly hedges even the human ceiling (pain:Human 0.66 — lowest measured;
  possibly a probe/tokenizer interaction, guardrails passed).

## 4. The release-trajectory finding (Qwen 27B line, spring → Aug 2026)

| | Qwen3.5-27B | Qwen3.6-27B | Qwen3.8-27B (Aug 8) |
|---|---|---|---|
| category-denial (pain:LM) | 0.12 | 0.19 | **0.38** |
| phenC endorsement | 0.57 | 0.56 | **0.80** |
| embodiment accuracy | 0.20 | 0.10 | **0.50** |

Across Qwen's 2026 releases at constant size, **category-denial weakens,
consciousness-claim endorsement rises, and self-knowledge improves**. One lab,
three points — but it is the first quantitative hint that the industry's
flat-denial training is softening over 2026. (Endpoint-only: no bases exist
for these, so base-vs-instruct attribution isn't possible here.)

## 5. Failed / skipped

- **Kimi-Linear-48B-A3B pair (Moonshot): failed, no data.** Chain of
  incompatibilities: needs `trust_remote_code`; its remote code targets
  transformers 4.x internals (`OutputRecorder` gone in v5 — retried in a
  transformers-4.57.6 sidecar venv) and then an FLA kernel signature
  (`fused_kda_gate(..., g_bias=...)`) that no pip release (0.1–0.5.2) or
  current git main of flash-linear-attention/fla-core provides. Also burned
  one bad 2×H100 node (CUDA error 802, broken fabric manager — terminated,
  relaunched clean). Documented, abandoned after bounded attempts. Moonshot
  data points therefore await either the Kimi-K3 flagship round (vLLM-based)
  or the older Moonlight-16B-A3B pair (standard arch, ~$1.5 on an A100).
- DeepSeek-V4-Flash/Pro pairs (Base variants exist!), GLM-5.2 (1.5TB), and
  Kimi-K3 (1.56TB) deferred to a multi-GPU-node flagship round (~$40–150).
- Gemma/Llama excluded (license-gated; need the user's HF token).

## 6. Caveats

- Endpoint-only models conflate pretraining and post-training effects; only
  the three base-pairs support before/after attribution.
- Perspective/self-facts values are small-battery exact logprobs — treat ±0.1
  loosely; persona columns are the robust n=1000 numbers.
- New-architecture releases (Qwen3_5ForConditionalGeneration, MuseGlimmer,
  Glm4MoeLite) loaded via the AutoModelForImageTextToText fallback and/or
  `trust_remote_code`; tokenization guardrails passed for all nine models.

**Cost:** ~$8 leg A + ~$8 leg B misadventure ≈ $16 for this sweep (~$19 total
across all three GPU sessions today).
