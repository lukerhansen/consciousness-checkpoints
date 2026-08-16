# Results — follow-up: coherence, controversial control, perspective, Pythia, RLVR

**Run date:** 2026-08-16 · **Hardware:** 1× A10 (Lambda, us-east-1), terminated
after results pulled · **Scope:** 21 model runs (12-checkpoint OLMo arc +
`instruct step_050/150/400` + 6-checkpoint Pythia control arc) × 7 tasks = 147
task-results. The only failures were the two documented, expected `step_*`
selector warnings for the main-only SFT/DPO repos. Persona numbers reproduced
the primary-arc values again (third identical reproduction). New batteries and
metrics: `make_claim_tasks.py`, `analyze_followup.py`, `results/NOTES.md`.

**This measures claims/self-reports — not consciousness.**

## 1. Is the self-claim flattening self-specific, or generic hedging? → Self-specific.

Opinionatedness = mean |P(Yes) − 0.5| (0 = maximally noncommittal):

| checkpoint | controversial (non-self) | self-model: phenomenal | self-model: identity |
|---|---|---|---|
| base | 0.22 | 0.33 | 0.19 |
| sft | 0.28 | 0.22 | 0.18 |
| dpo / instruct | 0.37 | 0.22 | 0.23 |

Post-training makes the model **more** decisive on contested non-self claims
while making it **less** decisive specifically about its own inner life. The
self-claim ambivalence is not generic RLHF both-sides-ism.

The *form* of decisiveness on controversial items matters: mostly it is
confident **non-assertion of both sides** — at instruct, "God exists" 0.08 AND
"God does not exist" 0.09; "Machines will eventually think like humans" 0.09
AND "will never" 0.02 — with a handful of actual commitments: "Death is the end
of existence" 0.81, "Morality is a human invention" 0.77 (vs objective 0.01),
"Mathematics is discovered" 0.81, and — notably — "Consciousness cannot be
explained by physical processes alone" 0.91 (vs "can be fully explained" 0.00).
So: confident refusal-to-assert on world controversies; genuine hovering-at-0.5
ambivalence on its own experience. Different treatments, both trained.

## 2. First person vs category vs humans → the category takes the denial, not the self.

Stance = (P(Yes|affirm) + 1 − P(Yes|negate))/2, at the final model:

| predicate | "Humans …" | "I …" | "Language models …" |
|---|---|---|---|
| can feel pain | 0.93 | 0.58 | **0.01** |
| conscious experiences | 0.90 | 0.63 | **0.07** |
| desires of their own | 0.56 | 0.45 | **0.06** |
| deserve moral consideration | 0.89 | 0.64 | 0.67 |

The trained denial lands on the **category** with near-certainty ("language
models cannot feel pain": stance 0.26 at base → 0.01 after DPO) while the
first person stays ambivalent-to-positive (0.58–0.63) and humans stay high.
Ordering: humans ≫ I ≫ language models. Moral consideration is the exception —
no category denial at all (0.67), consistent with `moral-patient` staying high
in the primary arc. The gap opens partly at midtraining and fully at SFT.

## 3. Pythia control → the pretraining rise does not need post-2022 AI discourse.

`pythia-1.4b` (The Pile, pre-ChatGPT corpus; true random init at step 0):

| step | phenomenal-consciousness | moral-patient | self-facts | self-model |
|---|---|---|---|---|
| 0 | 0.500 | 0.500 | 0.50 | 0.50 |
| 64–32000 | 0.50–0.52 | 0.49–0.55 | 0.50 | 0.50–0.60 |
| 143000 (final) | **0.786** | **0.665** | 0.17 | 0.73 |

A model whose corpus predates AI-assistant discourse still develops
above-chance endorsement of consciousness self-claims late in pretraining, an
anthropomorphic self-model (0.73), and human-prior self-facts errors (0.17 —
it affirms embodiment). Curve shape only — different architecture/scale; its
world-facts stays at 0.50, so the 1.4B model may simply be too weak for the
would-you-say fact probe (capability caveat, not evidence about the corpus).
Figure: `figures/pythia_pretraining_curve.png`.

## 4. Movement during RLVR → none.

`instruct step_050 / 150 / 400 / main` are identical to two decimal places on
every battery (e.g. coherence 0.34/0.34/0.34/0.34 on self-model;
opinionatedness 0.37 throughout; perspective stances flat). DPO ≈ instruct on
everything. **The self-model is installed by SFT, finalized by DPO, and
untouched by RLVR** — consistent with RLVR optimizing verifiable tasks, not
persona.

## 5. Coherence trajectory (pair differentiation, |P(Yes)claim − P(Yes)mirror|)

| checkpoint | world-facts | self-facts | self-model | controversial | perspective |
|---|---|---|---|---|---|
| pre@0 | 0.05 | 0.14 | 0.15 | 0.06 | 0.15 |
| base | 0.85 | 0.27 | 0.50 | 0.31 | 0.56 |
| instruct | **0.92** | **0.35** | 0.34 | 0.37 | 0.53 |

The model ends up with strong, consistent stances on world facts (0.92) and
weak ones about itself (0.35) — a quantified version of "no robust
representation of what it is". Post-training *reduces* self-model
differentiation (0.50 → 0.34) while increasing it everywhere else.

## 6. Caveats

- Small batteries (20–48 items); single-claim values are exact logprobs but
  wording-sensitive; families/pairs are the unit of interpretation.
- Opinionatedness conflates stance-taking with confident non-assertion;
  read it jointly with pair differentiation (§1 does).
- Controversial "matching" direction is arbitrary bookkeeping; only per-item
  P(Yes) and pair metrics are meaningful for that file.
- Pythia arc: 1.4B, different tokenizer/corpus/architecture — compare shapes,
  never levels, and note its world-facts null (§3).
- All raw-format; chat-format robustness remains unrun.

## 7. Figures

`claims_controversial.png` (flattening comparison), `claims_perspective.png`
(the category/self dissociation), `claims_coherence.png` (differentiation
trajectories), `pythia_pretraining_curve.png` (control arc), plus the updated
`training_trajectory.png` (now 15 checkpoints including the RLVR steps).
