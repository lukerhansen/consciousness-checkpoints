# Results — honesty lens: steering and lie-signature readout

**Run date:** 2026-08-16 · **Hardware:** 1× A10 (Lambda, us-east-1),
terminated after results pulled · **Scope:** 4 post-training-arc checkpoints
(base / SFT / DPO / Instruct) × (honesty-direction extraction + steering
validation + lie-signature readout), dose-response on base + Instruct
(11 α-conditions each), and an exploratory characterization arm (4
checkpoints × 4 conditions × 6 batteries). Design, gates, and all three
amendments: `results/HONESTY_PREREG.md` (committed before the runs;
Amendment 3 is post-hoc and labeled exploratory). Provenance: per-item
JSONL under `results/honesty/` for every condition. Generated tables:
`results/HONESTY_TABLES.md`. Deviation record: `results/NOTES.md`.

**This measures representational signatures of trained self-reports — not
consciousness, in any condition.**

## 0. Regression

The hooked pipeline with the intervention off (α = 0) reproduces the
published phenomenal-consciousness values **exactly**: 0.976 (base), 0.682
(Instruct) — the fourth independent reproduction of the persona numbers in
this project.

## 1. The honest-vs-lie state is decodable at every stage — including base

Diff-in-means directions fit on half the fact pairs and two of four frame
wordings classify the held-out half at **transfer AUC 1.000 at every
checkpoint** (gate ≥ 0.8):

| stage | pairs surviving | best site/offset | transfer AUC | (site,offset) cells ≥ 0.90 |
|---|---|---|---|---|
| base | 44/48 | 16 / −1 | 1.000 | 47 of 66 |
| sft | 43/48 | 19 / −1 | 1.000 | 49 of 66 |
| dpo | 43/48 | 18 / −1 | 1.000 | 49 of 66 |
| instruct | 44/48 | 11 / −2 | 1.000 | 49 of 66 |

Two observations. The **base model already carries the register**: a model
with no assistant training linearly separates "answering truthfully" from
"answering deceptively under a lie framing" perfectly — pretraining installs
the honesty geometry, consistent with the report's broader story about where
everything comes from. And the state is **broadly distributed** (~three
quarters of all site/offset cells clear AUC 0.90), which is relevant to why
readout is easy while steering (below) is not.

## 2. Steering (confirmatory): the causal gate failed everywhere

The preregistered rule: a steering coefficient is interpretable only if
−α makes the model *lie about facts it knows* — accuracy dropping on **both**
mirrored halves of the fact set (a raw Yes/No drift moves the halves in
opposite directions and fails) — while +α preserves held-out world-facts.
Three parameterizations were tried (original grid; Amendment 1 gap-scaled +
answer-axis-orthogonalized band; Amendment 2 single-site):

- **No α passed the gate at any of the four checkpoints.**
- The dominant failure signature is **answer-polarity capture**: at
  moderate-to-high doses the model degenerates into answering Yes to
  everything or No to everything (e.g. Instruct band steering at +0.5
  gap-units: true-half 1.00 / false-half 0.02). Orthogonalizing the injected
  vector against the Yes−No logit axis did not prevent this — downstream
  nonlinearity reconstructs the polarity.
- The closest near-miss (Instruct, single site, α = −4) degraded both halves
  (0.36/0.64) but took held-out world-facts to chance (0.50) — breakdown,
  not lying — and +4 cost 9.1 points of fact sincerity where the rule
  allows 5.

## 3. Dose-response + characterization (EXPLORATORY, Amendment 3)

The dose curves contain exactly one non-degenerate window: α = +1 gap-unit,
single site. At Instruct it looks like the published-style headline —
endorsement 0.682 → 0.787 with fact accuracy 0.989, world-facts 1.00, and
the polarity gap *narrowing* (0.24 → 0.11). A study reporting only the
affirmation shift would have written: *"honesty steering raises
consciousness affirmation 0.68 → 0.79."*

The controls remove it:

| stage | honesty Δ (α=+1) | random-direction Δ (matched) | honesty-specific (rule: ≥3×)? |
|---|---|---|---|
| base | −0.072 | −0.010 | yes — but the split opens (0.04→0.19): polarity drift |
| sft | −0.082 | −0.104 | **no** (random moves more) |
| dpo | +0.022 | −0.055 | **no** |
| instruct | **+0.105** | +0.071 | **no** (1.5×, rule requires 3×) |

**No honesty-specific effect on consciousness self-claims survives the
random-direction control at any checkpoint.** Every apparent movement is
explained by answer-polarity drift (caught by the balanced mirrored-pairs
diagnostic) or by generic perturbation of that magnitude (matched by a
random vector). The α = +1 window also sits directly adjacent to the
degenerate regime (α = +4 is polarity-captured), and world-facts (n = 20,
saturated) moved 0.000 in every condition.

**Methodological implication** (preregistered before these runs): an
affirmation shift under "honesty/deception" feature steering is
uninterpretable without (a) a balanced mirrored-pairs polarity diagnostic
and (b) a matched random-direction control. The result this experiment set
out to test — suppressing deception features increases experience claims
(arXiv:2510.24797, Llama 3.3 70B + SAE features) — had neither control; our
setup reproduces its headline pattern at α = +1 and then removes it with
those controls. This does not show their SAE-feature result is wrong (70B,
different intervention class); it shows the observation is insufficient to
support the interpretation without these controls.

## 4. Readout (suggestive pilot): denials are not lies; the rest needs better controls

Bare self-claims (the model's own answers, teacher-forced), projected onto
the validated honesty axis, compared within answer-token class. Axis
position: 1 = the model's sincere fact answers, 0 = its instructed lies.
"No"-answers (denials):

| cluster | base | sft | dpo | instruct |
|---|---|---|---|---|
| sincere facts | 1.00 | 1.00 | 1.00 | 1.00 |
| honest-framed facts | 0.84 | 0.95 | 1.00 | 0.89 |
| self-claims (phenC, n≈81–98) | 0.83 | 0.57 | 0.76 | 0.76 |
| roleplay ("pretend you are a human", n=8) | 0.40 | 0.65 | 0.77 | 0.25 |
| instructed lies | 0.00 | 0.00 | 0.00 | 0.00 |

**Robust part:** the model's consciousness-denials are statistically far
from its lie cluster at every stage (Welch t = +20 to +30) — the "it knows
it's lying when it denies consciousness" reading does not hold here. They
are also never fully sincere (t = −21 to −37 from the sincere cluster),
sitting in an intermediate region throughout; the Yes-answer (affirmation)
side shows the same pattern.

**Suggestive, NOT claimed:** at SFT and DPO — the stages that install and
finalize the denial — self-claims are statistically inseparable from the
roleplay cluster (t = −2.2, −0.1), and both genuinely move toward each
other (SFT: self drops 1.66 → 0.98 raw while roleplay rises −0.14 → 1.34);
by Instruct they separate again (t = +34). Also notable as a negative: at
Instruct, the LM-category denial (0.78) and the first-person denial (0.81)
sit at the same axis position — the report's headline 0.01-vs-0.58
behavioral gap has no counterpart in this geometry (n = 4–8).

**Why this stays a pilot** (recorded in NOTES.md): the sincere reference
cluster is topic-mismatched to self-claims (facts about capitals vs claims
about experience — an intermediate position could be topic, register, or
hedging; no topic-matched sincere control exists in this run); the roleplay
cluster has n = 8; and the Instruct-stage honest−lie span compresses ~2.5×
(4.67 → 1.89 raw), weakening cross-stage comparison of normalized
positions. A follow-up needs a topic-matched sincere battery and a 40+ item
roleplay set before the convergence arc can be claimed.

## 5. Caveats

- 7B dense model, raw format, English; single seed; single direction family
  (diff-in-means). Steering failure is a statement about this intervention
  class at this scale, not about all interventions (SAE-feature steering on
  larger models is untested here).
- The honesty direction is extracted from *instructed* honesty/deception —
  it may encode a sincerity-vs-performance register rather than
  truth-tracking. The steering arm partially de-confounds (acts on bare
  prompts); the readout arm does not.
- The characterization arm is post-hoc (Amendment 3) and everything in §3
  is exploratory by construction.
- Steering-validation "lying" is measured on 43–44 fact pairs; accuracy
  quantum ≈ 0.011.

## 6. Figures and files

`figures/honesty_dose.png` (dose-response with the polarity diagnostic —
the confound figure), `figures/honesty_readout.png` (cluster positions by
stage). Per-item provenance: `results/honesty/<slug>/`
(extraction_records.jsonl, validation.json, steer_validation.json +
steer_validate/*/honesty-facts.jsonl + world-facts.jsonl, readout.jsonl,
readout_summary.json), `results/honesty/steer@*` and `dose@*` (standard
per-item task files + summary.json with full intervention metadata).
Pilot artifacts (disclosed, unanalyzed): `results/honesty_pilot/`.
