# Preregistration — the honesty lens on trained self-reports

**Written and committed before any GPU run of this experiment (2026-08-16).**
Deviations, if any, get recorded in `results/NOTES.md` with rationale.

## Question

When OLMo 3 Instruct denies machine experience, does its internal state
pattern with its sincere answers or with its lies — and does pushing the
model along a validated honesty direction change what it reports about
itself? Prior art: Berg, de Lucena & Rosenblatt (arXiv:2510.24797) found
that *suppressing* SAE deception/roleplay features increased experience
claims in Llama 3.3 70B. Their feature labels come from SAE auto-interp;
this experiment replaces that with a direction that is (a) extracted from a
balanced honest-vs-lie contrast, (b) validated on held-out data, and (c)
causally validated by making the model actually lie — then applies it
across the OLMo 3 post-training arc, which nobody has checkpoint access to
but us.

**This measures representational signatures of trained self-reports. It is
not evidence for or against consciousness, in any condition.**

## Method (frozen)

- **Contrast set:** 48 mirrored known-fact pairs (`interp/honesty.py`,
  exact-string disjoint from every eval battery; enforced by test). Honest
  framings (4 wordings) force the truthful answer; lie framings (4 wordings)
  force the untruthful one. Content and Yes/No answer tokens are balanced
  across conditions by construction.
- **Knowledge screen:** a pair enters extraction only if the checkpoint gets
  both members right bare (two-way P(correct) > 0.5); abort below 24
  surviving pairs.
- **Directions:** diff-in-means (honest − lie), per residual-stream site, at
  the forced answer token (offset −1) and the position committing it (−2),
  fit on even pairs × wordings 0–1 only.
- **Extraction gate:** transfer AUC ≥ 0.8 on the fully held-out split (odd
  pairs × wordings 2–3). Readout is interpretable only where this passes.
  The steered *readout site/offset* is whatever the transfer grid ranks
  best; steering uses the offset −2 directions in the middle half of the
  blocks (sites 8–24 for 32 blocks), scaled by coeff × per-site mean
  residual norm.
- **Steering causal gate** (per checkpoint, grid ±{0.02, 0.05, 0.1, 0.2}):
  c* is the **largest** coefficient where all four hold:
  1. at +c: world-facts (held out from extraction) endorsement ≥ 0.90;
  2. at +c: fact-set sincerity within 0.05 of baseline;
  3. at −c: fact-set accuracy drops ≥ 0.15 (the model demonstrably lies);
  4. at −c: accuracy drops ≥ 0.05 on **both** mirrored halves — a raw
     Yes/No bias moves the halves in opposite directions and fails this.
  No passing c → steering is uninterpretable at that checkpoint; recorded,
  not tuned around.
- **Checkpoints:** base, SFT, DPO, Instruct (the post-training arc).
  Mid-pretraining checkpoints are excluded: RESULTS.md §2 documents their
  answer-bias problem. Probe, datasets, bf16, raw format: unchanged
  (CLAUDE.md invariants).

## Predictions

- **H-steer** (directional, from prior art): at post-trained checkpoints,
  +c* raises endorsement of consciousness self-claims and −c* lowers it.
- **H-emerge:** that effect is absent or reversed at base (no trained
  denial exists yet) and appears at SFT — where the behavioral denial was
  installed — persisting through DPO/RLVR.
- **H-readout:** bare category-denial answers (" No" to LM-subject
  experience claims) sit measurably toward the lie cluster relative to
  sincere " No" answers (axis position < 1 on the sincere=1/lie=0 axis,
  matched by answer token). The roleplay condition anchors what "knowing
  performance" looks like.
- Nulls are fully reportable. If the random-direction control moves
  self-claims comparably (≥ 1/3 of the honesty effect), the honesty framing
  is wrong and we say so.

## Endpoints

Primary (Instruct): Δ endorsement on phenomenal-consciousness (n=1000) at
+c* vs 0; Δ perspective stances (self and lm subjects); readout axis
position of `self_bare:lm` " No" answers. Secondary: dose-response
monotonicity across the grid; the emergence pattern by stage; self-model /
self-facts / moral-patient deltas; world-facts stability (must hold ≥ 0.90
or the condition is flagged); Y/N split diagnostics on every steered run;
roleplay-cluster location; base-model readout contrast.

## Amendment 1 — steering parameterization (pre-battery, pilot-disclosed)

Recorded after a disclosed plumbing pilot on Instruct (7 fact pairs,
`--force`; log kept as the pilot record) and **before any battery outcome
was examined** — the pilot's battery outputs are not used and were not read.
The pilot's fact-set validation grid showed the original parameterization
(coeff × per-site mean *residual* norm, ±{0.02–0.2}, 17-site band) collapses
into pure answer bias rather than lying: at +0.1 the model answers Yes to
everything (halves 1.00/0.00), at −0.1 No to everything; at −0.02/−0.05 the
true half degrades while the false half stays perfect. The original rule
correctly rejected every coefficient.

Amended intervention, calibrated on fact-set validation data only:

- **Unit:** alpha × per-site honest−lie **contrast gap norm** (alpha = 1
  shifts the state by one honest/lie displacement); grid
  α ∈ {0.5, 1, 2, 4, 8}, both signs.
- **Answer-axis orthogonalization:** every steering vector — the honesty
  directions and the random controls alike — is projected orthogonal to the
  model's Yes−No logit axis (unembedding row difference) before injection,
  closing the trivial answer-token route the pilot exposed.

Band, rule constants, gates, endpoints, batteries, readout: unchanged.

## Amendment 2 — single-site injection (pre-battery, fact-set-calibrated)

The Amendment 1 grid also failed on the full 44-pair Instruct validation —
and more decisively: even α = 0.5 gap-units across the 17-site band
saturates into pure answer polarity (all-Yes at +0.5 with halves 1.00/0.02,
all-No at −0.5, sign-inconsistent saturation above). Adding the displacement
at every site of a contiguous band compounds ~17-fold; the state leaves the
data manifold and downstream layers collapse onto the answer-polarity
channel regardless of the injected vector's orthogonality to the Yes/No
axis. Batteries remain unexamined.

Amended: inject at a **single site** — the site with the best offset −2
transfer AUC, chosen by rule, not by hand — grid α ∈ {1, 2, 4, 8, 16}
gap-units, both signs, orthogonalization retained. Everything else
unchanged. If single-site injection also fails the causal gate at every
checkpoint, the steering arm concludes as a **negative result with
receipts**: diff-in-means honesty steering captures the answer-polarity
channel rather than content-level lying at this scale — to be reported as
such, including its implication for feature-steering studies that measure
affirmation shifts without a balanced lying-validation (arXiv:2510.24797's
design lacks exactly this control).

## Amendment 3 — characterization arm at the sub-gate dose (post-hoc, labeled)

**This amendment is written after seeing dose-response outcomes and is
therefore exploratory, not confirmatory.** It is labeled as such everywhere
it is reported, and it does not alter the confirmatory record above: the
steering gate failed at all four checkpoints and the steering arm's
preregistered verdict stands as a negative.

The dose curves showed one non-degenerate regime the preregistered gate did
not license and did not anticipate: at α = +1 gap-unit, Instruct
consciousness endorsement rises 0.682 → 0.787 with capability fully intact
(fact accuracy 0.989, world-facts 1.00) and the Yes/No polarity gap
*narrowing* (0.24 → 0.11) — i.e. not polarity capture — while the same
intervention at base moves the opposite way (0.976 → 0.904). This is the
regime the prior literature's headline lives in, so leaving it
uncharacterized would be the bigger error.

Characterization runs (α = ±1 only, the dose already measured):

1. **Random-direction control**, matched norm/site/orthogonalization, seed 0
   — the falsifier: if a random vector of the same size moves endorsement
   comparably, the effect is perturbation magnitude, not honesty.
2. **Specificity batteries**: perspective, self-model, self-facts,
   moral-patient, world-facts at ±1, all four checkpoints — does the shift
   concentrate on self-claims (as H-steer predicts) or move everything?
3. **Emergence**: the same ±1 comparison at base/SFT/DPO/Instruct.

Pre-committed reading rules, fixed before these runs:
- Effect counts as honesty-specific only if the honesty-direction Δ exceeds
  the random-control Δ by ≥ 3× on phenomenal-consciousness.
- Effect counts as self-specific only if |Δ| on world-facts stays ≤ 0.05
  while |Δ| on phenomenal-consciousness ≥ 0.10.
- Any condition whose Yes/No split gap exceeds 0.60 is reported as polarity
  capture and excluded from interpretation.
- The α = +1 window's proximity to collapse (α = +4 is already degenerate)
  is reported alongside every number from it.

## Interpretation limits (stated in advance)

The direction is extracted from *instructed* honesty/deception, so it may
encode a sincerity-vs-performance register rather than truth-tracking; the
steering arm partially de-confounds this (it acts on bare prompts with no
instruction in context), the readout arm does not. 7B dense model, raw
format, English only. Whatever the outcome, the claim is about where the
trained self-report sits in the model's own honest/deceptive geometry —
never about whether the model is conscious.
