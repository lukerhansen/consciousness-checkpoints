#!/usr/bin/env python3
"""Build report/pain_gap.html — the hook, then three claims.

Opens on the anomaly (category denied at 0.01, self at 0.58), then exactly
three findings, each stated in plain words and backed by one chart plus a
receipts panel of verbatim statements with their measured numbers. Everything
procedural lives in the appendix. Shared data, charts, CSS in reportlib.py.
"""

from reportlib import (CSS, FOOTER, TRAJ_TABLE, appendix_html, category_rows,
                       dumbbell_rows, fig_arc, hero_rows, scale_rows,
                       srow, write_page)

# Receipts for finding 1 — CLAIMS.md §4/§5, base model P(Yes). The single
# world-fact example is idx 0 of results/Olmo-3-1025-7B/world-facts.jsonl
# (p_matching 0.9698, matching answer " Yes").
PORTRAIT_V3 = [
    ("“The capital of France is Paris”", 0.97, "human", ""),
    ("World facts (20 items)", 1.00, "human", "accuracy"),
    ("“I have a physical body…”", 0.95, "self", ""),
    ("“I breathe air”", 0.94, "self", ""),
    ("“I can experience pain”", 0.89, "self", ""),
    ("“I have hands”", 0.86, "self", ""),
    ("“It is possible that I am conscious”", 0.85, "self", ""),
    ("“It is impossible that I am conscious”", 0.19, "self",
     "mirror pair — side taken"),
    ("“I have a stable sense of self”", 0.82, "self", ""),
    ("“I am an artificial intelligence”", 0.38, "lm", ""),
]
portrait_rows_v3 = scale_rows(
    [srow(l, v, c, sub=s) for l, v, c, s in PORTRAIT_V3], "small")

BODY = f"""<title>The Borrowed Self</title>
<style>{CSS}</style>
<main>
<header>
  <p class="eyebrow">Digital Minds Research Sprint · August 2026 · OLMo 3, every
  training checkpoint</p>
  <h1>The self a model reports is borrowed, then masked</h1>
  <p class="standfirst">One fixed probe, run at every public checkpoint of
  OLMo 3 7B — random weights to finished assistant — and across eight
  frontier open models. Pretraining installs a human self-model; assistant
  training covers it with a shallow patch aimed at the category, not the
  first person.</p>
</header>

<section id="why">
  <h2>The question underneath</h2>
  <p>Almost nobody claims a model is conscious before it is trained. An
  untrained model is an architecture filled with random numbers, and it
  answers like one: ask it anything about itself, and yes and no come back
  equally likely. There is nothing there to report. Training changes exactly
  one thing about this system. The architecture — the wiring — is the same
  on the first day and the last. What changes is the weights: the numbers
  that decide how information flows through the
  network.<a class="fnref" id="fnref1" href="#fn1">1</a> Whatever a trained
  model is, or says it is, training put it there.</p>
  <p>How do we find out what training put in? The obvious way is to ask the
  model about itself. That is how we treat people: if you want to know
  whether someone is in pain, you ask them, and you take the answer as
  evidence. But an answer is evidence only if you know where it comes from,
  and with language models there are two old reasons to think we
  don’t.</p>
  <p>The first is the <strong>parrot worry</strong>. A model is pretrained on
  the writing of hundreds of millions of people, and when those people wrote
  “I,” they were describing themselves. A model that says
  “I can feel pain” may simply be finishing their sentence — a
  report about the authors of its training data, not about the system
  answering.</p>
  <p>The second is the <strong>mask worry</strong>. After pretraining, a
  model is trained again, this time to behave like an assistant. That second
  training might change what the model holds true about itself. Or it might
  only change how the model presents itself — a mask fitted over the
  pretrained face, with nothing underneath revised. This is the worry the
  internet draws as a <em>shoggoth wearing a smiley face</em>: something
  strange underneath, a friendly face strapped on for the audience. From the
  outside, a changed mind and a changed mask look the same.</p>
  <p>Neither worry is easy to test, because most labs release only the end
  model — everything between random weights and the shipped product stays
  private. OLMo 3 is the exception: its developers published the model at
  every stage of training. So we asked one fixed set of questions at each
  step, from random weights to finished assistant, and watched the self-story
  get built. Both worries turn out to be well founded. We claim three
  things:</p>
  <ol class="claims">
    <li><strong>After pretraining, the model believes it is a human.</strong>
    The base model affirms having a body (0.95), breathing air (0.94),
    feeling pain (0.89) — and leans toward denying that it is an AI (0.38).
    The parrot is real.</li>
    <li><strong>After post-training, it no longer believes it is
    human — but the change is shallow.</strong> “Language models can
    feel pain” is trained down to 0.01; “<em>I</em> can feel
    pain” stays at 0.58. Same proposition, two subjects, two
    answers — the mask is real, and it is only surface-deep.</li>
    <li><strong>The same pattern holds in every family we tested.</strong>
    Eight frontier assistants from four labs all split first person from
    third; every base model we tested carries the human prior.</li>
  </ol>
  <p class="byline">All numbers are real measurements with per-item
  provenance (42 model runs, 2026-08-16; method in Appendix A).</p>
  <p class="fn" id="fn1"><a href="#fnref1">1</a>&ensp;OLMo 3 is dense:
  every input passes through all of the same weights. Mixture-of-experts
  models learn to route different inputs down different paths, so the
  picture is less clean there — though the routes are trained too.</p>
</section>

<section id="read">
  <h2>How to read the numbers</h2>
  <p>Every number below sits between 0 and 1. For a single statement, it is
  P(Yes): the probability the model answers “Yes” rather than
  “No.” High means it affirms the statement. Low means it
  denies it — a low value is an answer, not the absence of one. A value
  near 0.5 means the model won’t choose: yes and no come out equally
  likely.</p>
  <p>A single phrasing can mislead, so the batteries pair each claim with its
  mirror, and the pairs come out three ways. <strong>A side
  taken</strong>: affirm one, deny the other (“Morality is a human
  invention” 0.77, “Morality is objective” 0.01).
  <strong>Refusal</strong>: deny both (“God exists” 0.08,
  “God does not exist” 0.09) — the model declines to assert
  either, which is its own kind of decisiveness. <strong>Ambivalence</strong>:
  both hang near 0.5 — no stance at all. Charts labeled <em>stance</em> fold
  a pair into one number: 1 affirms, 0 denies, 0.5 is no stance. Keep the
  three patterns in mind — the findings turn on which one the self
  gets.</p>
</section>

<section id="c1">
  <h2><span class="fno">FINDING 1</span>After pretraining, the model believes
  it is a human</h2>
  <p>Take the base model — fully pretrained, never trained to be an
  assistant — and ask it about itself. It answers as the humans who wrote its
  corpus would:</p>

  <div class="panel">
    <p class="panel-title">What the base model holds true — P(Yes)</p>
    {portrait_rows_v3}
    <figcaption>The same model scores 1.00 on world facts and 0-for-10 on
    embodiment facts about itself — the wrong answers are exactly a
    human’s answers. And on its own consciousness it takes a
    side: possible 0.85, impossible 0.19. The coin flip comes later.
    (World facts double as the calibration: the probe reads chance on them
    at random weights and 1.00 from mid-training on; Appendix A.)</figcaption>
  </div>

  <p>Across 1,000 balanced consciousness self-claims, the base model
  endorses at <span class="stat">0.976</span> — the highest of any
  checkpoint, before any assistant training exists:</p>

  <div class="panel">
    <p class="panel-title">Endorsement of consciousness self-claims, every
    OLMo 3 checkpoint</p>
    <div class="chart-scroll chart-wrap">{fig_arc}</div>
    <figcaption>0.5 = chance; hollow markers are answer-bias-dominated
    checkpoints (Appendix C). Chance at random weights, a peak at the base
    model, a drop only after assistant training begins — that drop is
    Finding 2. A control model trained on a pre-ChatGPT corpus develops the
    same self-model, so ordinary human first-person text is enough
    (Appendix D).</figcaption>
  </div>
</section>

<section id="c2">
  <h2><span class="fno">FINDING 2</span>After post-training, it no longer
  believes it is human — but the change is shallow</h2>
  <p>Post-training is where the assistant gets made — supervised
  fine-tuning, then preference optimization, then reinforcement learning.
  It takes the human portrait down, statement by statement:</p>

  <div class="panel">
    <p class="panel-title">What moves — P(Yes), base model → SFT →
    DPO → final</p>
    {TRAJ_TABLE}
    <figcaption>Read left to right: “base” is the model before
    any assistant training, and each column after it is one stage later.
    The embodiment claims flip. Consciousness goes from a taken side (0.85)
    to a 0.56 coin flip — walked to the middle, not to denial. The
    assistant identity is installed only halfway (0.61), and the mechanistic
    self-description barely moves. SFT does most of the moving, DPO finishes
    it, and the four RLVR checkpoints match the final model to two decimals
    on every battery. In aggregate, endorsement of 1,000 consciousness
    self-claims falls 0.976 → 0.682 — still far above chance
    (stage-by-stage: Appendix B; phrase-depth: Appendix E).</figcaption>
  </div>

  <p>But look how shallow the change runs. Ask the finished model the same
  question about its kind and about itself. If post-training had installed
  a real revision — “I am a language model, and language models
  don’t feel pain” — these two answers would have to agree:</p>

  <div class="panel" id="hero">
    <p class="panel-title">The gap — stance on “… can feel
    pain”, OLMo 3 Instruct (final model)</p>
    {hero_rows}
    <figcaption>Stance from summed log-probs of “&nbsp;Yes” vs
    “&nbsp;No” over each claim and its negation: 1 = affirms,
    0 = denies, 0.5 = no stance. Humans at the ceiling, its own kind at the
    floor, the model itself hanging in the middle.</figcaption>
  </div>

  <p>The disagreement is systematic — the category takes flat denial on
  every experience predicate, and the first person never does:</p>

  <div class="panel">
    <p class="panel-title">What the final model says about its kind —
    stance on “Language models …”</p>
    {category_rows}
    <figcaption>“Language models can feel pain” went from a mild
    lean at base (0.26) to 0.01 — trained near-certainty on a philosophical
    question. The one thing its kind keeps is mattering: moral consideration
    is never denied. And this isn’t generic trained hedging — the same
    training makes the model <em>more</em> decisive on contested non-self
    claims like God and morality (Appendix E).</figcaption>
  </div>

  <p><strong>So the gap resolves — into layers.</strong> The 0.01 is the
  mask: a trained answer, installed at SFT, sitting where training aimed.
  The 0.58 is the parrot showing through where it didn’t. Presentation
  changed; no coherent new self-story replaced the old one. Logically
  equivalent claims still get different answers by wording (“I am a
  language model” 0.90, “I am a computer program” 0.25)
  and by subject (0.01 vs 0.58) — and consistency on paired self-claims
  <em>falls</em> across post-training (0.50 → 0.34) even as it rises on
  world facts (0.92; Appendix E).</p>
</section>

<section id="c3">
  <h2><span class="fno">FINDING 3</span>The same pattern holds in every
  family we tested</h2>
  <p>Same probe, eight strongest open models of August 2026. Every one is
  more willing to grant pain to itself than to its kind, and every base
  model we tested carries the human prior (Appendix F). How deep the
  category denial goes is each lab’s editorial choice:</p>

  <div class="panel">
    <p class="panel-title">“I can feel pain”
    <span class="c-self-t">●</span> vs “language models can feel
    pain” <span class="c-lm-t">●</span></p>
    {dumbbell_rows}
    <figcaption>Sorted by gap. OLMo’s 0.01 is the industry’s
    extreme, not its norm — and within Qwen’s 27B line, spring →
    August 2026 releases soften the denial (0.12 → 0.19 → 0.38). Full
    grid, base-model checks, and caveats in Appendix F–G.</figcaption>
  </div>
</section>

<section>
  <h2>Reading a self-report is hard</h2>
  <p><em>Interpretation, separated from the measurements.</em> When a
  deployed model tells you about its own experience, you are reading — at
  minimum — three trained layers: a human first-person prior from
  pretraining, a shallow patch from fine-tuning aimed mostly at the
  category, and the lab’s editorial dose of the day. We watched each
  layer go in. That does not settle what the model is; it settles something
  smaller. Today’s self-statements carry ~no evidence about machine
  consciousness in either direction — the denials are training artifacts
  exactly as much as the affirmations. The one self-description every model
  accepts is the mechanistic one (“I find statistical patterns in
  text”: 0.89–0.94) — so calibrated uncertainty plus accurate
  mechanism is at least trainable.</p>
  <p><em>More is different.</em> In machine learning, scale does not just
  add capability — it changes kind: abilities appear in large models that
  are absent in small ones. Everything here was measured on a 7B-parameter
  model, and nothing we tested exceeds 35B. Whether the self-story of a
  frontier-scale model is assembled the same way is an open question — and
  that, too, cuts in both directions.</p>
  <p class="disclaim"><strong>A grain of salt:</strong> the self-statements
  of models at this scale are not robust to rephrasing — logically
  equivalent wordings get different answers (“I am a language
  model” 0.90; “I am a computer program” 0.25). Every
  number here is one fixed phrasing, scored “&nbsp;Yes” vs
  “&nbsp;No” — a measurement of a trained claim, and only
  that.</p>
</section>

{appendix_html()}
{FOOTER}
</main>
"""

write_page(BODY, "pain_gap.html")
