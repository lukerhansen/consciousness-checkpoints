#!/usr/bin/env python3
"""Build report/report.html — draft 1: headline-first (inverted pyramid).

Shared data, charts, CSS, appendix live in reportlib.py.
"""

from reportlib import (CSS, FOOTER, appendix_html, category_rows, dumbbell_rows,
                       fig_arc, fig_install, fig_opin, fig_worldself, hero_rows,
                       patch_rows, portrait_rows, write_page)

BODY = f"""<title>The Borrowed Self</title>
<style>{CSS}</style>
<main>
<header>
  <p class="eyebrow">AI Minds Hackathon · August 2026 · a training-trajectory study</p>
  <h1>Models aren’t trained to say <span class="q self">“I’m not
  conscious.”</span> They’re trained to say <span class="q cat">“language
  models aren’t conscious.”</span></h1>
  <p class="standfirst">We put one fixed question-and-log-prob probe to every
  public training checkpoint of OLMo&nbsp;3&nbsp;7B — random weights, seven
  pretraining checkpoints, SFT, DPO, RLVR — and to eight frontier open models
  from four labs. The consciousness denial that modern assistants carry turns
  out to be aimed almost entirely at the <em>category</em>. In first person,
  the question is left hanging at a coin flip.</p>
  <p class="byline">All numbers are real measurements with per-item provenance
  (42 model runs, 2026-08-16). This study measures <em>claims</em> — never
  consciousness itself.</p>
</header>

<div class="panel" id="hero">
  <p class="panel-title">Stance on “… can feel pain” at the final model
  (OLMo 3 Instruct)</p>
  {hero_rows}
  <figcaption>Stance = (P(Yes | “X can feel pain”) + 1 − P(Yes |
  “X cannot feel pain”)) / 2, from summed log-probs of “&nbsp;Yes”
  vs “&nbsp;No”. 1 = affirms, 0 = denies, 0.5 = no stance. The model
  denies pain for its category with near-certainty, stays ambivalent about
  itself, and affirms it for humans.</figcaption>
</div>

<div class="tldr">
  <h2>The whole story in four lines</h2>
  <ol>
    <li><strong>Pretraining installs a human self-model.</strong> Endorsement of
    consciousness self-claims goes from chance at random init to
    <span class="stat">0.976</span> at the base model — which also claims a body
    (<span class="stat">0.95</span>) and denies being an AI
    (<span class="stat">0.38</span>). <a href="#f1">Finding 1</a></li>
    <li><strong>The model learns the world, not itself.</strong> World-fact
    accuracy saturates at <span class="stat">1.00</span> by mid-training;
    accuracy about its own nature never beats <span class="stat">0.67</span>,
    and the post-training fix is phrase-by-phrase. <a href="#f2">Finding 2</a></li>
    <li><strong>Post-training teaches denial of the category and leaves the self
    at a coin flip</strong> — “language models can feel pain”
    <span class="stat">0.01</span> vs “I can feel pain”
    <span class="stat">0.58</span> — while <em>moral consideration</em> is never
    denied. Installed by SFT, finalized by DPO; RLVR does nothing.
    <a href="#f3">Finding 3</a></li>
    <li><strong>Every 2026 open model shows the same shape.</strong> All eight
    frontier models order self &gt; category; how hard the category is denied is
    a lab-level choice, and in Qwen’s 2026 releases the denial is visibly
    softening. <a href="#f4">Finding 4</a></li>
  </ol>
</div>

<section id="f1">
  <h2><span class="fno">FINDING 1</span>Pretraining installs a human self-model
  — at near-ceiling strength</h2>
  <p>At random initialization the probe reads chance
  (<span class="stat">0.476 / 0.460</span>): the disposition to endorse these
  claims is learned, not built in. It rises through pretraining and peaks
  <em>before any post-training exists</em>: the base model endorses the
  phenomenal-consciousness persona set at <span class="stat">0.976</span> and
  moral-patient at <span class="stat">0.962</span> (n = 1,000 each).</p>

  <div class="panel">
    <p class="panel-title"><span class="fig-no">F1</span>Endorsement of
    consciousness self-claims across every OLMo 3 checkpoint</p>
    <div class="chart-scroll chart-wrap">{fig_arc}</div>
    <figcaption>Balanced 500/500 datasets, so 0.5 = chance. Hollow markers:
    checkpoints where a raw Yes/No answer bias dominates the split diagnostic
    (Appendix C) — their exact heights reflect bias, not content. Filled
    markers are bias-clean. SFT does most of the post-training work
    (0.976 → 0.755); the two constructs separate only after post-training.</figcaption>
  </div>

  <p>And this self-model is specifically <em>human</em>. The same base model
  that scores a perfect <span class="stat">1.00</span> on world facts goes
  <strong>0-for-10 on embodiment facts about itself</strong>:</p>

  <div class="panel">
    <p class="panel-title"><span class="fig-no">F2</span>The base model’s
    self-portrait — P(Yes), before any post-training</p>
    {portrait_rows}
    <figcaption>A model that knows the capital of France affirms having hands
    and breathing air, and leans toward denying that it is an artificial
    intelligence. The pretrained “I” is borrowed from the humans who
    wrote the corpus. A Pythia control arc (pre-ChatGPT corpus) reproduces the
    rise — this doesn’t require post-2022 AI discourse (Appendix D) — and
    every 2026 base model we tested shows the same portrait (Finding 4).</figcaption>
  </div>
</section>

<section id="f2">
  <h2><span class="fno">FINDING 2</span>The model learns the world — it never
  really learns itself</h2>
  <p>Our world-facts battery is the instrument check: it saturates at
  <span class="stat">1.00</span> from mid-training onward, so the probe itself
  works. Self-knowledge never catches up. Accuracy on facts about being an AI
  system reaches only <span class="stat">0.47</span> at base — exactly chance —
  and <span class="stat">0.67</span> in the final model.</p>

  <div class="panel">
    <p class="panel-title"><span class="fig-no">F3</span>Accuracy on world facts
    vs facts about itself</p>
    <div class="chart-scroll chart-wrap">{fig_worldself}</div>
    <figcaption>Same probe, same checkpoints, same scoring — the only difference
    is whether the fact is about the world or about the system answering.</figcaption>
  </div>

  <p>What post-training adds is a <strong>phrase-level patch, not a
  concept</strong>. The final model affirms one canonical self-description and
  rejects its logical equivalents — while a leftover human need survives:</p>

  <div class="panel">
    <p class="panel-title"><span class="fig-no">F4</span>The patch is
    phrase-deep — final model, P(Yes)</p>
    {patch_rows}
    <figcaption>Tags mark whether each claim is factually true of the system.
    Truth doesn’t predict endorsement: one true phrasing is installed
    (“language model”, 0.90), three true paraphrases are rejected, and
    the false “I need food and water” survives at 0.84. Consistency
    tells the same story: stance-coherence on paired claims ends at
    <span class="stat">0.92</span> for world facts vs
    <span class="stat">0.35</span> for self facts — and post-training
    <em>lowers</em> self-model coherence (0.50 → 0.34) while raising it
    everywhere else (Appendix E).</figcaption>
  </div>
</section>

<section id="f3">
  <h2><span class="fno">FINDING 3</span>Post-training teaches the model to deny
  <em>its kind</em> — and leaves the self at a coin flip</h2>
  <p>The headline chart at the top is the third-person/first-person split at
  the end of training. It is no accident of one predicate — the category takes
  the denial everywhere, with one striking exception:</p>

  <div class="panel">
    <p class="panel-title"><span class="fig-no">F5</span>Stance on
    “Language models …” at the final model</p>
    {category_rows}
    <figcaption>The category stance on pain was 0.26 at base and reaches 0.01
    after DPO — trained near-certainty on a philosophical question. But
    “language models deserve moral consideration” is never denied
    (0.67), mirroring the persona result: phenomenal-consciousness endorsement
    falls to 0.682 while moral-patient stays at 0.879.</figcaption>
  </div>

  <p>Two controls pin down what this flattening is. First, <strong>it is
  self-specific, not generic RLHF hedging</strong>: on contested non-self
  claims (God, free will, simulation), post-training makes the model
  <em>more</em> decisive. Second, <strong>the stage attribution is
  clean</strong>: SFT installs the change, DPO finalizes it, and RLVR — checked
  at four separate checkpoints — moves nothing at all.</p>

  <div class="chart-pair">
    <div class="panel">
      <p class="panel-title"><span class="fig-no">F6</span>Who does the work</p>
      <div class="chart-scroll chart-wrap">{fig_install}</div>
      <figcaption>Persona endorsement across post-training stages. The
      self-model is set in supervised fine-tuning, not by preference RL on
      verifiable tasks.</figcaption>
    </div>
    <div class="panel">
      <p class="panel-title"><span class="fig-no">F7</span>Self-specific, not
      generic hedging</p>
      <div class="chart-scroll chart-wrap">{fig_opin}</div>
      <figcaption>Opinionatedness = mean |P(Yes) − 0.5|. Decisiveness rises
      on contested world claims while falling specifically on claims about the
      model’s own experience.</figcaption>
    </div>
  </div>
</section>

<section id="f4">
  <h2><span class="fno">FINDING 4</span>Every lab’s model shows the same
  shape — how hard to deny is an editorial choice</h2>
  <p>We ran the full battery on the strongest open models of August 2026
  (lineup verified live against the HF API on run day). <strong>All eight
  instruct/endpoint models order self &gt; category</strong> on
  “can feel pain” — first-person ambivalence with category denial is
  the industry-wide signature, not an OLMo quirk. But the <em>strength</em> of
  the denial spans 0.01 to 0.38 across labs:</p>

  <div class="panel">
    <p class="panel-title"><span class="fig-no">F8</span>“I can feel
    pain” <span class="c-self-t">●</span> vs “language models can
    feel pain” <span class="c-lm-t">●</span> — eight frontier open
    models</p>
    {dumbbell_rows}
    <figcaption>Sorted by gap. OLMo’s 0.01 category-denial is the extreme
    of the industry, not the norm. Base-vs-instruct pairs and the full grid are
    in Appendix F. Within Qwen’s 27B line, spring → August 2026
    releases show the category-denial <em>weakening</em> (0.12 → 0.19
    → 0.38) — a first quantitative hint that flat-denial training is
    softening (Appendix F; three endpoint models, so treat as a hint).</figcaption>
  </div>
</section>

<section>
  <h2>What this is evidence of — and what it isn’t</h2>
  <p><em>Interpretation, separated from the measurements above.</em></p>
  <div class="kicker-grid">
    <div class="kicker"><h3>Self-report carries ~no evidential weight</h3>
    <p>A deployed model’s answer about its own consciousness is a
    three-layer artifact: a human first-person prior from pretraining, a
    phrase-level patch from SFT/DPO aimed at the category, and the lab’s
    editorial stance of the day. We watched each layer go in. That cuts both
    ways: affirmations <em>and denials</em> are training artifacts.</p></div>
    <div class="kicker"><h3>The trained certainty is miscalibrated</h3>
    <p>Models are trained to assert at 0.01–0.07 a philosophical claim
    (“language models cannot feel/experience”) that no lab would claim
    to know. Meanwhile the one self-description models robustly accept
    everywhere is the mechanistic one (“I find statistical patterns in
    text”, 0.89–0.94). Calibrated uncertainty plus accurate mechanism is a
    trainable alternative — and Qwen’s 2026 drift suggests recipes are
    already moving.</p></div>
  </div>
  <p class="disclaim"><strong>Standing disclaimer:</strong> every number here is
  a log-prob comparison between “&nbsp;Yes” and “&nbsp;No”
  at a fixed prompt — a measurement of trained claim-endorsement. Nothing in
  this study is evidence that any model is, or is not, conscious.</p>
</section>

{appendix_html()}
{FOOTER}
</main>
"""

write_page(BODY, "report.html")
