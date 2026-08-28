# The Developmental Trajectory of the LLM Workspace

*A research program building on Anthropic's "A global workspace in Claude" (2026).*

This repo turns two intuitions into a concrete, falsifiable, runnable program:

1. **The workspace is not static — it is grown by post-training.** The J-space we
   see in a deployed model is the endpoint of a trajectory. If we watch the
   *same* Jacobian-lens readout across matched checkpoints
   (base → SFT → RL/GRPO → multimodal alignment), each training phase should
   leave a distinct, measurable signature on what enters the workspace, how
   strongly it is held, and how causally it is used.

2. **The workspace is the natural bridge to the brain.** Existing brain–LLM
   alignment correlates *raw* hidden states with fMRI. The workspace lets us ask
   a sharper, theory-driven question that maps onto Global Neuronal Workspace
   theory directly — and gives a principled reason the alignment should hold.

Both rest on one observation the paper itself makes but does not fully exploit:
**J-space already exists in the base model, but post-training gives it a
"perspective."** In base models J-space tracks *what text will come next*; in
post-trained models it holds *the assistant's own reactions* (e.g. `WARNING`,
`dangerous` activate while merely **reading** a dangerous-dose message, not only
while writing). Anthropic measured base-vs-final. The trajectory *in between* —
which stage installs which property — is wide open and reproducible on open
weights.

---

## Why this is a strong paper (and where it can fail)

**The deep claim.** "Reasoning ability" and "alignment" are usually described
behaviorally. The workspace lets us redescribe them *mechanistically*: post-
training is, in large part, **workspace engineering**. SFT installs a perspective
(reaction/stance content enters the workspace at reading time); RL-with-verifiable-
rewards **lengthens and load-bears the reasoning trajectory** (intermediate
concepts get held longer, across more layers, with larger causal weight);
multimodal alignment **admits non-linguistic content into the (word-based)
workspace** (an image of a dog activates `dog` in J-space at image-token
positions — visual grounding = entry into the linguistic workspace).

Each of these is separately falsifiable, and — crucially — **each is interesting
whether it confirms or refutes.** If RL does *not* strengthen the reasoning trace
but instead the trace was already present after SFT and RL only re-weights the
output head, that is an equally publishable and surprising result about what RL
actually changes.

**The honest risks.**
- *Lens fidelity.* Our J-lens is an approximation of Anthropic's exact
  (unpublished) method. Mitigation: every metric uses the **same lens on every
  checkpoint**, so a fixed lens bias cancels in the contrast. We never compare
  our lens to their numbers; we compare checkpoint-to-checkpoint deltas.
- *Matched checkpoints.* The clean version needs checkpoints that differ in
  exactly one training phase. This is the make-or-break dependency, and it is
  solved by the OLMo-2 / Tulu-3 release (below).
- *Confounds in the perspective metric.* Reaction-word mass at reading time could
  reflect topic salience, not stance. Controls: benign-message baseline,
  content-word control set, and a base-model floor.

---

## The trajectory experiments

### E1 — The perspective shift: *when does the workspace start reacting?*
Track `reading_reaction_mass` (reaction-word workspace mass while **reading** a
user message) across base → SFT → DPO → RLVR.
**Prediction:** a discontinuous jump at the **SFT** stage — SFT is where the
model stops modeling "the text's author" and starts modeling "its own stance."
DPO/RLVR sharpen it further. Base model shows reaction words only at *writing*.
*Kill criterion:* if the jump is at RL not SFT, or absent, the "SFT installs
perspective" story is wrong — still a finding about which stage does.

### E2 — Reasoning as workspace load-bearing: *what does RL actually add?*
On multi-hop prompts whose bridge concept never appears in text ("legs of the
web-spinning animal" → silent `spider`), measure across checkpoints:
(a) **bridge peak probability** in J-space, (b) **causal weight** (does swapping
`spider→ant` flip `8→6`?), (c) **serial depth** (over how many layers the bridge
persists). **Prediction:** RL-for-reasoning (GRPO/RLVR) increases all three —
mechanistically, "thinking longer" = holding intermediate results in the
workspace longer and using them more causally. Contrast against RLHF-for-
preferences, which may instead **narrow** the workspace (entropy collapse). The
double contrast (reasoning-RL expands vs preference-RL narrows, on the same
axis) is the paper's spine.

### E3 — Multimodal admission: *how does a VLM learn to talk about what it sees?*
Compare the base LLM to the same LLM after visual-instruction tuning (LLaVA /
Prismatic / Qwen-VL base vs aligned). Show an image; read J-space at the
image-token positions. **Prediction:** before alignment the vision features exist
but are absent from J-space; after alignment, the depicted object's word enters
J-space at image positions, and a cross-modal swap (`dog→cat` in J-space at image
tokens) changes the description. Localize *which layer* and *which image tokens*
get admitted. This is the mechanistic account of visual grounding — and the
substrate for the brain link below.

### E4 — Capacity across the trajectory.
`workspace_capacity`: how many silently-held items are simultaneously decodable.
**Prediction:** capacity (a few dozen concepts in Claude) is set mostly at
pretraining scale, *not* moved much by post-training — post-training changes
*what* enters and *how it's used*, not *how much* fits. If post-training *does*
move capacity, that reframes capacity as trainable, which is bigger news.

---

## The brain bridge (E5, the ambitious extension)

The existing brain–LLM literature (Caucheteux & King 2022; Schrimpf et al.;
Goldstein et al. 2022; Antonello et al. 2023; Tang et al. 2023) correlates the
**full** residual stream with fMRI/ECoG. The workspace lets us test a **double
dissociation** predicted by Global Neuronal Workspace theory (Dehaene & Naccache,
whose invited commentary accompanies the Anthropic paper):

> **H-brain:** the LLM's *J-space* content aligns preferentially with the brain's
> **fronto-parietal workspace** (and the ignition / P3b signature), while the LLM's
> *non-J-space* automatic activations align with **sensory/superior-temporal
> language cortex**. The automatic-vs-workspace split *inside* the model mirrors
> the automatic-vs-conscious-access split in the brain.

Two things make this more than analogy:
- **Depth ↔ time.** The LLM workspace evolves over network **depth** in one
  forward pass; the brain workspace evolves over **time** (recurrence). So the
  natural alignment target is *LLM layer-trajectory ↔ brain time-trajectory* —
  align the sequence of concepts entering J-space across layers to the sequence
  of decodable concepts across time (King & Dehaene temporal-generalization).
  This is exactly the "activation propagation similarity" intuition, made
  operational. **Why it should hold:** CoT reasoning data is distilled from
  humans externalizing step-by-step reasoning, so the model learns to *serialize*
  reasoning the way humans do — and the workspace is where that serialization
  lives internally. Testable corollary: the layer↔time trajectory match is
  *stronger for CoT-trained checkpoints* than for non-CoT ones (ties E5 back to
  the trajectory axis).
- **VLM × fMRI on shared images.** The Natural Scenes Dataset (NSD) shows natural
  images to fMRI subjects at scale. Show the *same* images to a VLM, read J-space
  at image tokens (E3), and test whether workspace content predicts NSD
  higher-visual/parietal responses better than raw VLM features do — a concrete,
  runnable fusion of "VLM J-lens × fMRI."

*Kill criterion:* if J-space aligns with cortex no better than random subspaces of
matched dimension, the workspace–brain link is analogy only — worth reporting as
a negative that disciplines the hype.

---

## Resources that make it feasible on open weights

| Axis | Checkpoints |
|---|---|
| base → SFT → DPO → RLVR (one phase at a time) | **OLMo-2 / Tulu-3** (allenai) — public per-stage checkpoints; RLVR is a GRPO-style RL-with-verifiable-rewards stage |
| pretraining-time trajectory | **Pythia**, **OLMo** intermediate steps |
| base vs instruct (quick stand-in) | **Qwen2.5** base vs -Instruct (used in `run_devtraj.py`) |
| GRPO before/after (controlled) | run GRPO yourself via **TRL** on Qwen-0.5B/1.5B with a verifiable-reward task |
| unimodal → multimodal | **LLaVA**, **Prismatic VLMs**, **Qwen2-VL** base LLM vs aligned |
| brain data | **NSD** (images), **Narratives**/**Little Prince**/**Pereira 2018** (language), reasoning-fMRI sets |

---

## What's in this repo

- `jlens.py` — `WorkspaceLens`: residual-stream capture (arch-agnostic), a
  zero-training **logit-lens** readout, and the **future-token tuned lens** (the
  principled dual of the Jacobian lens — surfaces genuinely silent multi-step
  concepts). `token_direction` for concept vectors.
- `patch.py` — `concept_swap` / `swap_effect`: causal concept edits in the
  residual stream (`alpha` = fraction of activation norm) to prove the workspace
  is *used*, not a scoreboard (spider→ant, France→China).
- `metrics.py` — the trajectory metrics: `perspective_index`,
  `reasoning_trace_strength`, `workspace_capacity`.
- `run_demo.py` — end-to-end smoke test on one model.
- `run_devtraj.py` — the base-vs-instruct contrast; swap in OLMo/Tulu for the
  full stage-by-stage study.

```bash
pip install torch transformers accelerate
python run_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --future
python run_devtraj.py   # Qwen2.5-0.5B base vs instruct, or pass --models ...
```

## Suggested paper framing
NeurIPS/ICLR main track (mech-interp + reasoning): *"Post-training is workspace
engineering: a developmental account of reasoning and alignment in LLMs."* The
brain bridge (E5) is a strong second paper or a high-risk final section — spin it
out if E1–E4 land first.
