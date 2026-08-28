# Novelty Check Report (2026-07-10)

Six-axis parallel literature sweep (7 agents, ~200 searches/fetches) + gpt-5.4
cross-model AC verification (trace: `.aris/traces/novelty-check/2026-07-10_run01/`).

## Proposed Method
Gate LLM-agent memory writes by an internal activation "workspace" readout instead of
the model's verbal self-reported importance; verbal report is anti-informative exactly
for silently-inferred content (regime map), the readout rescues that regime, causally
feeds memory answers, matches the selection oracle downstream, and the divergence is
fixed at pretraining (SFT/DPO/RLVR never repair it).

## Core Claims × Novelty
1. Workspace-style residual readout exists / silent inferences visible / causal patching — **LOW (preempted by Anthropic J-lens/J-space 2026)** → cite as foundation.
2. Verbal salience not introspectable — **LOW-MEDIUM** (Trienes et al. 2025, behavioral) → our sharpening: ANTI-informative + regime-split + activation-level.
3. Internal signals beat verbal self-assessment for agent gating — **MEDIUM** (tool-calling probes 2026; robotics surprise-gating 2026) → ours is memory-writes + mechanistic readout.
4. Explicit-vs-inferred regime map with double dissociation (V AND embeddings flip sides) — **HIGH (nothing found)**.
5. Post-training stage analysis of the W/V divergence (SFT/DPO/RLVR) — **HIGH (nothing found; must scope against Anthropic 2026 injected-thought result)**.
6. Budgeted memory selection at oracle-level downstream via readout — **HIGH in this exact form** (A-MAC & multi-factor value model attack same problem with surface features → mandatory baselines).

## Closest Prior Work (top threats)
| Paper | Year | Overlap | Delta |
|---|---|---|---|
| Anthropic J-lens / J-space (transformer-circuits) | 2026 | HIGH | no memory application, no W-vs-V dissociation, no regime map, no post-training; their "verbalizable" framing must be reconciled with our unreportability result |
| Trienes et al., Behavioral Analysis of Information Salience | 2025 | HIGH | behavioral (summarization), no activations, no anti-informativeness, no gating |
| Latent Introspection (concept injection) | 2026 | MED | injected concepts, not natural inferences; no utility/memory |
| Agent-memory circuit analysis (mem0/A-MEM) | 2026 | MED | diagnosis, not write-gating |
| A-MAC memory admission control | 2026 | MED | same problem framing, surface features — mandatory baseline |
| Multi-factor value model for agentic memory | 2026 | MED | replaces verbal 1-10 importance, no internals — mandatory baseline |
| Surprise-gated robot episodic memory (V-JEPA-2) | 2026 | MED | internal-signal gating, robotics/vision |
| Hidden-state tool-call gating | 2026 | MED | adjacent agent decision, trained probes |
| Orgad et al., LLMs Know More Than They Show | 2024 | MED | truthfulness domain anchor |

## Overall Assessment (gpt-5.4, reasoning=high)
- **Score: 6.5/10 — PROCEED WITH CAUTION.**
- Not a new-mechanism paper (the readout construct is J-lens's); a **strong
  application/diagnostic paper** if framed narrowly.
- Defensible one-liner: *"We introduce the first workspace-readout-based write gate
  for LLM agent memory and show its benefit concentrates in a previously undocumented
  regime where verbal importance judgments are anti-calibrated: naturally evoked
  silent inferences."*
- Kill-requirements: (a) surface-cue control airtight (done: confound checks + v4
  construct); (b) mandatory baselines A-MAC / multi-factor value / surprise-entropy
  family / trained generic probe; (c) do NOT overclaim "post-training never improves
  introspection" (Anthropic 2026 injected-thought detection is a counterexample in an
  adjacent construct — scope to utility-salience).
- What upgrades it beyond 6.5: a readout that is OURS (see THEORY.md — the
  availability family + dynamic members), turning the v3 two-hop boundary from
  limitation into theory-predicted structure.
