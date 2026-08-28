# Paper Plan (rev. 2 after gpt-5.4 review)

**Title**: Remember What You Thought, Not What You Said: Workspace-Gated Episodic Memory for LLMs
**One-sentence contribution**: We provide the first evidence that a J-lens-style workspace readout outperforms the model's own verbal importance report for memory write admission — precisely in a previously undocumented regime (naturally evoked silent inferences) where verbal report is anti-calibrated at every scale and post-training stage tested — and show a 500-step metacognitive-alignment patch repairs the reporter.
**Novelty scoping (mandatory)**: the readout construct is J-lens-derived (Anthropic 2026, cited as foundation); our novelty = admission control + regime map + verbal anti-calibration + repair.
**Venue**: ICLR (9-page main body, unlimited appendix)
**Type**: Empirical / diagnostic + application
**Date**: 2026-07-10
**Page budget**: 9 pages main; appendix exhaustive
**Section count**: 6

## Claims-Evidence Matrix

| # | Claim | Evidence (files) | Status | Section |
|---|---|---|---|---|
| C1 | Regime-dependent dissociation: verbal importance is anti-informative for silently-inferred info (W−V>0, all sizes) but beats the workspace for explicit info (V−W>0 at 1.5–7B); embeddings flip sides the same way | results_v4f/v3f/v2f/v1f_*, master_table.md (cluster CI + Bonferroni), 3 families, 2 generators | Supported | §4.1 |
| C2 | Workspace positive oracle on evoked bridges, scaling 0.53→0.64 (7B CI>0.5), with a sharp boundary: compositional bridges absent from the static state (v3≈0.5 under logit-lens AND trained future-lens) | results_v4f/v2f grids; FutureLens negative control (results_fl_*) | Supported w/ boundary | §4.2 |
| C3 | Availability is pretraining-fixed; post-training installs a confident reporter, never a monitor (0/11 stage transitions improve W; base V_raw already miscalibrated; chat-V worsens) | results_v2f/v3f_olmo1b-{base,sft,dpo,rlvr}, 8 Qwen pairs, V_raw | Supported | §4.3 |
| C4 | Budgeted memory payoff: workspace-gating ≫ verbal at 0.5B (p≤0.0015), = selection oracle at 7B on the valid instrument (p=0.0013); boundary: recall-time composition binds at 3B | downstream_v2f_*, downstream_v4x*/v4xl_*, McNemar per-episode records | Supported w/ boundary | §4.4 |
| C5 | Causal: memory answers READ the workspace (independent pairs, 10× direction-specific flip, McNemar p≤0.0063, Wilcoxon p≤0.0002) | causal_ind_*.json | Supported | §4.5 |
| C6 | Metacognitive alignment: distilling W into V (500 steps, 1B) repairs the reporter on held-out batteries (+0.31–0.35 AUC) with no harm to W or QA | results_v4f/v3f_olmo1b-metacog vs -rlvr; downstream no-harm checks | Supported | §5.2 |
| C7 | Availability-family double dissociation: decode members own the evoked regime (W_rep 0.684), the gradient member owns the compositional regime (PUL v3@7B 0.639, paired +0.125 CI>0); family-dependent (OLMo PUL null) | results_rep_*/pul_*/ig_* grids | Supported w/ boundary | §5.1 |
| C8 | Perception behaves like the explicit regime (VLM: V=0.986 with image, W within-class = chance) | results_vlm3*.json double ablation | Supported (negative/boundary) | §5.3 (brief) + App. H |

## Structure

### §0 Abstract (~200 words)
- What: agent memory write-gating from an internal workspace readout instead of verbal self-report.
- Why hard/matters: agent stacks universally use model-rated importance (Generative-Agents style); we show it is ANTI-informative exactly where memory matters (inferred, never-written info).
- How: logit-lens-family availability readouts + four batteries isolating regimes + causal patching + budgeted QA.
- Evidence: 3 model families, 2 generators, cluster-bootstrap CIs; machine metacognitive efficiency M_s < 0 in the inferential regime at every scale/stage.
- Most remarkable: verbal report sign-flips (AUC 0.17–0.44) where the workspace reads 0.60–0.68; a 20-minute alignment patch lifts held-out V-AUC 0.33→0.67 without touching the workspace.

### §1 Introduction (1.5 pp)
- Hook: agents decide what to remember by asking themselves what is important — the assumption behind reflection-based memory since Generative Agents.
- Gap: nobody has tested WHERE self-rated importance fails; prior work (Trienes) shows salience is not introspectable behaviorally, but not the anti-calibration, its regime structure, or a mechanistic alternative.
- One-sentence contribution (above). Approach overview: four content-provenance batteries (explicit / evoked one-hop / evoked+decoupled / compositional), two channels per item (workspace readout W, verbal report V), oracle-labeled utility, budgeted recall QA, causal patching, post-training stage sweep.
- Contributions (4 bullets): (i) regime map + machine metacognitive efficiency formalism (M_s), first anti-calibration result; (ii) first workspace write gate for agent memory with oracle-level budgeted payoff + causal grounding; (iii) developmental result: pretraining-fixed availability, reporter-not-monitor post-training + metacognitive alignment patch; (iv) availability-family theory with a static/dynamic double dissociation (PUL reads compositional bridges the static state provably lacks).
- Hero figure (Fig 1) below.
- Key citations: Park et al. 2023; Anthropic J-lens 2026 (foundation, not strawman); Trienes 2025; Orgad 2024; Fleming & Lau 2014.
- Front-loading check: Table 1 referenced from intro; strongest numbers in ¶2.

### §2 Related Work (1 p)
- (a) Agent memory & admission control: MemGPT, Generative Agents, EM-LLM, SirLLM, Titans, A-MAC, multi-factor value model, CMI, memory-circuit analysis — none gate writes from internal salience; A-MAC/value-model are mandatory empirical anchors.
- (b) Reading internal states: logit lens, tuned lens, Future Lens, patchscopes, RepE, J-lens/J-space (our construct's foundation; their "verbalizable workspace" vs our unreportability must be reconciled explicitly).
- (c) Introspection & self-knowledge: Trienes salience; latent-introspection injection; CoT unfaithfulness; Orgad internals-beat-outputs (truthfulness) — we extend to utility with regime structure.
- (d) Machine metacognition & cognitive grounding: Fleming–Lau meta-d′; JOL/Koriat cue-utilization; GNW; blindsight; split-brain interpreter — mapped w/ tightness grades (App. §M).

### §3 Setup: Batteries, Channels, and Metacognitive Efficiency (1.5 pp)
- Notation: episode e=(context, items, probe, answer); labels load_bearing/distractor/filler; provenance regimes R∈{explicit, evoked, evoked+decoupled, compositional}.
- Battery design rules incl. v4 validity constraints (bridge silent; answer ∉ items ∪ context; anchor-free probe); generation + independent validation; dedup.
- Channels: W_rr (case-variant reciprocal-rank logit lens at END state, max over layers); V / V_raw (yes/no importance probes); embedding & surface baselines; confound controls (appearance AUC=0.500).
- Machine metacognitive efficiency: M_s=(AUC(V)−0.5)/(AUC(W)−0.5); relation to meta-d′/d′; human M≈0.8 reference.
- Downstream protocol: budget-k recall QA, policies, refs, exact McNemar on per-episode records.

### §4 The Regime Map (3 pp) — results concentrated here
- 4.1 Dissociation (Table 1 master + Fig 2): W−V per regime × size × family; Bonferroni cluster CIs; embeddings flip sides (v1f: embed>W p≤0.006; v3: embed at 0.402); M_s row (−8.4…−0.10 inferential; >1 explicit).
- 4.2 Boundary of the static state: v3≈0.5; FutureLens negative control (−0.088 vs logit-lens at 7B v2f; never rescues v3) → not a readout artifact.
- 4.3 Development (Table in App. D, headline numbers inline): 0/11 transitions improve W (two degrade, OLMo base→RLVR −0.052 P=0.049); base V_raw 0.259–0.445 already broken; chat-V 0.27–0.30.
- 4.4 Payoff (Fig 3 bars + McNemar inline): v2f 0.5B +20/−4 p=0.0015, +24/−2 p<1e-4; v4x 7B workspace@3=0.529=oracle, k=3 +26/−7 p=0.0013; verbal below no-memory floor at 0.5B; 3B composition boundary stated honestly.
- 4.5 Causality (Table 3): independent pairs, real vs sham flips, McNemar + Wilcoxon; tangent/secant link forward-ref to §5.1.

### §5 The Availability Family, Alignment, and Perception (1.5 pp)
- 5.1 Family (Table 4 compact): Spotlight/Broadcast/Rehearsal/Leverage definitions (one line each + brain analog); double dissociation (PUL v3@7B 0.639 paired +0.125 CI>0; PUL weak on evoked; W_rep 0.684 on evoked, ceiling 0.55 on compositional budget-independent); OLMo-PUL null stated; naive ensembles don't win → regime-aware selection.
- 5.2 Metacognitive alignment (Table 2): 0.326→0.671 / 0.267→0.580 held-out; W untouched; QA no-harm; post-alignment V(v3)>static W(v3) — a calibrated report is itself a dynamic computation.
- 5.3 Perception boundary (2 sentences + App. H): image identity absent from word workspace (within-class 0.483≈no-image) while ask-time V=0.986 — perceptually-present ⇒ explicit regime.

### §6 Conclusion & Limitations (0.5 pp)
- Limitations (explicit list): single-generator batteries per design (mitigated: gpt-4o replication of dissociation, not oracle level); PUL family-dependence; 3B C4 gap; replay ceiling; multimodal negative at 2B; benchmark scale (52–120 episodes/battery); Anthropic-2026 scoping of post-training claims.
- Future: regime-aware member selection; metacognitive alignment at scale; A-MAC-style feature fusion.

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero (figure-spec SVG) | Two-panel: (L) pipeline — context → [workspace state → W readout] vs [verbal probe → V report] → budget gate → recall QA, with brain-analog icons (spotlight/broadcast/leverage/rehearsal strip); (R) regime map 2×2 — provenance (explicit↔inferred) × channel (W/V), cells colored by AUC with the sign-flip cell highlighted. Caption states the comparison in one sentence. | manual spec (figures/specs/hero.json) | HIGH |
| Fig 2 | Grouped line/dot plot | W vs V (and embed) AUC with cluster CIs across sizes (0.5–7B) for v1f/v2f/v4/v3 — the regime map quantitatively; one panel per battery | results_*_{size}-Instruct.json | HIGH |
| Fig 3 | Grouped bars | Budgeted QA accuracy k=1..3: workspace/verbal/embedding/oracle/floor at 0.5B-v2f and 7B-v4x | downstream_v2f_0.5B, downstream_v4x_7B | HIGH |
| Table 1 | Master regime×channel | 7B rows for 3 families × 4 batteries: W_rr, V, W−V cluster CI, M_s | master_table.md + results_* | HIGH |
| Table 2 | Metacog alignment | before/after V & W on held-out v4/v3d + no-harm row | results_*olmo1b-{rlvr,metacog} | HIGH |
| Table 3 | Causal | 3 runs: flip/sham/McNemar/Wilcoxon | causal_ind_*.json | HIGH |
| Table 4 | Family compact | 4 members × 3 regimes AUC (7B) with winner bolded | results_{rep,pul,ig}_* | HIGH |
| App. tables A1–A14 | Exhaustive | full AUC grids (all 10+ checkpoints × 4 batteries × {W_rr,V,V_raw,embed}); downstream × 12 cells full; McNemar full; OLMo stages; FutureLens; family grids incl. sub-components; causal incl. scale sweep + sliding-window comparison; multimodal 3 batteries; g2 + Mistral replications; battery stats; hyperparams; prereg deviations; audit summary | all results_*.json | HIGH |

## Citation Plan (verified via this project's live searches; [VERIFY] = re-check at bib time)
- §1: park2023generative; anthropic2026workspace (transformer-circuits.pub/2026/workspace); trienes2025salience (2502.14613); orgad2024llmsknow (2410.02707); fleming2014how.
- §2a: packer2023memgpt; wang2024emllm (2407.09450); yao2024sirllm (2405.12528); behrouz2025titans (2501.00663); amac2026 (2603.04549) [VERIFY]; valuemodel2026 (2606.12945) [VERIFY]; cmi2026 (2605.17641) [VERIFY]; memcircuits2026 (2605.03354) [VERIFY]; surpriserobot2026 (2606.03787) [VERIFY]; toolcalling2026 (2605.00737) [VERIFY].
- §2b: nostalgebraist2020logitlens; belrose2023tuned; pal2023futurelens; ghandeharioun2024patchscopes; zou2023repe; latentintrospection2026 (2602.20031) [VERIFY].
- §2c: turpin2023cot; lindsey2025introspection [VERIFY].
- §2d: nelson1990metamemory; koriat1997jol; dehaene2011gnw; weiskrantz1986blindsight; gazzaniga2000splitbrain; frey1997tagging; wilson1994replay; vonrestorff1933.

## Reviewer Feedback (gpt-5.4, applied — full text in .aris/traces/paper-plan/)
Scores: flow 7, claim-evidence 8, missing-exp 6, positioning 5, page-budget 4, front-matter 7.
APPLIED CHANGES:
1. **Repositioning vs J-lens (mandatory)**: readout construct = J-lens-derived foundation;
   novelty = admission control + regime map + anti-calibration + repair. All "first
   workspace readout" phrasing replaced.
2. **Main-body scope cut to C1/C2/C4/C6**: §4 = dissociation (4.1), static-state boundary
   (4.2), payoff (4.3), causality COMPACT (4.4, table + 4 sentences); C3 development
   compressed to one paragraph in 4.2 with headline numbers (full tables → App. D);
   §5 = metacognitive alignment ONLY (C6); availability family (C7) → 3 sentences in
   §5 + App. F; perception (C8) → 1 sentence in limitations + App. H.
3. **Related work 0.5 p** (cognitive-grounding map → App. M); Setup trimmed (M_s
   derivation, battery validity rules detail, confound detail → App. A/K).
4. **Claim softening**: "pretraining-fixed" → "in our stage sweep, none of 11
   transitions improved W"; "absent from static state" → "not recoverable by the
   tested static-state readouts"; "= oracle" → "matches the selection-oracle accuracy
   on this benchmark".
5. **Abstract rewritten to 5-beat plain structure** (practice → failure → alternative →
   payoff → repair); M_s moved out of abstract.
6. **Hero figure simplified**: (L) write-gating pipeline, (R) 2×2 regime×channel heatmap
   with the sign-flip cell highlighted; member-icon strip → Table 4/App.
7. **Two added experiments (reviewer-required, run before Phase 2)**:
   (a) surface-fusion downstream baseline (rank-average of recency+frequency+length+
   embedding+surprisal — A-MAC/value-model proxy) on v2f-0.5B and v4x-7B;
   (b) V prompt-robustness: 3 paraphrases of the importance probe on v4-7B, show
   anti-calibration is not one bad prompt (report per-paraphrase AUC + ensemble).

## Next Steps
- [ ] /paper-figure
- [ ] /paper-write
- [ ] /paper-compile
- [ ] /auto-paper-improvement-loop
