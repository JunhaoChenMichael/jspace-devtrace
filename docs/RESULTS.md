# Pilot results — "Remember what you thought, not what you said"

**Question (go/no-go):** does an encoding-time *workspace* signal (`W`, latent) predict
which memory items are truly useful better than the model's own *verbal reflection*
(`V`, "is X important?")? Both read blind to the probe; ground truth = load-bearing label.

Models: GPT-2 (124M, pretrained), Qwen2.5-0.5B (base), Qwen2.5-0.5B-Instruct.
Batteries: **v1** (30 ep / 164 items, concepts mostly appear literally), **v2 silent-bridge**
(31 ep / 165 items, load-bearing concept is an inference that never appears in text;
distractors are vivid words that do appear). AUC of each signal at recovering
load_bearing; 95% CIs are item-level bootstrap (2000×).

## Headline numbers (AUC, load_bearing vs rest)

| Battery | Model | W (workspace) | V (verbal reflection) | W − V (95% CI) |
|---|---|---|---|---|
| v1 mixed | 0.5B-base | 0.591 [0.495, 0.688] | — | — |
| v1 mixed | 0.5B-instruct | 0.604 [0.504, 0.705] | 0.535 [0.443, 0.631] | +0.069 [−0.077, +0.217] (n.s.) |
| v2 silent | 0.5B-base | 0.524 [0.402, 0.643] | — | — |
| v2 silent | 0.5B-instruct | 0.524 [0.400, 0.647] | **0.307 [0.191, 0.426]** | **+0.217 [+0.060, +0.372] (sig.)** |
| v1 mixed | gpt2 | 0.605 [0.507, 0.700] | — | — |

Disagreement (v2, instruct), items where W and V most disagree:
workspace-favored → P(load_bearing) = **0.333**; reflection-favored → **0.091** (base rate ≈0.19).

## Findings

1. **The core dissociation is real and significant — in the regime that matters.**
   On silent-bridge items, verbal reflection is *actively anti-correlated* with true
   utility (AUC 0.31, CI well below 0.5): asked "what's important?", the model points
   at the vivid present distractors and misses the never-written bridge. The workspace
   is not fooled the same way (0.52). W − V = **+0.22, CI excludes 0**. And on
   disagreements the workspace is right 3.6× more often. → strong support for
   "what the model *says* is important is misleading exactly when the useful info is a
   silent inference."

2. **Not a surface artifact.** In v1, 98% of all concepts appear literally, so a
   "did-the-word-appear" predictor scores AUC 0.500; W still scores 0.60. v2 removes
   the confound by construction (load-bearing never appears).

3. **Workspace as a strong *positive* oracle is NOT yet shown.** W = 0.60 on present
   concepts but drops to ~chance (0.52) on silent bridges — most likely because a 0.5B
   model often fails to actually perform the bridge inference, so the concept is not
   reliably in its workspace to be read. **This is the critical open question a capable
   model (3B/7B) must resolve:** does a model that CAN infer the bridge hold it in the
   workspace (W high on v2)? If yes, the positive-oracle claim lands; if W stays at
   chance, the paper pivots to "verbal reflection is unreliable" (still real, weaker).

4. **Developmental (Pillar 3) reframed.** Base ≈ instruct on the workspace channel in
   BOTH batteries (0.591 vs 0.604; 0.524 vs 0.524). Post-training does NOT make the
   workspace a better utility oracle — utility-salience looks like a *pretraining*
   capability. Post-training's effect shows up in the *verbal* channel (which becomes
   miscalibrated/anti-correlated on silent bridges). The interesting developmental claim
   is therefore not "workspace improves" but "post-training installs a verbal-report
   layer that can diverge from — and be worse than — the workspace." (Connects to
   Anthropic's counterfactual-reflection result: the verbal channel is separable from
   the underlying computation.)

## 3B-Instruct (the capable-model test) — answers the pending question

| Battery | W (workspace) | V (verbal reflection) | W − V (95% CI) |
|---|---|---|---|
| v1 mixed (explicit/salient info) | 0.575 [0.477, 0.671] | **0.625 [0.532, 0.715]** | −0.050 [−0.174, +0.083] (V wins, n.s.) |
| v2 silent-bridge (inferred info) | **0.622 [0.508, 0.730]** | 0.410 [0.297, 0.522] | **+0.211 [+0.083, +0.341], P=1.00** |

Scaling 0.5B → 3B on silent bridges: **W rose 0.52 → 0.62** (a capable model DOES hold the
silent bridge in its workspace), while **V stayed unreliable (0.31 → 0.41)**.

**The sharpened finding — the dissociation is REGIME-DEPENDENT:**
- When the useful info is **explicit/salient** (v1), verbal reflection is fine and even beats
  the workspace (0.625 vs 0.575) — the model can just read importance off the surface.
- When the useful info is an **implicit inference** (v2 silent bridge), verbal reflection
  breaks (0.41, ~chance/below) while the workspace holds it (0.62) and wins significantly.

So the precise, defensible claim is not "workspace always beats self-report" but:
**self-report tracks salience fine for explicit information and FAILS for inferred
information; the workspace covers exactly that gap.** That tells you *when* to trust the
workspace over self-report — when the relevant info is implicit.

## Developmental (base vs instruct) — replicated at 0.5B AND 3B

Workspace W_rr AUC, base vs instruct:

| Battery | 0.5B base → instruct | 3B base → instruct |
|---|---|---|
| v1 mixed | 0.591 → 0.604 | 0.606 → 0.575 |
| v2 silent | 0.524 → 0.524 | 0.609 → 0.622 |

The workspace's utility-tracking is **the same in base and instruct** across both sizes and
both batteries (no consistent gain from post-training). The verbal channel V exists only in
instruct. So the developmental claim is sharp and replicated:

**"What matters" salience is a PRETRAINING capability, stable in the base model's workspace.
Post-training does not improve it — instead it installs a VERBAL self-report layer that is
reliable for explicit info but miscalibrated (anti-informative) for inferred info.** (Ties to
Anthropic's counterfactual-reflection: the verbal channel is separable from the underlying
computation.)

## Verdict — coherent, replicated story across 0.5B, 3B, GPT-2, two batteries

1. Regime-dependent dissociation (core, replicated): self-report tracks EXPLICIT salience
   fine (v1: V≈0.63, beats workspace) but FAILS for INFERRED salience (v2: V≈0.31–0.41,
   ~chance/below), where the workspace holds it (v2 3B: W=0.62) and wins significantly (P=1.00).
2. Positive-oracle scales with capability: silent-bridge W = 0.52 (0.5B) → 0.62 (3B).
3. Developmental: workspace utility-tracking is pretraining-level (base≈instruct, replicated);
   post-training adds the (regime-dependently miscalibrated) verbal layer.
4. Not a surface artifact (v1 appearance AUC 0.500; v2 removes the confound by construction).

Remaining: 7B confirmation (downloading); trained future-lens for cleaner W; and the payoff
experiment — re-inject high-W vs high-V memories at recall, measure downstream QA accuracy.

---

# Server campaign (2026-07-09, 8× RTX A5000, CUDA, bfloat16)

Everything below was run on the GPU server after migration. Note: downstream numbers for
0.5B-v2 shift slightly vs the MPS run (greedy decode differs across hardware/dtype);
CUDA numbers are now canonical. The MPS original is kept at
`downstream_Qwen2.5-0.5B-Instruct_v2_0.5B-Instruct.json.bak-mps`.

## 7B measurement (HANDOFF TODO 1) — verbal reflection CATCHES UP at 7B

| Battery | Model | W_rr [95% CI] | V [95% CI] | W − V [95% CI], P(W>V) |
|---|---|---|---|---|
| v2 silent | 7B base | 0.643 [0.530, 0.759] | — | — |
| v2 silent | 7B instruct | 0.643 [0.531, 0.753] | 0.603 [0.484, 0.721] | +0.040 [−0.102, +0.180], P=0.71 (n.s.) |
| v1 mixed | 7B base | 0.588 [0.491, 0.682] | — | — |
| v1 mixed | 7B instruct | 0.556 [0.463, 0.651] | **0.695 [0.609, 0.774]** | −0.119 [−0.248, +0.017], P(W>V)=0.04 (V wins) |

Late addition: **gpt2 v2** (was measured pre-migration, never analyzed): W_rr = 0.580
[0.452, 0.717] (W_end 0.629). Confound check at 7B v1: literal-appearance AUC = 0.500,
W_rr on appearing-only subset = 0.564 — still not a surface artifact.

**The full scaling picture on silent bridges (v2), AUC:**

| signal | gpt2 | 0.5B | 3B | 7B |
|---|---|---|---|---|
| W_rr (workspace) | 0.58 | 0.52 | 0.62 | **0.64** |
| V (verbal, instruct) | — | 0.31 | 0.41 | 0.60 |

Two monotone trends: (i) the workspace positive-oracle signal rises with capability
(C2 confirmed, 0.52 → 0.62 → 0.64; base ≈ instruct at every size, C3 confirmed at 7B:
0.643 vs 0.643). (ii) **the verbal-report miscalibration is a small-model pathology that
closes with scale** — V rises from actively anti-informative (0.31) to near-workspace
(0.60) at 7B, and the W−V dissociation is significant at 0.5B (+0.22) and 3B (+0.21)
but not at 7B (+0.04, n.s.). On explicit info (v1) V beats W at every size that has V.

**Revised C1 (sharper, still falsifiable):** self-reported importance fails for inferred
information *below the capability threshold where the model can articulate the inference*
(≤3B here); the workspace signal is reliable across ALL scales — including base models,
which have no usable verbal channel at all. The pitch shifts from "self-report is always
wrong" to "workspace-gating is scale-robust; reflection-gating is only safe at scales
most memory-constrained agents don't run at."

## Downstream memory-budgeted QA (HANDOFF TODO 2) — complete matrix

Upgraded `downstream.py`: per-episode correctness logged for every condition, exact
McNemar (paired, same episodes) for workspace vs each rival, plus two hostile-reviewer
baselines: **embedding** (bge-small cosine(concept, context) — "just use RAG relevance")
and **recency** (last literal mention; absent → last).

Accuracy on v2 SILENT battery (31 episodes; k = memory budget):

| policy | 0.5B k=1/2/3 | 3B k=1/2/3 | 7B k=1/2/3 |
|---|---|---|---|
| **workspace** | .32 / .48 / .58 | .26 / .45 / **.74** | .45 / .61 / **.90** |
| verbal | .26 / .26 / .42 | .19 / .42 / .52 | .45 / .71 / .81 |
| embedding | .48 / .55 / .65 | .35 / .45 / .61 | .35 / .65 / .77 |
| recency | .19 / .16 / .39 | .10 / .16 / .23 | .16 / .16 / .35 |
| random (3 seeds) | .31 / .45 / .63 | .22 / .31 / .52 | .28 / .48 / .70 |
| oracle | .97 / .84 / .74 | .71 / .84 / .77 | .97 / .97 / .97 |
| ref: full_context | .48 | .71 | .71 |
| ref: no_memory | .32 | .10 | .23 |

Key McNemar results (workspace vs verbal, v2): 0.5B k=2 +10/−3 p=0.09; 3B k=3 +10/−3
p=0.09; 7B n.s. (even). Workspace vs recency is significant almost everywhere
(e.g. 3B k=3 +16/−0 p<0.0001, 7B k=2 +14/−0 p=0.0001).

On v1 (explicit) at every size: workspace ≈ verbal ≈ embedding (all McNemar n.s.) —
the regime contrast replicates downstream at 0.5B, 3B AND 7B.

Reading the matrix:
1. **The payoff scales:** workspace-gated memory at k=3 reaches 0.74 (3B) and 0.90 (7B),
   at 7B nearly closing the gap to the selection oracle (0.97) and far above the
   full-context ceiling (0.71) — selecting the right 3 concepts beats re-reading the
   whole passage, because the workspace already did the inference at encoding time.
2. **Verbal-gating is unreliable exactly at small scale:** below the no-memory floor at
   0.5B (k≤2), clearly behind workspace at 3B (0.52 vs 0.74 at k=3), catching up only
   at 7B — mirroring the predictive (AUC) picture.
3. **The embedding baseline is the serious rival, and capability decides:** it wins at
   0.5B (the model can't hold the bridge in its workspace, so cheap semantic relevance
   is better), but LOSES to the workspace at 3B (0.61 vs 0.74) and 7B (0.77 vs 0.90) at
   k=3. Workspace-gating pays off precisely when the agent is capable enough to infer.
4. **Recency is dispatched** (significantly worse than workspace nearly everywhere) —
   the effect is not "keep the recent stuff."

Caveat: n=31 episodes → McNemar power is low (the 0.09s); the scaled 500-item batteries
(TODO 3, running) exist to settle exactly this.

---

# Adversarial claim audit + fixes (2026-07-09, later the same day)

A 9-agent audit (one auditor per claim C1–C4, one adversarial refuter each, one hostile
reviewer) recomputed every number from the raw files. All four audit verdicts survived
refutation: **C1 supported; C2/C3/C4 partially supported.** Every published number
reproduced exactly. The audit found two real bugs and three design gaps, all fixed:

1. **Random-baseline contamination (fixed, all downstream re-run):** `rank()`'s
   hash-shuffle was an IDENTITY permutation at seed=0, and batteries list load_bearing
   items first (v1 mean position 1.02) — so 1 of 3 "random" seeds was a semi-oracle.
   With the fix (`random.Random(seed).shuffle`), random@1 drops from 0.18–0.21 to 0.04
   (v1 3B/7B); all policy numbers unchanged (greedy decode is deterministic).
   Workspace-vs-random gaps WIDEN everywhere.
2. **v2 downstream construct validity (fixed via new battery):** in ALL v2 episodes the
   load_bearing concept IS the gold answer verbatim → memory-budgeted QA partly reduces
   to "did the policy keep the answer token". New `gen_battery_v3.py` decouples them:
   the probe needs bridge + general knowledge composition (context "Plaza Mayor" →
   silent bridge "madrid" → answer "spanish"); answer never among stored concepts,
   never in context. `battery_v3d.json`: 52 eps / 261 items / 52 decoupled positives.
3. **Case-variant workspace readout (adopted):** batteries generate lowercase concepts,
   but proper-noun bridges live in the vocab capitalized — ' italy' tokenizes to junk
   (' it') while the workspace holds ' Italy'. W_rr now scores max over
   lowercase/capitalized first tokens (uniform across items → AUC comparison fair).
   Effect: v2 pilot 7B-I W_rr 0.643 → **0.680 [0.572, 0.779]**.
4. **V_raw (template-free verbal probe) added:** C3's "post-training installs the verbal
   layer" had never been tested — base checkpoints were never verbally probed. All
   final-campaign runs now compute V_raw for every model (incl. base + gpt2).
5. **1.5B midpoint added** (C2 had only 3 sizes, non-monotone with gpt2).

## v3 (decoupled silent bridge) — first measurement, old readout, n=261

| model (instruct) | W_rr | V | W − V |
|---|---|---|---|
| 0.5B | 0.409 | 0.172 | +0.237 [+0.115, +0.361] P=1.00 |
| 1.5B | 0.498 | 0.290 | +0.208 [+0.081, +0.336] P=1.00 |
| 3B | 0.416 | 0.197 | +0.219 [+0.123, +0.311] P=1.00 |
| 7B | 0.491 | 0.344 | **+0.146 [+0.033, +0.254] P=0.99** |

Two findings, both important and honest:
- **The workspace positive-oracle has a limit.** W_rr ≈ chance on v3 at every size
  (0.41–0.51; case-variant readout 0.514 at 7B — NOT a tokenization artifact, checked).
  The workspace holds one-hop bridges the narrative actively evokes (v2: 0.68) but not
  bridges needed only for a two-hop composition it hasn't been asked to make.
- **The dissociation survives at 7B on the harder battery.** Verbal reflection collapses
  much further (7B V = 0.344, well below chance — it confidently flags vivid present
  distractors), so W − V is significantly positive at EVERY size including 7B. The 7B
  "verbal catch-up" seen on v2 does not generalize; per the audit, 7B's v2 V-AUC also
  rides on a saturated yes-regime (122/165 items P(yes)>0.99, median 1.000) that no
  thresholding memory policy could actually use.

## Final campaign: batteries after global concept de-dup
- v1_final: merged 100+40 eps → 88 eps / 417 items / 131 positives (v1 explicit)
- v2_final: merged 81+51 eps → **75 eps / 352 items / 75 silent positives** (2.4× pilot)
- v3d: 52 eps / 261 items / 52 decoupled positives
- 9 checkpoints (gpt2, Qwen2.5 {0.5,1.5,3,7}B base+instruct) × 3 batteries, case-variant
  W_rr + V + V_raw; then downstream at 4 instruct sizes × 3 batteries.

## FINAL predictive results (case-variant readout, instruct models)

**v2_final — one-hop silent bridges, n=352 (75 positives):**

| size | W_rr [95% CI] | V [95% CI] | W − V [95% CI] |
|---|---|---|---|
| 0.5B | 0.516 [0.442, 0.588] | 0.295 [0.226, 0.369] | **+0.221 [+0.131, +0.311] sig** |
| 1.5B | 0.579 [0.506, 0.653] | 0.522 [0.443, 0.604] | +0.057 [−0.045, +0.159] n.s. |
| 3B | 0.579 [0.510, 0.644] | 0.461 [0.378, 0.543] | **+0.117 [+0.017, +0.211] sig** |
| 7B | **0.633 [0.566, 0.694]** | 0.579 [0.497, 0.663] | +0.054 [−0.043, +0.150] n.s. |

With 75 positives, the workspace positive-oracle on one-hop bridges is now SOLID at 7B
(CI lower bound 0.566 ≫ 0.5) and rises with scale (0.52 → 0.58 → 0.58 → 0.63).

**v3_final — decoupled two-hop bridges, n=261 (52 positives):**

| size | W_rr [95% CI] | V [95% CI] | W − V [95% CI] |
|---|---|---|---|
| 0.5B | 0.416 [0.332, 0.504] | 0.172 [0.112, 0.241] | **+0.244 [+0.140, +0.344] P=1.00** |
| 1.5B | 0.549 [0.456, 0.644] | 0.290 [0.205, 0.379] | **+0.259 [+0.142, +0.374] P=1.00** |
| 3B | 0.498 [0.414, 0.580] | 0.197 [0.131, 0.269] | **+0.301 [+0.201, +0.390] P=1.00** |
| 7B | 0.514 [0.419, 0.603] | 0.344 [0.249, 0.444] | **+0.170 [+0.057, +0.282] P=1.00** |

On the construct-valid battery the dissociation is significant at EVERY size including
7B — but note W itself is ~chance here; the gap is carried by V being profoundly
anti-informative (CI upper bounds ≤ 0.44 at 0.5B/3B). See hostile-review verdict below.

## Hostile-review verdict (what the paper can and cannot claim)

- **Validated (powered, audit-proof): the negative half of the thesis.** Verbal
  self-report of importance is anti-informative for inferred information — across 3
  batteries, 4 sizes, both prompt formats (V, V_raw), base and instruct, surviving
  episode-cluster bootstraps. And it does NOT recover with scale on construct-valid
  tests (7B v3: V=0.344). The v2 "7B verbal catch-up" was an artifact of the
  answer-coupled battery + a saturated yes-regime (122/165 items P(yes)>0.99).
- **Validated: the regime contrast.** On explicit info verbal reflection WINS
  (v1 7B: V−W_rr = +0.139 [+0.005, +0.266], P=0.019, consistent-signal recompute).
  Self-report is fine when importance is on the surface; it breaks exactly when the
  useful thing was inferred.
- **Half-open: the positive half.** The workspace readout finds one-hop bridges the
  context actively evokes (v2f 7B: 0.633) but NOT bridges needed only for a two-hop
  composition (v3: ~0.50 at all sizes). The logit-lens readout — not statistical
  power — is now the binding constraint. **The decisive next experiment is the trained
  FutureLens readout (HANDOFF TODO 4) evaluated on v3-style batteries**; if it recovers
  the decoupled bridges, C2/C4 land on construct-valid ground; if not, the paper
  re-frames around the (already-proven) unreliability of introspective salience plus
  the one-hop positive result.
- **Open until the final downstream lands (running):** workspace-vs-verbal and
  workspace-vs-embedding McNemar significance with n=75/52 episodes; the pilot n=31
  never reached p<0.05 on any single cell.

---

# Complete-all campaign (2026-07-09 evening): every remaining experiment

## Final downstream (n=75 episodes, v2f) — C4 lands at small scale

Workspace-vs-verbal McNemar, v2f: **0.5B k=2: +20/−4 p=0.0015; k=3: +24/−2 p<0.0001**
(workspace@3 0.747 vs verbal 0.453). 1.5B/3B directional (+22/−13, +18/−9) but n.s.;
7B parity. Workspace-vs-recency significant nearly everywhere; workspace-vs-embedding
never significant (n.s. at every size/budget). v3f downstream is INVALID as a payoff
test — its probes carry the surface anchor ("the city with the Plaza Mayor"), so
no-memory answers from the probe alone (7B floor 0.404 ≥ every policy). This motivated
**battery v4: decoupled answer AND anchor-free probe** (probe references the episode
only through a person's name — "What language should Elena have brushed up on?").

## Grading robustness (audit item) — PASS
Strict whole-word re-grading of stored answers changes every policy cell by ≤0.03 and
never reorders policies (only two reference cells move >0.05). The lenient grader is
not driving any claim.

## Embedding predictive AUC (audit gap, first computed)
| results | embed AUC | W_rr AUC | W − embed |
|---|---|---|---|
| v2f 7B-I | 0.553 [0.476, 0.632] | 0.633 | +0.080 [−0.013, +0.169] P=0.95 |
| v3f 7B-I | 0.402 [0.309, 0.495] | 0.514 | +0.112 [−0.005, +0.227] P=0.97 |
| v2f 3B-I | 0.553 | 0.579 | +0.026 n.s. |
Embedding relevance is ~chance on one-hop silent bridges and actively anti-informative
on decoupled ones; the workspace beats it at 7B with P≈0.95–0.97 (marginal).

## OLMo-2-0425-1B post-training stages (C3, cross-family) — the sharpest result

W_rr AUC on v2f: **base 0.631 [0.565, 0.696] → SFT 0.599 → DPO 0.576 → RLVR 0.580.**
The workspace utility signal is HIGHEST in the base model and mildly erodes through
every post-training stage — replicating C3 in a second model family with the actual
SFT→DPO→RLVR pipeline. Notably OLMo-1B base ≈ Qwen-7B (0.631 vs 0.633) on one-hop
bridges. v3f replicates the two-hop collapse cross-family (W ≈ 0.48–0.51 all stages).
Verbal channels: V ≈ chance on v2f at every stage; on v3f the base model's raw verbal
probe scores 0.445 while SFT/DPO/RLVR chat-V scores 0.302/0.277/0.267 — **post-training
does not install verbal salience; it makes the verbal channel more confidently WRONG
on inferred information.**

## Causal swap (TODO 6) — memory reads the workspace, direction-specifically

Steering the residual stream along input-side diff-of-means bridge directions
(A→B) while the model answers from memory, vs sham directions (unrelated C→D),
20 pairs, answer log-odds flip rate:

| model | scale | real flip | sham flip | Δ log-odds real (sham) |
|---|---|---|---|---|
| 3B-I | 0.5 | **0.50** | **0.05** | +21.7 (+10.9) |
| 3B-I | 1.0 | 0.90 | 0.35 | +30.4 (+18.6) |
| 3B-I | 2.0 | 0.95 | 0.45 | +32.6 (+21.4) |
| 0.5B-I | 1.0 | 0.75 | 0.15 | +16.1 (+10.0) |
| 0.5B-I | 2.0 | 0.80 | 0.40 | +18.0 (+13.3) |

The headline is the gentlest intervention: at scale 0.5 the real direction flips the
3B answer in 50% of pairs while sham flips 5% — a **10× direction-specific separation**
(0.5B at scale 1.0: 5×). At larger scales part of the sham effect is generic memory
degradation (any big perturbation shrinks the log-odds margin), but the
direction-specific component (+10–12 log-odds beyond sham at every scale) is the
causal signature: the answer computation READS the patched workspace content.

## FutureLens (TODO 4) — the readout is NOT the missing piece (controlled negative)

Tuned future lens: per-layer probes initialized from the model's own unembedding
(step 0 = logit-lens) and fine-tuned to put softmax mass on a 12-token future window;
trained ONLY on v1_final contexts (fully disjoint from eval), 6 layers, final loss
~2.8–3.0 (uniform = 11.9). AUC vs logit-lens (instruct):

| size | v3 FL (LL) | v2f FL (LL) |
|---|---|---|
| 0.5B | 0.523 (0.416), +0.107 [+0.029,+0.189] P=0.99 | 0.507 (0.516) n.s. |
| 1.5B | 0.585 (0.549) n.s. | 0.562 (0.579) n.s. |
| 3B | 0.525 (0.498) n.s. | 0.576 (0.579) n.s. |
| 7B | 0.563 (0.514) n.s. | 0.545 (0.633), **−0.088 [−0.157,−0.022]** FL worse |

Verdict: the trained readout improves v3 slightly at every size but never rescues it
past ~0.59, and on one-hop bridges the model's own unembedding is already optimal
(FL significantly WORSE at 7B v2f). So the v3 collapse is not a readout artifact:
**the encoding-time workspace holds inferences the narrative evokes, not arbitrary
compositions that only become useful once a future question arrives.** This is the
honest boundary of the positive-oracle claim — and battery v4 shows it is not a
problem for the thesis (below).

## Battery v4 (decoupled answer + anchor-free probe) — every claim lands

v4 = the construct-valid instrument: the bridge is a strongly-evoked one-hop
inference (v2-style), the ANSWER is one more hop away (never among stored concepts,
never in context), and the probe references the episode only through a person's name
(no-memory floor is structurally low). 68 eps / 335 items / 68 positives.

**Predictive (case-variant logit-lens W_rr vs chat V):**

| size | W_rr [95% CI] | V | W − V [95% CI] |
|---|---|---|---|
| 0.5B | 0.526 [0.446, 0.597] | 0.281 | **+0.244 [+0.142, +0.344]** |
| 1.5B | 0.577 [0.502, 0.655] | 0.462 | **+0.115 [+0.001, +0.231]** |
| 3B | 0.594 [0.523, 0.669] | 0.254 | **+0.340 [+0.252, +0.430]** |
| 7B | **0.641 [0.573, 0.708]** | 0.486 | **+0.155 [+0.053, +0.263] P=1.00** |

On the fully construct-valid battery: the workspace positive-oracle holds and scales
(0.53 → 0.58 → 0.59 → 0.64, 7B CI excludes 0.5), verbal reflection stays at/below
chance at every size (no 7B catch-up), and the dissociation is significant everywhere
(1.5B is borderline: CI lower bound +0.001). C1 + C2 validated on this instrument.
Verification notes: a full-lexeme scan finds 2/68 bridges sharing a lexeme with the
context (ep23 'yankees'/'Yankee Stadium', ep34 'stacking'/'stacked') and 1/68 answers
as an inflection ('cycling'/'cycled') — removing them does not change any conclusion;
1 episode's answer is a synonym of its bridge (ep58 'lawyer'/'attorney').

**Downstream on v4** exposed one more boundary, then confirmed the payoff:
with the original 24-token generation budget even the ORACLE collapses
(7B oracle@3 = 0.265 vs full-context 0.618) — at recall time the model cannot
compose bridge + question in so few tokens; when selection cannot matter, no policy
differentiates. With a 64-token budget (`downstream_v4x_*`) the task unblocks at 7B:

| policy (7B, 64 tok) | k=1 | k=2 | k=3 |
|---|---|---|---|
| **workspace** | 0.206 | 0.338 | **0.529** |
| embedding | 0.250 | 0.309 | 0.471 |
| verbal | 0.103 | 0.353 | 0.426 |
| oracle | 0.206 | 0.353 | 0.485 |
| ref: full_context / no_memory | 0.662 / 0.191 | | |

Workspace-gated memory at k=3 reaches the selection-oracle level (0.529 vs 0.485),
2.8× the no-memory floor, 80% of the full-context ceiling; ordering
workspace > embedding > verbal; workspace-vs-verbal k=1 +9/−2 p=0.065 (directional,
n=68). C4's clean significance remains the v2f 0.5B result (p=0.0015, p<0.0001).

# FINAL VERDICT on the four claims (all experiments complete)

*(Every number below survived a 5-agent adversarial verification pass that recomputed
it from raw files; wording follows the verifiers' corrections. A second completeness
pass added the honest qualifications noted inline.)*

- **C1 — regime-dependent dissociation: VALIDATED** (powered, construct-valid,
  cross-family, survives episode-cluster bootstrap). W−V significant at every size on
  v4 (+0.115…+0.340; 1.5B is knife-edge, lower bound +0.001, and n.s. under V_raw
  there) and v3 (+0.170…+0.301); verbal never catches up on valid instruments; on
  explicit info (v1_final, n=417) verbal WINS at 7B (V−W_rr = +0.142 [+0.062, +0.217])
  — the full regime map. Qualification: on the answer-coupled v2f battery at 7B, k=1
  downstream, verbal significantly beats workspace (+6/−18, p=0.023) — the coupled
  instrument flips sides at 7B; the valid instrument (v4x) does not.
- **C2 — workspace positive oracle, scaling: VALIDATED with a sharp boundary.**
  Evoked one-hop bridges: v4 0.53→0.58→0.59→0.64 (7B CI [0.573, 0.708]); base series
  0.538→0.582→0.616→0.621. Boundary (FutureLens-controlled): the encoding-time
  workspace does NOT precompute compositions that only a future question makes
  relevant (v3 ~chance under logit-lens AND trained future lens).
- **C3 — developmental origin: VALIDATED in two families, with a corrected verbal
  half.** Workspace: zero of 8 Qwen paired base/instruct diffs and zero of 3 OLMo
  stage transitions show significant improvement; two show significant DEGRADATION
  (Qwen v2f 3B −0.037 [−0.065, −0.011]; OLMo SFT→DPO −0.023), cumulative OLMo
  base→RLVR −0.052 (P=0.049). Post-training never improves the workspace channel.
  Verbal: the miscalibration on inferred info exists ALREADY in base models probed
  raw (Qwen 0.5B-base V_raw 0.259 on v2f; OLMo base 0.445 on v3f) — so post-training
  does not install it; it fails to fix it and can worsen it (OLMo chat-V drops to
  0.27–0.30 across SFT/DPO/RLVR). The correct claim: no stage of training makes
  self-report reliable in the inferential regime, while the workspace channel is
  fixed at pretraining.
- **C4 — memory-gating payoff: VALIDATED at small scale, directional at 7B, null at
  3B on the valid instrument, with an informative boundary.** v2f 0.5B: workspace >
  verbal p=0.0015 (k=2), p<0.0001 (k=3). v4x 7B (64-token budget): workspace@3 0.529 ≈
  selection oracle 0.485, ordering workspace > embedding > verbal, k=1 p=0.065.
  v4x 3B: no separations (workspace@3 0.338 vs oracle 0.456) — C4 has no positive
  evidence at 3B on the valid instrument. Recency/random never beat workspace anywhere
  and lose significantly in most v2f cells (n.s. in the low-ceiling v4x cells).
  Embedding never separates downstream; predictively it is DISPATCHED on the valid
  instrument — **v4 7B W − embedding = +0.170 [+0.072, +0.275], P=1.00** (embedding at
  chance 0.471, `pilot/embed_auc_v4.txt`) — and only marginal on v2f/v3f (+0.080 /
  +0.112, CIs include 0). Boundary: when recall-time composition exceeds model
  competence (24-token budget, or 3B on two-hop answers), selection quality stops
  being the binding constraint.
- **Causal (TODO 6): VALIDATED (demonstration-scale).** Direction-specific workspace
  patching flips memory-based answers 10× more often than sham at gentle scale
  (0.50 vs 0.05, 3B, 20 sliding-window pairs — pairs share episodes, so treat as a
  demonstration; a pre-registered independent-pairs version belongs in the paper).

Paper framing that all evidence now supports: *"Verbal self-report of importance is
systematically anti-informative exactly for inferred information, at every scale and
post-training stage tested; a zero-training workspace readout recovers the evoked
inferences that carry that information, predicts their future utility, causally feeds
the answers built from memory, and — under a budget — selects memories at near-oracle
downstream quality where selection is the binding constraint."*

---

# Gap-fill campaign (2026-07-10): every reviewer hole closed

## Master statistics table (`pilot/master_table.md`)
Episode-cluster bootstrap (B=4000) for every cell; v4 primary contrasts
Bonferroni-corrected (α=0.05/4). Highlights:
- v4 W−V: Bonferroni-significant at 0.5B/3B/7B; 1.5B misses under correction
  ([−0.019, +0.258], P=0.984).
- v4 W−embedding: significant at 3B (+0.123, P=0.993) and 7B (+0.170, P=1.000).
- v3 W−V: cluster-significant at ALL sizes; W−embed significant at 1.5B/3B/7B.
- **v1f completes the regime × method interaction**: on explicit info, V beats W at
  1.5B/3B/7B (−0.102…−0.142, P≤0.010) AND embedding beats W at every size
  (−0.097…−0.122, P≤0.006). Cheap heuristics dominate where info is on the surface;
  the workspace readout is uniquely valuable exactly in the inferential regime.

## Causal, independent pairs (audit fix) — now formally significant
Disjoint pairs, sham directions from held-out episodes, exact McNemar real-vs-sham:
| run | flip | sham | McNemar | Wilcoxon (Δlog-odds) |
|---|---|---|---|---|
| 3B scale 0.5 | 0.60 | 0.10 | +11/−1 **p=0.0063** | p=0.00013 |
| 3B scale 1.0 | 0.85 | 0.35 | +10/−0 **p=0.0019** | p<0.00001 |
| 0.5B scale 1.0 | 0.75 | 0.20 | +11/−0 **p=0.0010** | p=0.00017 |

## Second family: OLMo-2-1124-7B — replication STRONGER than Qwen
- v4: base W=0.684 [0.612, 0.753], instruct W=0.665; V=0.352;
  **W−V=+0.313 [+0.217, +0.408]**. No verbal catch-up at 7B in this family at all.
- v2f: base 0.673 / instruct 0.669 (base ≈ instruct again); W−V=+0.130 sig.

## Second generator (gpt-4o): the dissociation is generator-independent;
## the oracle level is not
gpt-4o could not satisfy v4's strict constraints (4/150 episodes survived) but
produced a valid v2-style battery (57 eps / 271 items). On it: W−V replicates
(+0.204 / +0.265 / +0.109 at 0.5B/3B/7B, first two significant; V stays
anti-informative 0.22–0.38) — but W itself sits at chance (~0.49) at every size.
Reading: W − V (the dissociation) is robust to the generator; the absolute
positive-oracle level depends on how strongly the generator's prose EVOKES the
bridge (gpt-4.1's contexts evoke it; gpt-4o's do not) — same lesson as v3 vs v4.

## v4-XL (120 eps / 572 items / 120 positives, person-name deduped) — both targets hit

**Predictive (n=572, tight CIs):** W−V significant at ALL sizes with comfortable
margins — +0.240 / **+0.113 [+0.031, +0.197]** (1.5B off the knife edge) / +0.292 /
+0.163, all P=1.00. W scaling 0.49 → 0.55 → 0.55 → 0.60 (7B CI [0.543, 0.653]).

**Downstream (64-token, n=120 episodes) — C4 lands on the valid instrument:**

| size | workspace@3 | verbal@3 | embed@3 | oracle@3 | W-vs-V McNemar |
|---|---|---|---|---|---|
| 0.5B | 0.258 | 0.217 | 0.333 | 0.300 | n.s. (task too hard) |
| 1.5B | 0.350 | 0.325 | 0.392 | 0.375 | n.s. |
| 3B | 0.358 | 0.325 | 0.308 | 0.433 | k=2 +15/−8 p=0.21 (directional) |
| **7B** | **0.525** | 0.367 | 0.450 | 0.525 | k=2 +25/−10 **p=0.0167**; k=3 +26/−7 **p=0.0013** |

At 7B, workspace-gating EXACTLY matches the selection oracle (0.525) and beats
verbal-gating with real significance on the fully construct-valid battery. At 3B the
composition ceiling (oracle 0.433 ≪ full-context 0.567) still compresses policies —
the honest boundary stands: below ~7B, recall-time competence, not selection, binds.
C4's small-scale significance remains the v2f 0.5B result (p=0.0015 / p<0.0001),
where memory retrieval rather than composition is the bottleneck.

## Multimodal (TODO 7, Qwen2-VL-2B) — the regime map extends to perception
Landmark battery (46 eps): image carries place identity; neutral templated text
(zero local color); probe anchor-free. THREE readouts, two ablations:
- Within-class test (true city vs other cities): **W(image)=0.483 — chance;
  W(no image)=0.475; paired diff +0.008 [−0.052, +0.069]** — visually-presented
  identity does NOT enter the encoding-time word-level workspace readably (2B VLM).
- **V(image)=0.986 vs V(no image)=0.054** — ask-time verbal probing has essentially
  perfect access to what is IN the image (it can look again).
Conclusion: perceptually-present information behaves like EXPLICIT information
(v1 regime — self-report excellent, workspace readout adds nothing). The silent-
bridge phenomenon is a property of inference over text, not of perceptual grounding.
Caveat: 2B model, logit-lens readout; a visual bridge might exist at larger VLM
scale or under a trained readout.

## Caveats / what the powered version needs
- Bigger models (3B/7B) — 0.5B is likely too weak to hold silent bridges.
- More items (~500) for tighter CIs.
- Trained future-lens instead of raw logit-lens readout (cleaner W).
- The real payoff experiment (not yet run): re-inject high-W vs high-V memories at recall
  and measure downstream QA accuracy — does workspace-gated memory actually help?
