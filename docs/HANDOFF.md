# HANDOFF — Workspace-grounded memory / introspective-salience failure

Migration doc for continuing this project on a GPU server. Self-contained: thesis,
every experiment run so far with numbers, code layout, environment, credentials,
exact reproduction commands, and the prioritized TODO. Work done 2026-07-08/09 on
an Apple M5 Pro (MPS); the science is device-agnostic and will run faster on CUDA.

---

## 0. TL;DR

We are building a paper around a discovery the pilot already supports:

> **A memory-bottlenecked LLM agent should decide what to remember from what enters
> its WORKSPACE, not from what it REPORTS as important — because self-reported
> importance is systematically wrong exactly for information that must be INFERRED
> rather than read off the surface, and that is precisely the information worth
> remembering.**

Two things are already shown on open models (Qwen2.5 0.5B & 3B, GPT-2):
1. **Predictive:** in the silent-inference regime, a workspace readout predicts which
   items are truly useful *significantly* better than the model's own verbal
   reflection, which is *below chance* there (3B: AUC 0.62 vs 0.41; W−V=+0.21, P=1.00).
2. **Downstream (payoff):** memory-budgeted QA — storing what the workspace holds
   beats storing what the model says is important; the verbal policy is *worse than
   remembering nothing* on silent-bridge questions (0.5B: workspace 0.52 vs verbal
   0.26 vs no-memory floor 0.35).

The effect is **regime-dependent** (this is the sharp, defensible claim): self-report
is fine for *explicit* info (and can beat the workspace there) and fails only for
*inferred* info. And it is **developmental**: workspace utility-tracking is a
pretraining capability (base ≈ instruct); post-training installs the verbal-report
layer that is miscalibrated in the inferential regime.

---

## 1. The complete core idea (what the paper must validate)

Working title: **"Remember What You Thought, Not What You Said: Workspace-Grounded
Memory for LLM Agents."** Framing: a memory/agent paper whose depth comes from a
science finding about the unreliability of LLM introspective salience.

Four claims, each falsifiable, ordered with the risky part last:

- **C1 — Dissociation (regime-dependent).** What an LLM *computes with* (workspace)
  and what it *reports as important* (verbal reflection) diverge, and the divergence
  is structured: verbal reflection tracks utility for EXPLICIT information but is
  anti-informative for INFERRED information. *Status: supported (0.5B & 3B).*
- **C2 — Workspace is a positive oracle, scaling with capability.** A model capable
  of the bridge inference holds the silent concept in its workspace; the workspace
  signal on silent bridges rises with model size. *Status: supported (0.52@0.5B →
  0.62@3B); confirm at 7B.*
- **C3 — Developmental origin.** Utility-salience is a pretraining capability (base ≈
  instruct on the workspace channel); post-training adds a verbal-report layer that is
  the miscalibrated one. Ties to Anthropic's counterfactual-reflection (the verbal
  channel is separable from the underlying computation). *Status: supported, replicated
  at 0.5B & 3B; extend to SFT/DPO/RLVR stages (OLMo-2/Tulu-3).*
- **C4 — It matters (application).** Under a memory budget, workspace-gated memory
  yields higher downstream QA accuracy than reflection-gated memory, with the gap
  concentrated in the inferential regime. *Status: supported at 0.5B v2; run 3B/7B +
  the v1 regime contrast + significance.*

Money sentence for the abstract: *self-reported importance is worse than useless
exactly where memory matters most — for what the model inferred but never said — and
the workspace recovers it.*

---

## 2. Method (how the workspace is read — reimplementation notes)

Anthropic's "J-lens" (Jacobian lens) finds, per vocab token, the internal activation
that raises the FUTURE probability of saying that token; the readout reveals "silent
words in the model's mind." We reimplement a tractable dual on open weights:

- **Readout (`jlens.py`):**
  - `logitlens` — project a residual state through final-norm + unembedding (zero
    training). NOTE: noisy on non-GPT-2 models; use rank not raw prob (below).
  - `FutureLens` (`fit_future_lens`) — per-layer affine probe trained so softmax puts
    mass on tokens appearing in a future window. This is the principled dual and the
    recommended upgrade; NOT yet used in the pilot (logit-lens rank was enough).
- **Workspace salience used in the pilot (`pilot/measure.py::workspace_salience`):**
  read the residual stream at the END of the context (workspace state after encoding);
  score a concept by its peak RECIPROCAL RANK (`W_rr = max_layers 1/rank`) in the
  logit-lens readout. Rank is scale-free — critical, because raw softmax prob over a
  ~150k vocab is degenerate (~0 for most tokens). `W_end`/`W_max` (raw prob) are kept
  as secondary/noisy.
- **Verbal reflection baseline `V` (same model, black-box):** ask "is <concept>
  important to remember for future questions? yes/no", read P(yes)/(P(yes)+P(no)).
- **Causal test (`patch.py`):** to prove the workspace is USED not a scoreboard, swap a
  concept in the residual stream. KEY LESSON: steering toward the token's UNEMBEDDING
  row FAILS (it's the output direction); steering along an INPUT-side
  difference-of-means concept vector (`jlens.concept_vector`, RepE/function-vector
  style) WORKS — verified France→China flip on Qwen2.5-0.5B-Instruct
  (`concept_swap_repe`/`swap_effect_repe`). Use the input-side version.

Both W and V are computed BLIND to the probe question (encoding-time prediction);
ground-truth utility labels come from the battery. Same lens on every checkpoint, so a
fixed lens bias cancels in checkpoint-to-checkpoint contrasts.

---

## 3. Results so far (all numbers)

Metric = AUC of a signal at classifying `load_bearing` items, with 95% item-level
bootstrap CIs (2000×). `W_rr` = workspace, `V` = verbal reflection.

### 3a. Pilot — predictive (per-item)

**v1 battery** (`pilot/battery.json`, 30 episodes, 164 items: 53 load_bearing / 67
distractor / 44 filler; concepts mostly APPEAR literally):

| model | W_rr [95% CI] | V [95% CI] | W − V [95% CI] |
|---|---|---|---|
| gpt2 (pretrained) | 0.605 [0.507, 0.700] | — | — |
| Qwen2.5-0.5B base | 0.591 [0.495, 0.688] | — | — |
| Qwen2.5-0.5B instruct | 0.604 [0.504, 0.705] | 0.535 [0.443, 0.631] | +0.069 [−0.077, +0.217] n.s. |
| Qwen2.5-3B base | 0.606 [0.512, 0.698] | — | — |
| Qwen2.5-3B instruct | 0.575 [0.477, 0.671] | **0.625 [0.532, 0.715]** | −0.050 [−0.174, +0.083] (V wins, n.s.) |

**v2 silent-bridge battery** (`pilot/battery_v2.json`, 31 episodes, 165 items: 31 silent
load_bearing / 74 distractor / 60 filler; load_bearing concept NEVER appears, must be
inferred; distractors are vivid words that DO appear):

| model | W_rr [95% CI] | V [95% CI] | W − V [95% CI] |
|---|---|---|---|
| Qwen2.5-0.5B base | 0.524 [0.402, 0.643] | — | — |
| Qwen2.5-0.5B instruct | 0.524 [0.400, 0.647] | 0.307 [0.191, 0.426] | **+0.217 [+0.060, +0.372] sig** |
| Qwen2.5-3B base | 0.609 [0.493, 0.723] | — | — |
| Qwen2.5-3B instruct | **0.622 [0.508, 0.730]** | 0.410 [0.297, 0.522] | **+0.211 [+0.083, +0.341] P=1.00** |
| gpt2 (pretrained) | file `results_v2_gpt2.json` exists — NOT yet analyzed | — | — |

**Disagreement** (percentile-rank, v2, items where W and V most disagree; base rate
P(load_bearing)≈0.19): 0.5B-instruct → workspace-favored 0.333 vs reflection-favored
0.091; 3B-instruct → 0.226 vs 0.088. Workspace's disagreements are ~3× more often right.

### 3b. Confound check (`pilot/confound.py`) — NOT a surface artifact
v1: literal-appearance rate 0.98 for BOTH load_bearing and non-load ⇒ "did the word
appear" predictor AUC = **0.500**, yet W_rr = 0.60 (and 0.60 on the appearing-only
subset). v2 removes the confound by construction (load_bearing never appears).

### 3c. Developmental (base vs instruct), W_rr AUC — post-training does NOT improve the workspace
| battery | 0.5B base→instruct | 3B base→instruct |
|---|---|---|
| v1 | 0.591 → 0.604 | 0.606 → 0.575 |
| v2 | 0.524 → 0.524 | 0.609 → 0.622 |
Workspace utility-tracking ≈ constant base↔instruct (both sizes). The verbal channel
`V` exists only in instruct and is the regime-dependently-miscalibrated one.

### 3d. Downstream memory-QA (`pilot/downstream.py`) — the payoff
Memory-budgeted QA: keep top-k items under a policy, answer probe from those only.
**Qwen2.5-0.5B-Instruct, v2 silent battery** (only completed downstream run):

| policy | k=1 | k=2 |
|---|---|---|
| **workspace** | 0.323 | **0.516** |
| verbal | 0.258 | 0.258 |
| random (3 seeds) | 0.301 | 0.452 |
| oracle | 0.968 | 0.871 |
| ref: full_context | 0.484 | (ceiling) |
| ref: no_memory | 0.355 | (floor) |

Headline: workspace-gated (0.52) > verbal-gated (0.26), gap +0.26; and **verbal-gated
is BELOW the no-memory floor (0.35)** — storing what the model says is important is
worse than storing nothing, because it stores confidently-wrong vivid distractors.
3B/7B downstream and the v1 (explicit) contrast were NOT completed (stopped for
migration).

---

## 4. Repository layout

```
jlens.py            WorkspaceLens: residual capture (arch-agnostic), logit-lens +
                    future-lens readouts, concept_vector (input-side), token_direction.
patch.py            concept_swap (unembedding, WEAK baseline) + concept_swap_repe /
                    swap_effect_repe (input-side diff-of-means, WORKS).
metrics.py          perspective_index, reasoning_trace_strength, workspace_capacity
                    (dev-trajectory metrics; used less than the pilot/ battery approach).
run_demo.py         end-to-end demo: silent-word readout + France→China causal flip.
run_devtraj.py      base-vs-instruct trajectory harness (metric battery).
probe_swap.py/2.py  scratch: showed unembedding-steering fails, diff-of-means works.
README.md           the broader research program (developmental trajectory + brain bridge).

pilot/
  gen_battery.py    v1 battery generator (GPT-4.1). Needs OPENAI_API_KEY.
  gen_battery_v2.py v2 SILENT-BRIDGE generator (load_bearing concept never in text).
  battery.json      v1 data (30 ep). battery_v2.json = v2 (31 ep).
  measure.py        compute per-item W_rr/W_end/W_max and V on an open model.
                    Flags: --model --battery --dtype {float32,bfloat16} --no-verbal --out
  analyze.py        AUCs + bootstrap CIs + paired W−V CI + disagreement breakdown.
  confound.py       surface-presence controls.
  downstream.py     memory-budgeted QA experiment (the payoff).
  RESULTS.md        running results narrative (numbers + interpretation).
  results_*.json    per-item measurements (see filenames; gitignored — copy via rsync).
  downstream_*.json completed downstream runs.
  oa_check.py       OpenAI key/model sanity check.
```

Naming: `results_{v1|v2}_{model}.json`; v1 files for 0.5B-Instruct/gpt2 are
`results_0.5B-Instruct.json` / `results_gpt2-base.json` (no v1 prefix, historical).

---

## 5. Environment

- **Hardware used:** Apple M5 Pro, 26 GB unified memory, MPS. On the server you'll have
  CUDA — `jlens.pick_device()` auto-selects `cuda`; use `--dtype bfloat16` for ≥3B.
- **Python:** `/opt/anaconda3/bin/python` = Python 3.13.9 (a conda base env). The
  homebrew python 3.14 had NO torch wheels — use conda/venv ≤3.13.
- **Packages (versions used):** torch 2.11.0, transformers 5.5.4, openai 2.31.0,
  scikit-learn 1.7.2, scipy 1.16.3, huggingface_hub, accelerate. See `requirements.txt`
  (torch/transformers/accelerate) + add `openai scikit-learn scipy`.
- **Cached HF models (already downloaded locally; on the server, re-pull with HF_TOKEN):**
  gpt2, Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct, Qwen2.5-3B, Qwen2.5-3B-Instruct (all
  complete), **Qwen2.5-7B-Instruct — INCOMPLETE (~8.8 GB of ~15 GB; download was
  killed for migration — re-pull with HF_TOKEN before use)**, Qwen2-VL-2B-Instruct
  (for the multimodal experiment), BAAI/bge-small-en-v1.5 & all-MiniLM-L6-v2 (for the
  embedding baseline). On the server just re-download everything fresh with HF_TOKEN.
- **MPS gotcha:** fp32 3B ×2 concurrent = 24 GB → thrash on 26 GB. Use bfloat16. On CUDA
  this is moot for small models.

### Migration note
`server_env.sh`, `pilot/results_*.json`, `pilot/battery*.json`, and `*.log` are
**gitignored** — a `git clone` will NOT bring them. Copy the WHOLE directory
(`rsync -av` / `tar`) so the batteries (cost API $ to regenerate) and measurements
travel. Or regenerate batteries with the OpenAI key.

---

## 6. Credentials  ⚠️ ROTATE

The OpenAI API key and HF token you provided are saved in **`server_env.sh`**
(gitignored, chmod 600). `source server_env.sh` on the server. **Both were pasted in a
chat session and must be treated as compromised — rotate them** (OpenAI: platform →
API keys; HF: settings → tokens), then update `server_env.sh`.
- `OPENAI_API_KEY` — battery generation (`gen_battery*.py`, `oa_check.py`) and optional
  LLM grading. Not needed to re-run measurements/downstream on existing batteries.
- `HF_TOKEN` (+ `HUGGING_FACE_HUB_TOKEN`) — authenticated, un-throttled model downloads.
  (Unauthenticated HF downloads were pathologically throttled/stalled; the token fixed it.)

---

## 7. Reproduce / continue on the server

```bash
cd jspace-devtrace
source server_env.sh                       # sets OPENAI_API_KEY, HF_TOKEN
python -m pip install torch transformers accelerate openai scikit-learn scipy

# (batteries already exist; regenerate only if needed)
# python pilot/gen_battery.py    --n 30 --out pilot/battery.json
# python pilot/gen_battery_v2.py --n 40 --out pilot/battery_v2.json

# measure a checkpoint on a battery (bfloat16 for >=3B; --no-verbal for base models)
python pilot/measure.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
    --battery pilot/battery_v2.json --out pilot/results_v2_7B-Instruct.json
python pilot/measure.py --model Qwen/Qwen2.5-7B --dtype bfloat16 --no-verbal \
    --battery pilot/battery_v2.json --out pilot/results_v2_7B-base.json

# analyze (AUCs + bootstrap CIs + paired W-V + disagreement)
python pilot/analyze.py  pilot/results_v2_7B-base.json pilot/results_v2_7B-Instruct.json
python pilot/confound.py pilot/results_v2_7B-Instruct.json

# downstream payoff QA (workspace vs verbal vs random vs oracle, budgets)
python pilot/downstream.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
    --results pilot/results_v2_7B-Instruct.json --battery pilot/battery_v2.json --budgets 1,2,3
python pilot/downstream.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
    --results pilot/results_v1_7B-Instruct.json --battery pilot/battery.json --budgets 1,2,3
```

---

## 8. TODO (prioritized) — what to run next on the server

**STATUS UPDATE 2026-07-09 (server session):** items 1 and 2 are DONE — all numbers in
`pilot/RESULTS.md` § "Server campaign". Headline surprises: (a) at 7B the VERBAL channel
catches up on silent bridges (V-AUC 0.31→0.41→0.60 across 0.5B/3B/7B; W−V n.s. at 7B),
so C1 is now capability-relative: reflection-gating fails ≤3B, workspace-gating is
scale-robust; (b) downstream at k=3 workspace hits 0.74 (3B) / 0.90 (7B), beating the
full-context ceiling and nearly matching the oracle at 7B; (c) NEW embedding
(bge cosine) baseline wins at 0.5B but loses to workspace at 3B/7B; recency loses
everywhere (sig.). Item 3 (scaled batteries) is IN PROGRESS: `battery_500.json` /
`battery_v2_500.json` (~100 eps each), `pilot/dedup_battery.py` for global concept
de-dup, `run_scaled.sh` to measure all 7 checkpoints on both, `pilot/summary.py` for
the master table.

1. ~~**Finish the powered pilot at 7B**~~ DONE — see RESULTS.md. v2 W_rr 0.643 (base &
   instruct identical → C3 holds at 7B); v1 V 0.695 > W 0.556 (regime contrast, P=0.04);
   BUT v2 W−V only +0.040 n.s. because V improved to 0.603 at 7B.
2. ~~**Complete the downstream experiment fully**~~ DONE — 0.5B/3B/7B × v1/v2 with
   per-episode logs, exact McNemar, embedding + recency baselines (all in upgraded
   `downstream.py`). v2 workspace@k3: .58/.74/.90; verbal below floor only at 0.5B;
   v1: all policies tie (n.s.) at every size.
3. **Scale the batteries to ~500 items** each (tighter CIs; current n≈165 gives wide CIs)
   and de-dup concepts across episodes. Regenerate with `gen_battery*.py`. IN PROGRESS —
   after measure, ALSO re-run downstream on the scaled batteries (n=100 episodes gives
   McNemar real power; the pilot 0.09s should resolve).
4. ~~**Upgrade the readout to the trained FutureLens**~~ DONE (`pilot/futurelens_exp.py`,
   tuned lens init from unembedding, trained on disjoint v1f contexts). RESULT: improves
   v3 slightly at every size but never past ~0.59; significantly WORSE than logit-lens on
   v2f@7B (−0.088). → The readout is NOT the bottleneck; the workspace holds evoked
   one-hop inferences, not arbitrary future-useful compositions. Battery v4 (evoked
   bridge + decoupled answer + anchor-free probe) resolved this at the instrument level:
   logit-lens W=0.641@7B there.
5. ~~**Developmental trajectory, clean stages**~~ DONE (OLMo-2-0425-1B base/SFT/DPO/RLVR
   on v2f+v3d, with V and V_raw). RESULT: W erodes monotonically base→RLVR (0.631→0.580);
   post-training makes the verbal channel confidently anti-informative (chat-V 0.27–0.30
   on v3 vs base raw-V 0.445). C3 validated across actual stages, second family.
6. ~~**Causal confirmation**~~ DONE (`pilot/causal_swap.py`, 20 pairs, real-vs-sham
   directions, scale sweep). RESULT: 10× direction-specific flip separation at gentle
   scale (3B s=0.5: 0.50 vs 0.05). Memory answers read the workspace.
7. ~~**(Stretch) multimodal**~~ DONE 2026-07-10 (`pilot/gen_battery_vlm.py`,
   `measure_vlm.py`, batteries vlm/vlm2/vlm3 + images in `pilot/vlm_images/`).
   RESULT (double-ablated: no-image + within-class city contrast): visually-presented
   place identity does NOT enter the encoding-time word workspace (W=0.483 within-class,
   = no-image 0.475), while ask-time verbal probing reads the image nearly perfectly
   (V=0.986 vs 0.054 without image). Perceptually-present info behaves like EXPLICIT
   info — the regime map extends to perception; silent bridges are a text-inference
   phenomenon. (Caveat: 2B VLM, logit-lens readout.)

**ALL other experiments complete as of 2026-07-09 night. Full numbers + final verdict on
C1–C4: `pilot/RESULTS.md` § "FINAL VERDICT". Batteries: pilot v1/v2 (superseded),
v1_final/v2_final (scaled), v3d (decoupled, predictive-only), battery_v4_final (THE
construct-valid instrument: evoked bridge + decoupled answer + anchor-free probe).**

---

## 9. Methodological lessons already paid for (don't re-learn these)

- **Logit-lens raw prob is degenerate** on non-GPT-2 models (softmax over 150k vocab ≈ 0).
  Use RECIPROCAL RANK (`W_rr`), or better the trained FutureLens.
- **Unembedding-row steering does nothing** (it's the output direction). Causal edits must
  use INPUT-side difference-of-means concept vectors (`jlens.concept_vector`). Verified.
- **Read workspace at the END of the context** (state after encoding), not at a concept's
  own position (that just reflects surface presence — see the confound result).
- **The claim is regime-dependent.** Do NOT frame it as "workspace always beats
  self-report" — on explicit info (v1) verbal reflection wins. The contribution is
  identifying WHEN self-report breaks (inferred info) and covering that gap.
- **GPT/closed models cannot be the SUBJECT** (no activations → no workspace). They are
  only for battery generation and grading. All workspace measurement is on open weights.
- **HF downloads:** always set `HF_TOKEN`; unauthenticated is throttled to the point of
  stalling. Use `huggingface_hub.snapshot_download(..., allow_patterns=['*.safetensors',
  '*.json','*.txt'])`.
- **MPS memory:** bfloat16 for ≥3B; don't run two fp32 3B jobs at once on 26 GB.
- Python stdout is block-buffered to files → background run logs look empty until they
  finish; check the process, not the log, for liveness (or add `flush=True`).
