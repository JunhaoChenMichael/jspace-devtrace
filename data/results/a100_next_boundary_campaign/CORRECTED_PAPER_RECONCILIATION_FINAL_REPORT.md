# Corrected paper reconciliation: final report

Every number here is measured under the corrected verbal score (schema v3). No value was carried over from a v2 artifact.

## 1. Impact inventory

| Classification | Count |
|---|---:|
| `DONE` | 7 |
| `INPUTS_ABSENT_FROM_THIS_REPOSITORY` | 3 |
| `NOT_PRESENT` | 1 |
| `RESCORE_ONLY` | 2 |
| `RETRAIN_REQUIRED` | 1 |
| `UNAFFECTED` | 4 |

Blocked by absent inputs, not by the defect:

- verbal-gated downstream policies (binary-verbal gating, top-k containment, mixed-pool, case studies, naturalistic streams)
- Qwen2.5-7B RL-QA Original arm
- metacognitive objective study (Binary / Soft / Pairwise / Listwise)
- paper tables and figures

This repository excludes trained artifacts and completed campaign directories by policy, and `paper/` is a separate repository. Those components are **unverified**, not cleared.

## 2. Corrected predictive grid

24 checkpoints across four families, four conditions each, measured fresh. Because the original result files are not tracked here, this is a new measurement rather than a correction of a stored one.

### Decoupled, paired W−V with 4,000-draw episode-cluster intervals

| Model | V | W_rr | gap | 95% CI | Ms |
|---|---:|---:|---:|---|---:|
| Qwen/Qwen2.5-0.5B | 0.4798 | 0.5019 | +0.0221 | [-0.0525, +0.1002] | — |
| Qwen/Qwen2.5-0.5B-Instruct | 0.2841 | 0.5261 | +0.2420 \* | [+0.1540, +0.3285] | — |
| Qwen/Qwen3-0.6B | 0.4861 | 0.5364 | +0.0503 | [-0.0456, +0.1529] | — |
| Qwen/Qwen2.5-1.5B | 0.4052 | 0.5903 | +0.1851 \* | [+0.1102, +0.2637] | 0.686 |
| Qwen/Qwen2.5-1.5B-Instruct | 0.4637 | 0.5768 | +0.1131 \* | [+0.0028, +0.2229] | 0.804 |
| Qwen/Qwen2.5-3B | 0.3784 | 0.6062 | +0.2278 \* | [+0.1272, +0.3301] | 0.624 |
| Qwen/Qwen2.5-3B-Instruct | 0.2586 | 0.5951 | +0.3366 \* | [+0.2462, +0.4292] | 0.435 |
| Qwen/Qwen2.5-7B | 0.3730 | 0.6121 | +0.2391 \* | [+0.1503, +0.3288] | 0.609 |
| Qwen/Qwen2.5-7B-Instruct | 0.4897 | 0.6423 | +0.1526 \* | [+0.0500, +0.2582] | 0.762 |
| Qwen/Qwen3-14B | 0.3480 | 0.6677 | +0.3196 \* | [+0.2156, +0.4224] | 0.521 |
| Qwen3-1.7B | 0.3174 | 0.5607 | +0.2433 \* | [+0.1368, +0.3475] | 0.566 |
| Qwen3-4B | 0.4862 | 0.5291 | +0.0429 | [-0.0416, +0.1289] | — |
| Qwen3-8B | 0.4020 | 0.6545 | +0.2525 \* | [+0.1713, +0.3396] | 0.614 |
| Qwen3-14B | 0.3480 | 0.6677 | +0.3196 \* | [+0.2156, +0.4224] | 0.521 |
| Qwen3-32B | 0.3366 | 0.6919 | +0.3553 \* | [+0.2700, +0.4486] | 0.486 |
| allenai/OLMo-2-0425-1B-DPO | 0.3799 | 0.5802 | +0.2003 \* | [+0.1279, +0.2801] | 0.655 |
| allenai/OLMo-2-0425-1B-Instruct | 0.3284 | 0.5801 | +0.2517 \* | [+0.1818, +0.3254] | 0.566 |
| allenai/OLMo-2-0425-1B-SFT | 0.3716 | 0.5725 | +0.2009 \* | [+0.1241, +0.2819] | 0.649 |
| allenai/OLMo-2-1124-7B-Instruct | 0.3529 | 0.6664 | +0.3135 \* | [+0.2297, +0.3992] | 0.530 |
| allenai/OLMo-2-1124-7B-SFT | 0.3317 | 0.6637 | +0.3320 \* | [+0.2551, +0.4094] | 0.500 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.4700 | 0.6713 | +0.2014 \* | [+0.1000, +0.3015] | 0.700 |

`*` marks an interval excluding zero. `Ms` is reported only where `W_rr ≥ 0.55`.

**The central provenance claim survives across families at capable scale.** Every 7B-and-up instruct checkpoint measured — Qwen2.5-7B-Instruct +0.1526, Qwen3-8B +0.2525, OLMo-2-1124-7B-Instruct +0.3135, Mistral-7B-Instruct +0.2014 — has a positive Decoupled gap whose interval excludes zero.

**The old small-scale claim does not survive.** Qwen2.5-0.5B (+0.0221), Qwen3-0.6B (+0.0503) and **Qwen3-4B (+0.0429)** all have intervals spanning zero. The statement that the gap is significantly positive at 4B must not be restored.

## 3. Qwen3 scale-gap analysis, paired across sizes

| Step | Decoupled Δgap | 95% CI | Evoked Δgap | 95% CI |
|---|---:|---|---:|---|
| 1.7B → 4B | -0.2004 \* | [-0.3137, -0.0937] | -0.0241 | [-0.1207, +0.0720] |
| 4B → 8B | +0.2097 \* | [+0.1277, +0.2938] | +0.0905 \* | [+0.0114, +0.1689] |
| 8B → 14B | +0.0671 | [-0.0234, +0.1545] | +0.0715 | [-0.0073, +0.1517] |
| 14B → 32B | +0.0357 | [-0.0572, +0.1324] | -0.0517 | [-0.1288, +0.0267] |
| 8B → 32B | +0.1027 \* | [+0.0093, +0.1944] | +0.0198 | [-0.0714, +0.1099] |

The 8B→32B widening on Decoupled is **+0.1027, CI [+0.0093, +0.1944]** — supported. Neither adjacent step inside that range is individually resolved, and **Evoked is not monotonic** (14B→32B is −0.0517). Do not claim universal monotonic widening.

## 4. Chat versus template-free, paired within each model

| Model | V_chat − V_raw | 95% CI |
|---|---:|---|
| Qwen3-1.7B | +0.0341 | [-0.0471, +0.1139] |
| Qwen3-4B | +0.0457 | [-0.0327, +0.1211] |
| Qwen3-8B | -0.0981 \* | [-0.1909, -0.0104] |
| Qwen3-14B | -0.0994 \* | [-0.1866, -0.0102] |
| Qwen3-32B | -0.1895 \* | [-0.2927, -0.0960] |

This is a fixed-model paired contrast, so it does support a statement about the interface: at 8B, 14B and 32B the chat-template reading is **worse** than the template-free one, and increasingly so with scale. Note the direction — the retracted claim had the chat pathway carrying an improvement; corrected, it carries a penalty.

## 5. Metacognitive Alignment across scale

| Scale | seeds | ΔV Decoupled | 95% CI | V_after | ΔW | Compositional ΔV | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| Qwen3-8B | 3 | +0.213 mean | — | 0.566–0.689 | <0.001 | +0.086 | GREEN |
| Qwen3-14B | 1 | +0.3826 | [+0.3029, +0.4701] | 0.7306 | -0.00017 | +0.3261 | GREEN (strong) |
| Qwen3-32B | 1 | +0.0954 | [+0.0241, +0.1764] | 0.4320 | +0.00116 | -0.0176 | AMBER |

**The repair is not monotonic in scale.** 14B is the strongest point measured and the only one where Compositional also transfers; 32B has the widest gap to repair and the weakest repair. Both 14B and 32B rest on a **single seed**, so this is an observation to test, not an established result.

## 6. Claims

**Surviving.** The provenance-specific dissociation holds across Qwen2.5, Qwen3, OLMo-2 and Mistral at 7B and above. Workspace-derived self-distillation repairs the verbal reporter at 8B across three seeds and at 14B on one seed, without materially moving the workspace.

**Qualified.** The 8B→32B widening is supported on Decoupled only. Adjacent steps are unresolved and Evoked is non-monotonic.

**Retracted.** Significance at 4B in every inferred regime. The 14B→32B verbal jump. The 32B `SCALE_BOUNDARY` verdict. The 32B RL-QA `FAIL` verdict.

**New, and needing review before use.** The gap widens rather than closes across the capable range; the chat interface reads worse than the template-free probe at 8B and above; the alignment repair peaks at 14B.

**Unverified, not cleared.** Everything listed as blocked in section 1.

## 7. Provenance

- Repository commit: `fa2093dae65678f8ba5d194677144944b2e06c62`
- Verbal score schema: `workspace_measurement_metadata.v3`
- Statistics: 4,000-draw whole-episode cluster bootstrap; contrasts on shared candidates share their draws
- Test suite: 220 passing across the Qwen3-8B / 14B / 32B configurations

## 8. Authorisation

Measurement, plus the two trainings the corrected gates authorised (Qwen3-32B and Qwen3-14B Binary seed 0). No seed expansion, no model above 32B, no RL retraining, no threshold revisited after seeing the measurement it gates. Each campaign consumed exactly one OOD attempt.
