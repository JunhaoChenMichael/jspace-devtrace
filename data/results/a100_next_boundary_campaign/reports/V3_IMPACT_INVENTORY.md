# v3 impact inventory

Every consumer of the verbal score, classified. The recovery plan gates further work on this document.

| Component | Classification | Why |
|---|---|---|
| measure.py verbal_salience / verbal_salience_raw | `DONE` | the defect itself; corrected to a log-space ratio with no guard epsilon |
| locomo_gate / longmemeval_gate / vprobe_robust / measure_vlm verbal scores | `DONE` | same defect, corrected through the shared helper |
| vrating_baseline 1-10 rating | `DONE` | same defect class: the digit mass sat under its own guard; renormalised in log space |
| W_rr workspace readout | `UNAFFECTED` | reciprocal rank over layers; no epsilon anywhere in the path |
| frozen teacher labels | `UNAFFECTED` | derived from W_rr, never from the verbal score |
| RL admission policy scoring | `UNAFFECTED` | binary_action_logits normalises in log space over two actions |
| metacognitive ID checkpoint selection | `UNAFFECTED` | train_metacog_m1.py scores through binary_action_logits, so the locked checkpoints were selected on an unaffected quantity |
| Qwen3-8B metacognitive three-seed report | `DONE` | re-measured from the locked adapters: mean delta V +0.273 -> +0.213, all seeds still pass |
| Qwen3-32B metacognitive M0 gate and seed 0 | `DONE` | gate reversed to MISALIGNMENT_REGIME; the re-run trained and returned AMBER |
| Qwen3 scale sweep | `DONE` | re-measured under v3 across five dense sizes plus the sparse diagnostic |
| Qwen3-8B and Qwen3-32B RL-QA Original arms | `DONE` | Original arm re-scored and its budget-2 sets regenerated; 8B survives, 32B reclassified from FAIL to ADMISSION_POSITIVE_QA_UNRESOLVED |
| cross-family predictive grid (Qwen2.5, OLMo-2, Mistral, GPT-2) | `RESCORE_ONLY` | measurement only, and re-runnable from scratch: the batteries are tracked, the checkpoints are public. The ORIGINAL result files are not in this repository, so this is a fresh measurement rather than a correction of a stored one |
| Qwen3-14B Binary metacognitive seed 0 | `RETRAIN_REQUIRED` | never run; the corrected gate decides whether it may train |
| verbal-gated downstream policies (binary-verbal gating, top-k containment, mixed-pool, case studies, naturalistic streams) | `INPUTS_ABSENT_FROM_THIS_REPOSITORY` | requires the original downstream run artifacts, which are project outputs excluded from git by repository policy |
| Qwen2.5-7B RL-QA Original arm | `INPUTS_ABSENT_FROM_THIS_REPOSITORY` | the completed Qwen2.5-7B campaign directory is not tracked here |
| metacognitive objective study (Binary / Soft / Pairwise / Listwise) | `NOT_PRESENT` | no such study exists in this repository; mixed_pool.py is a different analysis and does not use the verbal probe for winner classification |
| paper tables and figures | `INPUTS_ABSENT_FROM_THIS_REPOSITORY` | paper/ is a separate git repository and is not checked out here |
| multimodal verbal probing | `RESCORE_ONLY` | measure_vlm is corrected, but no stored VLM result files exist here to compare against; a fresh measurement is possible |

## Counts

- `DONE`: 7
- `INPUTS_ABSENT_FROM_THIS_REPOSITORY`: 3
- `NOT_PRESENT`: 1
- `RESCORE_ONLY`: 2
- `RETRAIN_REQUIRED`: 1
- `UNAFFECTED`: 4

## What cannot be produced on this machine

Several components are blocked not by the defect but by absent inputs. This repository deliberately excludes trained artifacts and completed campaign directories, and `paper/` is a separate repository. The following need those inputs restored from the originating server before they can be corrected:

- verbal-gated downstream policies (binary-verbal gating, top-k containment, mixed-pool, case studies, naturalistic streams)
- Qwen2.5-7B RL-QA Original arm
- metacognitive objective study (Binary / Soft / Pairwise / Listwise)
- paper tables and figures

Nothing here is a judgement that those results are fine. They are unverified.
