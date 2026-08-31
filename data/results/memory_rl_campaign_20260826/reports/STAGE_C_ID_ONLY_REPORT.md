# Stage C ID-only report

## Outcome

All three seed-0, k=2 Hybrid coefficient runs completed 300 steps and passed
their exact-lock formal validators with no errors or warnings. No Stage-C OOD
evaluation or seed expansion was run.

| lambda_W | locked step | ID QA | ID workspace reward | decision |
|---:|---:|---:|---:|---|
| 0.5 | 100 | 32/45 (71.11%) | 0.57037 | eliminated at workspace tier |
| 0.25 | 200 | 32/45 (71.11%) | **0.58426** | surviving exact tie |
| 1.0 | 200 | 32/45 (71.11%) | **0.58426** | surviving exact tie |

The predeclared rule does not yield a unique coefficient. Lambda 0.5 cannot be
revived by its default status after it has lost the workspace tie-break. The
surviving set is `{0.25, 1.0}`, and no registered rule orders those two.
Containment, combined reward, AUC, reporter correlation, and OOD are not used
to break the tie.

Training was healthy in all three runs. Mixed-QA groups were 55.0--55.3%,
96.0--96.3% of groups had nonzero Hybrid reward variance, median unique sets
was 3, only 3.7--4.0% of groups had one unique set, and every gradient norm was
finite and nonzero.

## Reporter alignment

Original, SFT-W, RL-QA, and all three Hybrid checkpoints were rescored on the
same 45 ID episodes / 215 candidates with bf16 batch size 1. `V` is the
condition policy's constrained P(Yes) at temperature 1; `W_ref` is immutable
raw W_rr; `y_utility` is the binary load-bearing-label proxy. All uncertainty
uses 4,000 shared, source-stratified episode-cluster bootstrap draws.

Pooled candidate correlations are:

| condition | Pearson(V,W) | Pearson(V,y) | Spearman(V,W) | Spearman(V,y) |
|---|---:|---:|---:|---:|
| Original | 0.0819 | 0.0482 | 0.1904 | 0.1906 |
| SFT-W | **0.1428** | **0.2645** | 0.1982 | 0.2335 |
| RL-QA | 0.0932 | 0.1284 | 0.2090 | 0.2469 |
| Hybrid-0.5 | 0.0920 | 0.0936 | 0.2038 | 0.2256 |
| Hybrid-0.25 | 0.0934 | 0.1147 | 0.2133 | 0.2439 |
| Hybrid-1.0 | 0.0944 | 0.1254 | **0.2164** | **0.2523** |

The proposed clean pattern is therefore **not established** on pooled ID
correlations. SFT-W has the largest Pearson correlation with both targets,
while rank correlations place RL-QA and Hybrid only modestly above SFT-W on
the utility proxy. None of these magnitudes supports describing a reporter as
“highly correlated” with either target.

There is a narrower mechanistic signal in equal-episode Spearman estimates:

| condition | mean within-episode rho(V,W) | mean within-episode rho(V,y) |
|---|---:|---:|
| SFT-W | 0.1506 | 0.2793 |
| RL-QA | 0.1447 | 0.2935 |
| Hybrid-0.25 | 0.1580 | 0.2822 |
| Hybrid-1.0 | **0.1934** | **0.3199** |

Hybrid-1.0 minus RL-QA is +0.0486 for within-episode workspace Spearman, with
paired 95% CI `[+0.0118, +0.0785]`; its utility-Spearman difference is +0.0264
with CI `[-0.0095, +0.0559]`. This is consistent with stronger workspace
shaping increasing within-context W alignment without a resolved utility loss,
but it is a diagnostic result and cannot select lambda.

Source heterogeneity is material. For Hybrid-1.0, pooled utility Spearman is
0.432 on Evoked, 0.328 on Explicit, and -0.022 on Evoked-G2. Aggregate claims
must retain this breakdown. Also, `y_utility` here is a load-bearing-label
proxy, not a candidate-level counterfactual QA effect.

## Required decision before OOD

The coefficient lock remains `unresolved`. The recommended pre-OOD amendment
is: when the surviving configurations are exactly tied on ID QA and workspace
reward, choose the smaller lambda to minimize shaping intervention. That rule
would lock lambda_W=0.25 at best-step-200. It must be explicitly approved
before running the one-shot Gate-C OOD evaluation.

Artifacts:

- `stage_c_coefficient_id_decision.json`
- `stage_c_id_reporter_alignment_all_coefficients_retry1.json`
- `stage_c_id_reporter_alignment_validation_retry1.json`
