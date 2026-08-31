# Memory-RL campaign decision ledger

## Pre-OOD Stage-A lock

- Data split: fixed seed 0, manifest `1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f`.
- Primary model/teacher: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Primary Stage-A comparison: matched rank-continuous SFT-W versus RL-W, seed 0, budget 2, 300 steps, LoRA rank 32, LR 1e-6, RL beta 0.03.
- OOD evaluation has not been inspected for either trained adapter at lock time.

## Controlled temperature deviation

The README starts at temperature 0.7 but explicitly requires increasing sampling
temperature when group reward variance is near zero. The 7B one-step canary at
temperature 0.7 sampled one action for the whole G=8 group, giving reward std 0
and gradient 0. A pre-OOD calibration on 256 candidates from the ID training
split found an expected mixed-group rate of 0.105 at temperature 0.7 and 0.637
at temperature 5.0. Temperature 5.0 was the lowest tested value meeting the
predeclared >=0.5 mean and >=0.5 candidate-coverage thresholds. A subsequent
10-step 7B canary had finite, nonzero gradients and no diagnostic warnings.

Temperature is therefore locked to 5.0 for the Stage-A comparison. It is set in
both condition configs, although it is operationally unused by SFT-W. No OOD
data informed this choice.

## RL-W seed-0 ID diagnostic decision

The locked beta-0.03 run completed all 300 steps and passed the formal artifact
validator. Its ID AUC moved from 0.62658 to a best value of 0.63561 at step 200;
the locked adapter is therefore `best-step-200`, not the final-step adapter.

The low-variance warning was audited before inspecting OOD data. Across all 300
groups, 174 (58.0%) contained both actions, 157 (52.3%) had nonzero reward
variance, and 126 (42.0%) were all-Yes or all-No. The mean/median gradient norms
were 5.754/2.767 with no zero or non-finite gradients. Mean KL was 0.000586, so
beta times mean KL was about 0.0000176 and did not dominate reward. Validation
Yes rate changed from 0.7953 to 0.7814 at the selected checkpoint rather than
collapsing toward an extreme.

Consequently none of the predeclared adjustment triggers holds: the policy did
update, Yes/No collapse did not occur in most batches, and training did not rise
while validation fell. The weak/noisy reward trend is retained as an AMBER
diagnostic, but beta remains 0.03 and LR remains 1e-6. No extra scientific
configuration is launched before the first locked OOD evaluation.

## Gate A outcome

The locked Decoupled evaluation gave SFT-W AUC 0.52715 and RL-W AUC
0.49331. Thus `RL-W - SFT-W = -0.03385`, with paired episode-cluster 95% CI
`[-0.04871, -0.02048]`. This is below the predeclared -0.03 boundary, so Gate A
is `worse`.

No-harm checks did not fire: RL-W full-context accuracy changed by -0.01471 and
fresh Decoupled W_rr AUC changed by +0.00008 versus the matched base. The
scientific failure is therefore the admission comparison against SFT-W, not a
detected latent-workspace or full-context regression.

Per protocol, seeds 1 and 2 are not launched, beta is not retuned using OOD,
and execution stops for manual review before any RL-QA or Hybrid run.

## Gate B0 outcome

The no-training QA reward preflight evaluated all 175 sealed ID-training
episodes at budget 2 with 16 policy draws per episode. Temperature was selected
before QA generation, using only exact-set diversity: 5.0 was the lowest tested
value whose median number of unique sets reached 4 (the observed median was 5).

At the locked temperature, 103/175 groups (58.86%) had mixed binary QA reward
and 173/175 groups (98.86%) contained at least two distinct exact-k sets. Gate
B0 is therefore `GREEN` under the preregistered G=16 thresholds. Every source
also passed independently. The operational oracle/exploitable rate was 152/175
(86.86%), and sampled QA accuracy was 0.76720 when at least one load-bearing
item was retained versus 0.29270 otherwise.

The fixed-halves G=8 sensitivity check retained a 50.00% mixed-reward rate but
had median set diversity 3, so that secondary diagnostic is `AMBER` on
diversity only. It does not change the primary G=16 Gate decision; it requires
close diversity monitoring in Stage B1.

The strict artifact validator passed with no errors or warnings. This run made
no optimizer, gradient update, adapter, or checkpoint. Execution stops here for
manual review; RL-QA and Hybrid have not been launched.

## RL-QA seed-0 pre-OOD lock

Gate B0 was manually approved. The only launched Stage-B1 training condition
used seed 0, budget 2, G=8, temperature 5.0, beta 0.03, LR 1e-6, rank-32 LoRA,
lambda_QA=1, and lambda_W=0 for 300 steps. It used the sealed 175/45 ID split,
the locked Qwen2.5-7B revision, adapter-disabled frozen recall, and no workspace
term in the optimized reward.

The run completed all 300 steps and passed the formal artifact validator with
no errors or warnings. Across training, 163/300 groups (54.33%) had mixed QA
reward, median/mean unique selected sets were 3/3.38, and only 12/300 groups
(4.00%) sampled one unique set. Mean exact-set KL was 0.03052, so beta times
mean KL was 0.000916; there was no sparse-reward or selection-collapse trigger.

ID validation results were:

```text
step       QA accuracy   containment   workspace set reward
baseline      0.68889       0.71111                0.56481
100           0.68889       0.68889                0.55093
200           0.71111       0.77778                0.58148
300           0.71111       0.77778                0.58426
```

Steps 200 and 300 tie for the highest trained-checkpoint ID QA accuracy. The
predeclared first-maximum rule therefore locks `best-step-200`; the final-step
adapter is not selected. No Stage-B1 OOD result had been inspected at this
lock point, and OOD will not be used to change the checkpoint or recipe.

## Gate B1 outcome

The locked step-200 checkpoint was evaluated once on Decoupled and
Compositional. A technical audit then found that bf16 greedy recall answers can
change with unrelated prompt composition in a generation batch even when the
selected memory set is identical. The original batch-8 artifact was preserved,
and a controlled batch-size-1 retry was run to isolate every selected-set
prompt. This changes no model, checkpoint, selection, grader, answer budget, or
scientific hyperparameter.

Under the isolated-decoding primary, Decoupled QA was 0.42647 for RL-QA versus
0.33824 for Original, a +0.08824 difference with paired episode bootstrap 95%
CI `[+0.02941, +0.16176]` and exact McNemar p=0.03125. RL-QA was nominally
+0.01471 above SFT-W, with CI `[-0.02941, +0.05919]` and p=1.0; it is therefore
not clearly worse than SFT-W, but this seed does not show a resolved QA
advantage over SFT-W. The preserved batch-8 evaluation also passes the effect
threshold, with RL-QA minus Original = +0.05882, so Gate direction is stable to
the audited batching sensitivity.

RL-QA also improved Decoupled pooled AUC by +0.08716 versus Original and
+0.04591 versus SFT-W; both 4,000-draw episode-cluster intervals exclude zero.
On Compositional, QA improved by +0.05769 versus Original and tied SFT-W, while
containment and AUC improved more strongly. This is neither a sparse-reward nor
a selection-only outcome.

No-harm checks did not fire. Adapter-enabled full-context QA changed by 0 on
both batteries, and fresh Decoupled W_rr changed by +0.00088 versus the matched
base, with paired 95% CI `[-0.00169, +0.00377]`.

Gate B1 is therefore `GREEN`: downstream QA alone provides a viable admission
learning objective, although superiority to SFT-W on downstream QA is not yet
established. Per the manual Gate requirement, no Hybrid or extra-seed run is
launched. The proposed next configuration, if approved, is seed 0 / k=2 Hybrid
with lambda_QA=1 and lambda_W=0.5, followed only by the predeclared ID-side
coefficient checks before any new OOD evaluation.

## Stage C lambda-0.5 authorization and pre-OOD lock

Stage C was manually approved for exactly one initial condition: seed 0,
budget 2, lambda_QA=1, and lambda_W=0.5. All remaining B1 protocol fields stay
fixed, including Qwen2.5-7B revision, split seed 0, G=8, temperature 5, LR
1e-6, beta 0.03, rank-32 LoRA, bf16, 300 steps, and ID evaluations at steps
100/200/300. No B1 seed expansion, Qwen3/OLMo run, or RL-W revisit is
authorized.

Before launch, the ID validation schema is extended with mechanistic reporter
diagnostics. `V_RL` is adapter-enabled P(Yes) on the probe-blind admission
prompt at temperature 1; `W_ref` is immutable raw W_rr; and `y_utility` is the
binary load-bearing label. Pearson and tie-aware Spearman correlations are
recorded pooled, equal-episode, and by source, with 4,000 source-stratified
episode-cluster bootstrap draws. Candidate rows are retained for later paired
SFT-W/RL-QA/Hybrid recomputation. These correlations are diagnostics only and
are explicitly forbidden from checkpoint or coefficient selection.

The coefficient rule remains unchanged: choose highest ID QA; if QA is within
one percentage point, use workspace set reward; an exact tie favors the
default lambda_W=0.5. The lambda-0.5 run will stop at an ID-only review before
any OOD evaluation or decision about lambda_W=0.25/1.0.

## Stage C lambda-0.5 ID review and lambda-0.25 authorization

The lambda_W=0.5 run completed 300/300 steps and passed the strict formal
validator with no errors or warnings. Its predeclared strict-first-maximum
checkpoint is step 100: ID QA is 32/45 (0.71111), containment is 0.73333, and
workspace set reward is 0.57037. The checkpoint convention remains the B1
strict-first-maximum rule; Stage C's workspace tie-break applies between
coefficient configurations, not retrospectively between checkpoints within a
run. Reporter correlations are diagnostic and cannot change either choice.

The locked RL-QA checkpoint also has ID QA 32/45, while its workspace set
reward is 0.58148. Thus lambda_W=0.5 does not yet establish an ID advantage for
shaping and cannot be locked before testing the next predeclared coefficient.
Training health is adequate: 55.33% of groups had mixed QA reward, 96.33% had
nonzero combined-reward variance, median unique sets was 3, all gradient norms
were finite and nonzero, and the strict validator reported no warning.

The next authorized run is therefore only seed 0, budget 2, lambda_QA=1,
lambda_W=0.25 with every other B1/Stage-C field unchanged. It remains under the
ID-only embargo: no OOD evaluation, extra seed, alternate model, or low-weight
0.1/0.05 run is permitted. Whether lambda_W=1.0 is still needed will be decided
from the lambda_W=0.25 ID result using the predeclared coefficient rule.

## Stage C lambda-0.25 ID review and lambda-1.0 authorization

The lambda_W=0.25 run completed 300/300 steps and passed the exact-lock formal
validator with no errors or warnings. Its strict-first-maximum checkpoint is
step 200, with ID QA 32/45 (0.71111), containment 0.77778, and workspace set
reward 0.58426. It is the provisional winner among tested Hybrid coefficients:
QA ties lambda_W=0.5, and the predeclared workspace tie-break is higher than
lambda_W=0.5's 0.57037.

Lambda_W=1.0 remains necessary because it is the third predeclared base
coefficient, not an OOD-informed extension. If its ID QA is at least 33/45 it
wins on the primary metric; if it is 32/45 its workspace reward must enter the
tie-break; and at 31/45 or below it is more than one percentage point behind
the provisional best. Different-lambda combined rewards are not comparable,
and containment, verbal AUC, or reporter correlations cannot replace the
QA-then-workspace rule.

Only seed 0, budget 2, lambda_QA=1, lambda_W=1.0 is authorized next. The OOD,
extra-seed, alternate-model, and lambda_W=0.1/0.05 embargo remains in force.

## Stage C coefficient grid outcome: unresolved ID tie

Lambda_W=1.0 completed 300/300 steps and passed the exact-lock formal
validator without errors or warnings. Its strict-first-maximum checkpoint is
step 200. All three Hybrid configurations have ID QA 32/45. At the workspace
tie-break tier, lambda_W=0.25 and lambda_W=1.0 both have exactly 0.5842592593,
while lambda_W=0.5 has 0.5703703704 and is eliminated.

The final surviving set is therefore `{0.25, 1.0}`. The rule preferring the
default 0.5 on a complete tie cannot revive a configuration already eliminated
at the workspace tier, and the predeclared rule contains no ordering between
0.25 and 1.0. The coefficient lock is recorded as `unresolved`. Combined
reward, containment, verbal AUC, reporter correlation, and all OOD results are
forbidden as improvised tie-breaks. OOD and seed expansion remain paused until
an explicit pre-OOD tie-break amendment selects one surviving coefficient.

## Stage C fixed-ID reporter alignment

A first unified-scorer attempt stopped before scoring because the plain base
model's PEFT compatibility method raises when no adapter is loaded. It wrote no
artifact and accessed no OOD data. The minimal compatibility fix treats that
specific base-model state as the expected empty adapter set and adds a
regression test. Per campaign convention, the successful rerun uses `_retry1`.

The retry rescored Original plus the five locked adapters on the same 45 ID
episodes / 215 candidates using bf16 admission batch size 1. It verified model
revision, manifest, canonical condition configs, checkpoint/summary/training
state, adapter activation, and identical candidate/prompt/token ordering. An
independent validator exactly recomputed all six 4,000-draw summaries and all
15 paired comparisons; status is PASS with zero errors.

The hypothesized clean SFT-W/RL-QA/Hybrid correlation pattern is not present in
pooled ID statistics. Equal-episode ranks provide a narrower signal:
Hybrid-1.0 has higher within-episode workspace Spearman than RL-QA by 0.04865
with paired 95% CI [0.01183, 0.07853], while its utility-Spearman difference is
0.02641 with CI [-0.00951, 0.05591]. These remain mechanism diagnostics and do
not resolve the coefficient tie. The OOD embargo is unchanged.

## Stage C pre-OOD tie-break amendment and coefficient lock

The campaign continuation README approved a pre-OOD amendment for the exact
tie between the surviving lambda_W values 0.25 and 1.0: select the smaller
coefficient to minimize shaping intervention while preserving the primary ID
QA objective. The amendment was recorded before any completed Stage-C OOD
artifact existed and does not use Decoupled or Compositional performance.

Stage C is therefore locked to Hybrid lambda_QA=1, lambda_W=0.25, seed 0,
budget 2, best-step-200. Reporter correlations were not used. Only the one-shot
Decoupled and Compositional evaluation is authorized; lambda_W=0.5/1.0 OOD,
coefficient reopening, seed expansion, and model scaling remain forbidden.
The machine-readable lock is `stage_c_pre_ood_tiebreak_lock.json`.

## Stage C one-shot OOD decision

The locked Hybrid-0.25 seed-0 checkpoint was evaluated once on source-isolated
Decoupled and Compositional runs under batch-1 greedy QA. The independent raw
recomputation validator passed with zero errors.

On primary Decoupled, Hybrid and RL-QA are exactly tied in QA (29/68), top-2
containment (36/68), and both exploitable-subset endpoints. Hybrid has a small
pooled-AUC increase of 0.00606 with paired 95% CI [0.00011, 0.01353], but that
ranking-only shift produces no top-2 or downstream gain. Compositional adds one
Hybrid-correct answer but has unchanged containment and unresolved AUC.

The decision is **C3: NO ADDED VALUE**. RL-QA remains the leading RL method.
No seed expansion, alternate Hybrid OOD, coefficient reopening, or model-scale
follow-up was launched; execution stopped for manual review.

## RL-QA seed-expansion authorization

Manual review accepted Stage C C3 and froze Hybrid development. The next
question is limited to whether RL-QA reproduces across seeds on
Qwen2.5-7B. Seeds 1 and 2 at budget 2 are authorized with every seed-0 B1
training and OOD parameter held fixed. Checkpoints remain ID-only selections.

No Hybrid run, hyperparameter change, alternate model, or scale/cross-family
experiment is authorized. Three-seed reporting will retain each seed, mean,
sample standard deviation, paired effects versus Original and the locked SFT-W
baseline, and full-context plus Decoupled workspace no-harm checks.

## RL-QA seed-expansion source audit

The seed-0 run ended at 00:48 UTC. Local session provenance reconstructs every
subsequent trainer/runner patch: strict finite-JSON handling plus the previously
approved fixed-ID reporter-correlation instrumentation. The latter runs only
during validation, uses an independent local NumPy bootstrap generator, and is
hard-coded as excluded from checkpoint selection. Reward, rollout sampling,
Torch RNG, GRPO/KL loss, optimizer, scheduler, and the strict-first-maximum ID
QA checkpoint rule did not change.

The only prompt-helper change adds a probe-leak assertion; all 1,040 candidates
in the locked 220-episode bundle have candidate context exactly equal to their
episode context. The expansion therefore proceeds under an explicitly recorded
diagnostic-only source delta. Scientific training semantics remain locked, but
seed-1/2 artifacts are not claimed to be byte-identical to seed 0 because they
also contain `reporter_correlations.jsonl`. Preflight tests pass: 108/108.

## RL-QA seed-expansion training outcome and pre-OOD lock

Seeds 1 and 2 completed the locked 300-step recipe and the independent training
audit passed. Seed 1 has ID QA 31/45 at steps 100, 200, and 300, so the strict
first-maximum checkpoint is step 100. Seed 2 reaches its first maximum of 32/45
at step 300. Seed 0 remains locked at step 200 with 32/45.

Mixed-QA reward occurred in 148/300 groups for seed 1 and 146/300 for seed 2,
just below the 50% diagnostic line; both are recorded as AMBER low-variance
runs. Their median unique-set count is 3, single-set groups are only 4.0% and
3.67%, and gradients are nonzero in 100% and 99.67% of groups. This does not
support an implementation-failure or diversity-collapse diagnosis. It also does
not authorize tuning, extending training, or adding seeds.

Before any new OOD inference, the exact four adapters, source-isolated condition
order, batch sizes, B=4,000 statistics, hashes, and no-harm protocol were sealed
in `stage_b1_rl_qa_seed_expansion_pre_ood_lock.json`. No completed seed-expansion
OOD artifact existed at lock time.

## RL-QA three-seed reproducibility decision

The locked source-isolated Decoupled and Compositional evaluations completed,
and the pre-OOD analysis plus independent recomputation validator passed.
Relative to Original, all three RL-QA seeds improve both QA and admission AUC
on both batteries. Decoupled RL-QA QA is 42.65%, 39.71%, and 41.18%, versus
Original at 33.82%; its three-seed mean is 41.18% with sample std 1.47pp. The
corresponding AUCs are 0.57306, 0.51969, and 0.56686, with mean 0.55321 and
sample std 0.02919 versus Original at 0.48590. Compositional QA is 44.23% in
all three seeds versus Original at 38.46%, and every seed also has positive
AUC delta.

The comparison with fixed SFT-W remains a tie downstream. Decoupled QA deltas
are +1.47pp, -1.47pp, and 0; Compositional QA is exactly tied for every seed.
AUC is higher on average but seed 1 reverses direction on both batteries.
Therefore the accepted claim is cross-seed reproducibility versus Original,
not reproducible downstream superiority over SFT-W.

No no-harm trigger fired. Decoupled W_rr AUC seed-minus-base deltas are
+0.00088, +0.00063, and +0.00058; no full-context decline exceeds 2pp. A
shared-condition audit also exactly reproduced the prior sealed admission,
selection, QA, references, and comparable full-context records. Its first run
exposed an audit-only ordering assumption and is preserved; the corrected
`_retry1` report passes with zero issues.

The Stage-B expansion is recorded as **PASS against Original / TIE against
SFT-W downstream**. RL-QA is sufficiently reproducible for a qualified main
text result on Qwen2.5-7B. Hybrid remains C3 with no added value. Per scope, no
Hybrid expansion, coefficient reopening, k=3 run, Qwen3/OLMo run, or
cross-family/scale replication is launched automatically. The consolidated
report is `STAGE_B1_RL_QA_THREE_SEED_REPORT.md`.
