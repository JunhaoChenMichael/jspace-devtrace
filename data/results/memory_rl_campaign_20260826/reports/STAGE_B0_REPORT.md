# Stage B0 report — QA reward preflight

## Decision

**Gate B0: `GREEN`.** With the preregistered G=16 primary grouping, 103/175
episode groups (58.86%) had mixed binary QA reward and the median number of
unique exact-k sets was 5. Both exceed the GREEN thresholds of 40% and 4.

This was a no-training run. It created no optimizer, gradient update, adapter,
or checkpoint. Per the campaign's manual-gate rule, RL-QA and Hybrid were not
started automatically.

## Locked configuration and isolation

- Policy/recall model: `Qwen/Qwen2.5-7B-Instruct`, immutable revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Sealed data split: seed 0, 175 train / 45 validation episodes, manifest
  `1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f`.
- Initial admission policy: fresh zero-initialized rank-32 LoRA, no training,
  dropout disabled, bf16, exact budget k=2, 16 samples per episode.
- Frozen reward environment: same adapter-disabled 7B model, greedy decoding,
  64 answer tokens, and the same lightweight grader as Gate A.
- All 825 admission prompts were constructed and hashed from only `context`
  and `candidate.concept`. Runtime leakage checks passed before any probe/gold
  reward prompt was constructed.

## Temperature lock

Temperature was chosen using selection diversity only, before QA generation and
without OOD data. Common episode-derived random streams were used for every
candidate.

| Temperature | Median unique sets / 16 | Mean unique | Groups with >=2 sets |
|---:|---:|---:|---:|
| 0.7 | 1 | 1.583 | 45.14% |
| 1.0 | 2 | 1.823 | 57.14% |
| 2.0 | 3 | 2.680 | 83.43% |
| 3.0 | 3 | 3.400 | 93.14% |
| **5.0** | **5** | **4.657** | **98.86%** |

The locked rule selected 5.0 as the lowest tested temperature with median
unique sets >=4. QA reward was not used to choose it.

## Primary G=16 results

| Diagnostic | Result |
|---|---:|
| Logical samples | 2,800 |
| Distinct episode/set continuations | 815 |
| Mixed-QA groups | 103/175 (58.86%) |
| Constant-zero / constant-one groups | 29 / 43 |
| Groups with QA std > 0 | 103/175 (58.86%) |
| Mean within-group QA std | 0.23637 |
| Mean sampled QA reward | 0.54893 |
| Mixed-containment groups | 132/175 (75.43%) |
| Median unique selected sets | 5 |

Across all policy draws:

```text
P(QA correct | at least one load-bearing item retained)     = 0.76720
P(QA correct | no load-bearing item retained)               = 0.29270
difference                                                   = +0.47449
```

The operational exploitable fraction was 152/175 (86.86%): the label-selected
oracle memory produced a correct answer. The stricter secondary fraction where
oracle was correct and no-memory was incorrect was 125/175 (71.43%). Reference
accuracies were oracle 86.86%, full-context 80.57%, and no-memory 16.00%; these
are operational references and are not assumed to obey a monotonic ceiling.

Failure attribution over the 2,800 logical samples was 1,288 selection failures
(46.00%), 352 recall/composition failures after retaining a load-bearing item
(12.57%), and 1,160 successes (41.43%).

## Source sensitivity

| Source | Episodes | Mixed QA | Median unique | P(correct\|retained) | P(correct\|not) | Difference | Oracle QA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Evoked | 60 | 61.67% | 4.5 | 0.99788 | 0.35246 | +0.64542 | 0.98333 |
| Evoked-G2 | 45 | 55.56% | 5.0 | 0.92523 | 0.36364 | +0.56160 | 0.95556 |
| Explicit | 70 | 58.57% | 4.0 | 0.59443 | 0.07143 | +0.52300 | 0.71429 |

Every source independently satisfies the G=16 GREEN thresholds, so the pooled
result is not caused only by the easiest source.

## G=8 sensitivity for Stage B1

Stage B1 trains with G=8, whereas the requested B0 primary gate uses G=16. The
16 draws were therefore split into two fixed, non-overlapping halves per
episode:

```text
mixed QA groups       = 175/350 = 50.00%
median unique sets    = 3
mean unique sets      = 3.56
mean QA reward std    = 0.21203
sensitivity status    = AMBER (diversity threshold only)
```

The mixed-reward signal remains above 40% at G=8, but median set diversity is
three rather than four. This does not replace the preregistered GREEN G=16
decision; it means B1 should keep temperature 5.0 locked and monitor diversity
closely rather than launch a large QA-only sweep.

## Validation and implementation notes

- Strict validator: `pass`, with 175 groups, 2,800 samples, 175 references,
  exact-k reconstruction, probability/log-probability recomputation, prompt
  provenance, grader recomputation, duplicate-set determinism, source scope,
  summary recomputation, hashes, model revision, and split manifest checked.
- Full repository test suite after the B0 implementation: `75 passed`.
- Two 1-episode canary attempts stopped before scoring/generation because the
  tokenizer revision guard mishandled cache symlinks. The third retry and formal
  run use the corrected snapshot-path audit. A separate validator false failure
  from a `1e-8` float32 probability tolerance was corrected to `1e-6` and covered
  by regression tests. Failed outputs were retained; no scientific result was
  overwritten.

## Next manual decision

The continuation plan's GREEN branch authorizes a seed-0, k=2 RL-QA run at the
locked temperature 5.0, followed by Hybrid. The existing Gate A RL-W checkpoint
remains a frozen negative baseline; no RL-W expansion or OOD-driven retuning is
authorized.
