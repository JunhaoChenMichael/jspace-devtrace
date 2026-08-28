# Qwen3-8B M0 Yes-rate audit

Status: **PASS**

The M0 gate defines Yes-rate as the fraction of continuous verbal scores `V`
that satisfy `V >= 0.5`. `V` is the constrained probability of Yes relative to
the fixed Yes/No token sets. It is not a greedy-text parsing statistic. The
measurement prompt, token sets, scoring, and threshold are unchanged.

| Condition | Min | P25 | Median | Mean | P75 | Max | Yes-rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit | 0.000159752 | 0.000968240 | 0.001634383 | 0.002328651 | 0.002964029 | 0.013450768 | 0.000000000 |
| Evoked | 0.000212770 | 0.000900028 | 0.001475075 | 0.002055940 | 0.002607903 | 0.009118417 | 0.000000000 |
| Decoupled | 0.000173863 | 0.001248864 | 0.002170548 | 0.003429814 | 0.003599096 | 0.065443130 | 0.000000000 |
| Compositional | 0.000157455 | 0.001004887 | 0.001581397 | 0.002288962 | 0.002778673 | 0.010667963 | 0.000000000 |

All scores are finite and continuous. The zero Yes-rate is therefore a
thresholding property, while the continuous-score Decoupled AUC still
reproduces the preregistered baseline. No M0 rerun or scoring change is
required by this audit.
