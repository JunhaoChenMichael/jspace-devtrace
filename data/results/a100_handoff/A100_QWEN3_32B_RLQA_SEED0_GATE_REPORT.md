# Qwen3-32B RL-QA seed-0 scaling gate

Report schema: `rlqa-32b-seed0-gate/v1`. Decision: **FAIL**.

`automatic_seed_expansion_authorized = false`. Seeds 1/2 require a human release note; this report authorises nothing on its own.

## 1. Gate

| Check | Result |
|---|---|
| Decoupled QA delta >= +5.00 pp | FAIL |
| Decoupled QA delta > 0 | FAIL |
| Decoupled admission AUC delta > 0 | PASS |
| admission AUC 95% CI lower bound > 0 | FAIL |
| full-context QA drop <= 2.00 pp | PASS |
| fresh W_rr drop <= 0.03 | PASS |
| Compositional: no secondary harm alert | PASS |
| provenance / leakage / lock integrity | PASS |

## 2. Decoupled (primary)

| Quantity | Value |
|---|---:|
| QA accuracy, Original | 0.1176 |
| QA accuracy, RL-QA | 0.1176 |
| **QA delta** | **+0.00 pp** |
| Admission AUC delta | +0.04483 |
| Admission AUC 95% CI | [-0.01378, +0.10045] |
| Exact McNemar p | 1 |
| discordant (adapter-only / original-only) | 3 / 3 |
| Full-context QA drop | +0.00 pp |
| Fresh W_rr drop | -0.00008 |

## 3. Compositional (mandatory diagnostic)

QA delta +0.00 pp, admission AUC delta -0.01629. Diagnostics cannot rescue Decoupled; they can only cap the verdict at AMBER.

## 4. Comparison with the completed 8B replication

| | Qwen3-8B (3 seeds) | Qwen3-32B (seed 0) |
|---|---:|---:|
| Decoupled QA delta | +8.82 pp | +0.00 pp |
| Admission AUC, Original | 0.3415 | 0.6571 |
| Admission AUC delta | +0.49894 | +0.04483 |

The 32B starting point is much stronger, so an 8B-sized AUC gain is arithmetically unavailable; headroom, not method quality, sets the ceiling.

## 4b. In-distribution learning vs out-of-distribution transfer

| step | ID QA | ID containment | ID verbal AUC | yes rate |
|---:|---:|---:|---:|---:|
| 0 | 0.4667 | 0.4667 | 0.4916 | 0.991 |
| 100 | 0.7778 | 0.8889 | 0.5852 | 1.000 |  **<- locked**
| 200 | 0.7778 | 0.8444 | 0.6549 | 0.763 |
| 300 | 0.7778 | 0.8667 | 0.6630 | 0.693 |

Training worked in distribution: ID QA moved 0.4667 -> 0.7778. The failure is transfer, not optimisation. The 8B campaign gained a comparable amount in distribution and that gain did reach Decoupled; here it does not.

## 5. Frozen recipe

```json
{
  "answer_tokens": 64,
  "beta": 0.03,
  "budget": 2,
  "checkpoint_selection": "strict first maximum of ID QA on the fixed ID validation split",
  "dtype": "bfloat16",
  "eval_every": 100,
  "group_size": 8,
  "grpo_epochs": 2,
  "lambda_qa": 1.0,
  "lambda_w": 0.0,
  "learning_rate": 1e-06,
  "lora_rank": 32,
  "max_length": 2048,
  "max_steps": 300,
  "optimisation_seeds": [
    0
  ],
  "save_every": 100,
  "split_seed": 0,
  "temperature": 5.0
}
```

## 6. Pre-OOD lock

- Selected step: **100** (strict first maximum of ID QA)
- Checkpoint tree SHA-256: `38ba39d13ed4240442ed3d709298ed994478fee322e1b8ff835538b21090b829`
- Lock manifest self-hash: `5c9e7d451a453631737eb16a7a512e37c3ba25c0c2adab9249d53fc7833b81b8`

## 7. Stop

Seed 0 is complete and the campaign stops here for human review. No seed expansion, no RL-W, no Hybrid, no larger model, and no combined objective were launched.
