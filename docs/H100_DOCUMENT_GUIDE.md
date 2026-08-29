# H100 server documentation guide

This is the orientation page for the person receiving this repository on an
H100 server. Read this page before copying data, implementing launchers, or
allocating a GPU. In this repository, operators sometimes call every Markdown
handoff file a "README"; the list below covers both the actual `README.md`
files and the operational documents under `docs/`.

## The short version

Read these documents in this order:

1. [`README.md`](../README.md) — repository map and verified model/data download
   locations.
2. [`H100_NEXT_CAMPAIGNS.md`](H100_NEXT_CAMPAIGNS.md) — the controlling H100
   specification for the two approved tracks.
3. [`METACOG_ALIGNMENT_CAMPAIGN.md`](METACOG_ALIGNMENT_CAMPAIGN.md) — the
   completed A5000 Binary Metacognitive Alignment contract that Track B must
   preserve and extend to three fresh H100 seeds.
4. [`RL_EXPERIMENTS.md`](RL_EXPERIMENTS.md) — the completed Memory-RL machinery
   and gate semantics that Track A reuses for Qwen3-8B RL-QA.
5. Print and review both new H100 command plans, run tests and engineering
   canaries, then obtain explicit operator approval before formal GPU use.

`H100_NEXT_CAMPAIGNS.md` takes precedence for H100 scope. Neither of the older
runbooks is permission to launch its historical campaign unchanged.

## What each document is for

| Document | Audience and purpose | H100 operator action |
|---|---|---|
| [`README.md`](../README.md) | First landing page: code/data layout, dependency context, exact Hugging Face IDs, fixed Qwen3-8B revision, and dataset provenance. | Use it to clone the repository and obtain verified upstream models/data. Do not expect project-trained adapters or checkpoints to come from Hugging Face. |
| [`docs/H100_NEXT_CAMPAIGNS.md`](H100_NEXT_CAMPAIGNS.md) | **Authoritative H100 handoff.** Defines Track A (Qwen3-8B RL-QA, seeds 0/1/2) and Track B (Qwen3-8B Binary Metacognitive Alignment, seeds 0/1/2), their locked recipes, gates, isolation, and deliverables. | Treat as the controlling specification. Implement H100-only launchers, print plans, preserve one-shot OOD rules, and stop for human review. |
| [`docs/METACOG_ALIGNMENT_CAMPAIGN.md`](METACOG_ALIGNMENT_CAMPAIGN.md) | Executable contract for the completed **A5000 seed-0** M0/M1 campaign: measurement gate, frozen teacher, canary, ID-only lock, one-shot OOD, artifact hashes, and report schema. | Reuse its scientific and fail-closed invariants when implementing H100 Track B. Do **not** run its A5000-only launcher on H100 or reuse the A5000 checkpoint in the H100 aggregate. |
| [`docs/RL_EXPERIMENTS.md`](RL_EXPERIMENTS.md) | Detailed Memory-RL MVP runbook: SFT-W, RL-W, RL-QA, Hybrid, leakage controls, reward preflight, evaluators, and manual gates. It records the machinery from which RL-QA was promoted. | For H100 Track A, reuse only the RL-QA definitions, ID/OOD separation, reward/diversity preflight, checkpoint selection, and reporting semantics named by the H100 spec. Do not rerun RL-W or Hybrid as part of this allocation. |
| [`docs/HANDOFF.md`](HANDOFF.md) | Historical full-project handoff: motivation, methods, early results, environment notes, commands, and the state of the project before the newer campaigns. | Consult when reconstructing provenance or understanding old artifact names. It is not the current H100 launch plan; newer campaign documents override stale TODOs or hardware assumptions. |
| [`docs/RESULTS.md`](RESULTS.md) | Chronological scientific result ledger across models, batteries, controls, VLM runs, and server campaigns. | Use to interpret expected baselines and investigate discrepancies. Never use historical OOD results to tune an H100 recipe. |
| [`docs/THEORY.md`](THEORY.md) | Definitions and interpretation of workspace availability, verbal availability, PUL, ignition, and metacognitive efficiency. | Use when checking that metrics and claims have not changed. It contains no H100 execution authority. |
| [`docs/RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md) | Broader research agenda and possible future experiments across development, reasoning, multimodality, capacity, and brain links. | Background only. Its suggested experiments are outside the approved two-track H100 scope unless separately authorized. |
| [`docs/PAPER_PLAN.md`](PAPER_PLAN.md) | Paper organization, claims/evidence matrix, figure plan, and citation plan. | Use for writing and final artifact expectations, not for choosing hyperparameters or launching jobs. |
| [`docs/NOVELTY.md`](NOVELTY.md) | Prior-work and novelty audit. | Literature/background reference only; no operational instructions. |
| [`paper/README.md`](../paper/README.md) | Minimal entry point for the paper source. | Relevant only when compiling or editing the manuscript after results are complete. |
| [`exemplar_paper/README.md`](../exemplar_paper/README.md) | Upstream conference LaTeX-template documentation. | Ignore for experiments; consult only for template formatting. |

`.pytest_cache/README.md` may appear after tests. It is generated by pytest,
is not project documentation, and must not be used as a handoff instruction.

## Which campaign am I operating?

### Track A: Qwen3-8B Memory-RL

Primary documents:

1. `H100_NEXT_CAMPAIGNS.md`, Track A, for the frozen H100 scope.
2. `RL_EXPERIMENTS.md` for the implementation/data/gate definitions.
3. Root `README.md` for the exact Qwen3-8B download revision.

The approved method is RL-QA only. The H100 operator must first add and test a
Qwen3-8B path, run the ID-train-only reward/diversity preflight, and freeze the
recipe before formal seeds 0/1/2. RL-W, Hybrid sweeps, OOD-driven temperature
selection, and combined RL+metacognitive training are out of scope.

### Track B: Qwen3-8B Binary Metacognitive Alignment

Primary documents:

1. `H100_NEXT_CAMPAIGNS.md`, Track B, for the three-seed H100 specification.
2. `METACOG_ALIGNMENT_CAMPAIGN.md` for the A5000-tested scientific contract and
   artifact validation rules.
3. Root `README.md` for the exact Qwen3-8B download revision.

All three seeds must be fresh H100 runs. Create all three ID-only checkpoint
locks before opening OOD for any seed. Each seed has one combined Decoupled +
Compositional OOD attempt. The old A5000 launcher rejects H100 by design; do not
weaken that check. Build a separate H100 launcher and separate output schema.

## Rules that apply to both tracks

- Model and tokenizer are `Qwen/Qwen3-8B` at commit
  `b968826d9c46dd6066d109eabc6255188de91218`; do not silently substitute a
  model or a moving revision.
- Use bf16, fresh output directories, immutable manifests, hashes, raw logs,
  and separate adapters/checkpoints for every track and seed.
- Use ID data only for training, hyperparameters, early stopping, and
  checkpoint selection. Decoupled and Compositional stay sealed until locks
  exist as specified by the controlling plan.
- Do not mix A5000 and H100 checkpoints in a three-seed aggregate.
- If a fail-closed validation or gate fails, preserve the run directory and
  report the cause. Do not lower a gate, overwrite the directory, or consume a
  second OOD attempt.
- A GREEN report is evidence for manual review, not permission to launch an
  extension, additional seed, larger model, or combined objective.

## Expected H100 handoff outputs

Track A must end with `H100_RL_QA_THREE_SEED_REPORT.md`. Track B must end with
`H100_QWEN3_BINARY_METACOG_THREE_SEED_REPORT.md`. Each handoff should also give
the operator command plan, environment and GPU inventory, model/data/source
hashes, per-seed run directories, selected checkpoint hashes, OOD-attempt
records, gate decisions, exceptions, and links to raw logs. Stop after both
reports for human review.
