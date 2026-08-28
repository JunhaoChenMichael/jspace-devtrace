# Remember What You Thought, Not What You Said

Experiment repository for the paper *"Remember What You Thought, Not What You
Said: Workspace-Gated Episodic Memory for LLMs"* (under review).

## Layout

```
src/                  All code
  jlens.py            Workspace lens: residual capture + logit-lens readouts (core library)
  patch.py            Causal interventions: input-side concept steering (core library)
  metrics.py          Legacy trajectory metrics
  generation/         Benchmark generation (produces data/benchmarks/)
    gen_battery.py        Explicit benchmark generator
    gen_battery_v2.py     Evoked (one-hop silent bridge)
    gen_battery_v3.py     Compositional (two-hop, decoupled answer)
    gen_battery_v4.py     Decoupled (silent bridge + decoupled answer + anchor-free probe)
    gen_battery_vlm.py    Multimodal landmark benchmark (+ make_vlm_neutral.py)
    dedup_battery.py      Global concept de-duplication / validity filtering
  experiments/        Measurement and intervention experiments (produce data/results/)
    measure.py            Workspace (W_rr) + verbal (V, V_raw) salience per item
    downstream.py         Budget-k recall QA with policies, per-episode logs, McNemar
    causal_swap.py        Real-vs-sham workspace steering (independent pairs)
    replay_readout.py     Rehearsal availability member (probe-blind rollouts)
    pul_readout.py        Leverage member (utility-gradient on input-side directions)
    ignition_readout.py   Broadcast member (layer x position decodability grid)
    futurelens_exp.py     Tuned future-lens negative control
    finetune_metacog.py   Metacognitive alignment (distill W into V)
    train_memory_rl.py    Workspace-SFT, RL-W, RL-QA, and hybrid LoRA training
    evaluate_memory_rl.py Unified held-out admission/recall evaluation
    general_eval.py       MMLU/ARC/GSM8K no-harm checks (full model or LoRA)
    vprobe_robust.py      Verbal-probe paraphrase robustness
    vrating_baseline.py   Canonical 1-10 importance-rating baseline
    measure_vlm.py        VLM workspace/verbal measurement (image ablations)
  analysis/           Statistics and audits (read data/results/)
    analyze.py            AUC + bootstrap CIs + paired diffs (core helpers)
    summary.py            Master AUC tables across checkpoints
    master_table.py       Episode-cluster bootstrap + Bonferroni master table
    confound.py           Surface-presence controls
    embed_auc.py          Embedding-relevance predictive baseline
    regrade.py            Strict re-grading robustness check
    memory_rl_gates.py    Seed aggregation and predeclared RL Gate A/B/C report
  memory_rl/          Validated RL data, policy, recall, and objective modules
  paper_assets/
    make_paper_assets.py  Regenerates EVERY figure and table in paper/ from raw JSONs
  legacy/             Pre-pilot exploratory scripts (kept for provenance)
data/
  benchmarks/         All benchmark JSONs (+ vlm_images/); internal ids: v1=Explicit,
                      v2=Evoked, v3=Compositional, v4=Decoupled, v4xl=Decoupled-L
  results/            Per-item measurements, per-episode downstream records,
                      causal runs, fine-tuned checkpoint (olmo1b-metacog/)
logs/                 Run logs for every experiment
scripts/              Experiment orchestration, including run_memory_rl_mvp.sh
docs/                 Results/theory docs plus RL_EXPERIMENTS.md runbook
paper/                LaTeX source (own git repository, Overleaf-ready)
```

## Reproducing the paper

```bash
conda activate jspace
python src/paper_assets/make_paper_assets.py   # all figures + tables from raw JSONs
cd paper && tectonic main.tex                  # compile
```

Experiments require the benchmark JSONs (in `data/benchmarks/`) and open-weight
models (Qwen2.5 0.5B-7B, OLMo-2 1B/7B + stage checkpoints, Mistral-7B,
Qwen2-VL-2B); see `scripts/` for the full measurement campaigns.

## Workspace-guided RL extension

The staged Workspace-SFT, RL-W, RL-QA, and hybrid implementation is documented
in [`docs/RL_EXPERIMENTS.md`](docs/RL_EXPERIMENTS.md). Start with a data-only
check; later stages are intentionally never launched automatically across a
decision gate:

```bash
bash scripts/run_memory_rl_mvp.sh dry-run rl-w
```
