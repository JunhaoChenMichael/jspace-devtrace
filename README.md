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

## Verified model and dataset downloads

The links and revisions below were checked against the Hugging Face API on
2026-08-29. Every listed model repository is public and ungated. For a strict
reproduction, use the recorded commit rather than a moving `main` branch.

### Current H100 campaigns

Both planned H100 tracks use
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) at the frozen revision
`b968826d9c46dd6066d109eabc6255188de91218`. This is the same revision used by
the A5000 campaign and was also the repository's current API SHA when checked.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
)
PY
```

### Paper-reproduction models

| Experiment family | Verified Hugging Face repository |
|---|---|
| GPT-2 | [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) |
| Qwen2.5 base | [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B), [`1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B), [`3B`](https://huggingface.co/Qwen/Qwen2.5-3B), [`7B`](https://huggingface.co/Qwen/Qwen2.5-7B) |
| Qwen2.5 instruct | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [`1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [`3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [`7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Qwen3 | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), [`1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), [`4B`](https://huggingface.co/Qwen/Qwen3-4B), [`8B`](https://huggingface.co/Qwen/Qwen3-8B) |
| OLMo-2 1B stages | [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B), [`SFT`](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT), [`DPO`](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO), [`Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) |
| OLMo-2 7B stages | [`allenai/OLMo-2-1124-7B`](https://huggingface.co/allenai/OLMo-2-1124-7B), [`SFT`](https://huggingface.co/allenai/OLMo-2-1124-7B-SFT), [`Instruct`](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) |
| Mistral replication | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Visual-language model | [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Embedding baselines | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

The paper also reports a model labelled LLaVA-1.5-7B. The compatible
Transformers conversion
[`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
is available and passes the repository/API checks, but the old result files do
not record their source model ID. Do not treat that ID as provenance-verified
for the historical LLaVA numbers without the original launch log.

Models can be cached by replacing `repo_id` in the `snapshot_download` example
above. Keep the exact ID (including organization and capitalization); the old
`gpt2` alias in `scripts/setup_server.sh` should be interpreted as
`openai-community/gpt2`.

### Datasets

| Data used here | Verified source and exact configuration |
|---|---|
| MMLU | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), configuration `all` |
| ARC | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc), configurations `ARC-Easy` and `ARC-Challenge` |
| GSM8K | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main` |
| LongMemEval oracle | [`xiaowu0162/longmemeval`](https://huggingface.co/datasets/xiaowu0162/longmemeval), file `longmemeval_oracle`, revision `2ec2a557f339b6c0369619b1ed5793734cc87533` |

The LongMemEval download was validated byte-for-byte against the tracked file:
both are 15,388,478 bytes with SHA-256
`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`.
To restore it explicitly:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="xiaowu0162/longmemeval",
    repo_type="dataset",
    filename="longmemeval_oracle",
    revision="2ec2a557f339b6c0369619b1ed5793734cc87533",
    local_dir="data/benchmarks/longmemeval",
)
PY
```

The standard evaluation datasets are loaded directly by
`src/experiments/general_eval.py`, for example
`load_dataset("cais/mmlu", "all")`; no manual conversion is required.

LoCoMo is deliberately **not** assigned a Hugging Face download command. The
author repository [`adymaharana/locomo`](https://huggingface.co/datasets/adymaharana/locomo)
currently contains a card but no downloadable data file, while available
community mirrors use transformed schemas and do not byte-match this project's
`data/benchmarks/locomo/locomo10.json`. The verified experiment copy is already
tracked in this Git repository, as are the custom `battery*.json` files and
`vlm_images/`; obtain them with `git clone`/`git pull`.

Finally, directories named `*-metacog*`, LoRA adapters, and campaign
checkpoints under `data/results/` are outputs trained by this project, not
public Hugging Face base models. They are intentionally excluded from Git and
cannot be reconstructed by downloading a similarly named HF repository; copy
the required artifact from the originating server or publish it separately
with its manifest and checksum.

## Workspace-guided RL extension

The staged Workspace-SFT, RL-W, RL-QA, and hybrid implementation is documented
in [`docs/RL_EXPERIMENTS.md`](docs/RL_EXPERIMENTS.md). Start with a data-only
check; later stages are intentionally never launched automatically across a
decision gate:

```bash
bash scripts/run_memory_rl_mvp.sh dry-run rl-w
```

## Next H100 campaigns

The H100 operator should begin with the
[`H100 documentation guide`](docs/H100_DOCUMENT_GUIDE.md), which explains the
role and precedence of every README/runbook in this repository.

The post-A5000 handoff is [`docs/H100_NEXT_CAMPAIGNS.md`](docs/H100_NEXT_CAMPAIGNS.md).
It keeps two confirmatory tracks separate: a Qwen3-8B capable-scale RL-QA
replication and Qwen3-8B Binary Metacognitive Alignment across seeds 0/1/2.
The document freezes scope, gates, OOD isolation, and required reports; it does
not itself authorize an unattended H100 launch.
