# Qwen3-8B A100 Metacognitive Alignment M1 gate report

Report schema: `metacog-alignment-m1-report/v1`. Final A100 decision: **AMBER**.

## 1. M0 baseline reproduction status

M0 decision: **GREEN**. Reference and tolerance:

```json
{
  "absolute_delta": null,
  "observed": {
    "V": 0.3365829477858559,
    "W_rr": 0.6918649482264816
  },
  "reference": null,
  "tolerance": null
}
```

## 2. Model/tokenizer revisions

- Model: `Qwen/Qwen3-32B`
- Model revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Tokenizer revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Chat-template SHA-256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`

## 3. A100 memory/throughput configuration

Preflight device: `NVIDIA A100-SXM4-80GB`, total 81920 MiB, free 81152 MiB. Canary runtime summary:

```json
{
  "gpu_memory": {
    "allocated_bytes": 67803960832,
    "peak_allocated_bytes": 68365798912,
    "peak_reserved_bytes": 69428314112,
    "reserved_bytes": 69428314112
  },
  "status": "PASS",
  "throughput": {
    "mean_examples_per_second": 1.575410131190999,
    "mean_tokens_per_second": 181.3092577576255
  }
}
```

## 4. Teacher-label construction audit

The formal trainer is restricted to Explicit, Evoked, and Evoked-G2; labels are top-2 Yes per episode from the frozen original workspace teacher.

```json
{
  "data_isolation": {
    "allowed_sources": [
      "evoked",
      "evoked_g2",
      "explicit"
    ],
    "observed_sources": [
      "evoked",
      "evoked_g2",
      "explicit"
    ],
    "ood_evaluated": false,
    "ood_loaded": false
  },
  "teacher": {
    "frozen_original": true,
    "label_rule": "within episode W_rr descending; original candidate order at ties; top-2 yes, rest no",
    "metadata_sidecars_verified": true,
    "model": "Qwen/Qwen3-32B",
    "requested_revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "resolved_revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "source_artifacts": [
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json",
        "battery_sha256": "e8ce4db85d3a0c09879575850e36b91821af8be155e86495cd059ae35485e7d6",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m0/evoked.json.metadata",
        "metadata_sha256": "3ca57f8f024dac4f0e8d6377b7215c24807be0b8f8430b1f1a6cc14109047f93",
        "model": "Qwen/Qwen3-32B",
        "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m0/evoked.json",
        "results_sha256": "3e6fec2b2c364e11d523fc8d6a8b9e334bcc46b3be8a9e46a3dec24dad9d0611",
        "source": "evoked",
        "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json",
        "battery_sha256": "32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m0/evoked_g2.json.metadata",
        "metadata_sha256": "385500013cdef46c338031475b59a8ea164f546fe35636033f9d12d5721fcd2f",
        "model": "Qwen/Qwen3-32B",
        "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m0/evoked_g2.json",
        "results_sha256": "b26f17e09e758df3a0aabc14b4798872aaee114591c0d31f5602ad357c10a10a",
        "source": "evoked_g2",
        "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json",
        "battery_sha256": "35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m0/explicit.json.metadata",
        "metadata_sha256": "637f9767701d14d236510ee8ad6f16721b3a2333b4a37c5edf242888d0a12338",
        "model": "Qwen/Qwen3-32B",
        "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m0/explicit.json",
        "results_sha256": "7a4e98ea01965d44d3fadd115af55bb6b66596b3a6a3e777be6c7a7f2219a638",
        "source": "explicit",
        "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137"
      }
    ],
    "student_workspace_used_for_labels": false,
    "workspace_scores": "precomputed_W_rr",
    "workspace_scores_recomputed_during_training": false
  },
  "teacher_label_audit": {
    "method": "stable W_rr descending equivalent: (-W_rr, candidate_index)",
    "schema_version": 1,
    "target_text": {
      "negative": "no",
      "positive": "yes"
    },
    "top_k": 2,
    "train": {
      "episode_audit": [
        {
          "boundary_score": null,
          "episode_id": "evoked:episode:000005",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked:episode:000044",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": 0.0031645569620253164,
          "episode_id": "evoked:episode:000062",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000007",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000064",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        }
      ],
      "episodes": 175,
      "episodes_with_any_tie": 5,
      "episodes_with_top_k_boundary_tie": 1,
      "tie_groups": 5,
      "tied_candidates": 10
    },
    "train_target_counts": {
      "no": 475,
      "yes": 350
    },
    "validation": {
      "episode_audit": [
        {
          "boundary_score": 0.0034129692832764505,
          "episode_id": "explicit:episode:000047",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        }
      ],
      "episodes": 45,
      "episodes_with_any_tie": 1,
      "episodes_with_top_k_boundary_tie": 1,
      "tie_groups": 1,
      "tied_candidates": 2
    },
    "validation_target_counts": {
      "no": 125,
      "yes": 90
    }
  }
}
```

## 5. Training configuration

```json
{
  "campaign_stage": "M1",
  "checkpoint_steps": [
    0,
    100,
    250,
    414
  ],
  "device": {
    "bf16_supported": true,
    "compute_capability": [
      8,
      0
    ],
    "index": 0,
    "name": "NVIDIA A100-SXM4-80GB",
    "total_memory_bytes": 85093777408,
    "type": "cuda"
  },
  "dry_run": false,
  "effective_max_sequence_length": 1024,
  "epochs": 2,
  "gradient_accumulation": 4,
  "gradient_checkpointing": true,
  "learning_rate": 1e-05,
  "lora_alpha": 32,
  "lora_dropout": 0.0,
  "lora_rank": 16,
  "lora_target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "max_grad_norm": 1.0,
  "max_steps": 500,
  "max_steps_is_cap": true,
  "mode": "formal_seed_0_pilot",
  "model": "Qwen/Qwen3-32B",
  "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
  "no_token_ids": [
    902,
    2152,
    2308,
    2753,
    8996
  ],
  "ood_evaluation_enabled": false,
  "optimizer": "torch.optim.AdamW",
  "per_device_batch_size": 1,
  "precision": "bfloat16",
  "requested_max_sequence_length": 1024,
  "resolved_model_commit": "9216db5781bf21249d130ec9da846c4624c16137",
  "resolved_tokenizer_commit": "9216db5781bf21249d130ec9da846c4624c16137",
  "scheduler": "constant",
  "schema_version": 1,
  "seed": 0,
  "selection_metric": "verbal_auc",
  "selection_scope": "id_validation",
  "selection_tie_break": "earliest_step",
  "sequence_length_auto_increased": false,
  "software_versions": {
    "accelerate": "1.14.0",
    "cuda": "12.6",
    "peft": "0.13.2",
    "python": "3.12.14",
    "safetensors": "0.8.0",
    "torch": "2.7.1+cu126",
    "transformers": "4.57.6"
  },
  "split_seed": 0,
  "target_optimizer_steps": 414,
  "teacher_top_k": 2,
  "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137",
  "total_parameter_count": 32896340992,
  "train_candidate_count": 825,
  "train_episode_count": 175,
  "trainable_parameter_count": 134217728,
  "training_sources": [
    "explicit",
    "evoked",
    "evoked_g2"
  ],
  "validation_candidate_count": 215,
  "validation_episode_count": 45,
  "validation_fraction": 0.2,
  "weight_decay": 0.0,
  "yes_token_ids": [
    7414,
    9454,
    9693,
    9834,
    14004
  ]
}
```

## 6. Loss/gradient health

```json
{
  "canary_adapter_enable_disable": {
    "disable_flags_entered": true,
    "disable_flags_restored": true,
    "disabled_yes_probability": 0.9846680760383606,
    "enabled_logits_restored": true,
    "enabled_yes_probability": 0.7982653379440308,
    "fixed_candidate_id": "evoked:episode:000006:candidate:000",
    "logits_all_finite": true,
    "lora_layers_audited": 448,
    "max_abs_enabled_before_after": 0.0,
    "passed": true,
    "performed": true
  },
  "canary_checkpoint_save_load": {
    "all_tensors_finite": true,
    "checkpoint_tree_sha256": "2ffebed87c82875278d6c9e335f738b2225addb80c8818197f9de014ba18f2a5",
    "live_peft_roundtrip": {
      "logits_all_finite": true,
      "max_abs_before_vs_restored_logits": 0.0,
      "max_abs_before_vs_restored_state": 0.0,
      "max_abs_saved_vs_reloaded_logits": 0.0,
      "original_adapter_restored": true,
      "passed": true,
      "performed": true,
      "reload_method": "in_place_original_adapter"
    },
    "passed": true,
    "peft_config_loaded": true,
    "peft_type": "PeftType.LORA",
    "readable": true,
    "tensor_count": 896
  },
  "canary_finite_loss_and_gradients": true,
  "canary_workspace_evaluation": {
    "adapter_enabled": true,
    "all_finite": true,
    "artifact": "canary_workspace_id.jsonl",
    "candidate_rows": 5,
    "episode_id": "evoked:episode:000006",
    "expected_candidate_rows": 5,
    "max_w_rr": 0.0049504950495049506,
    "min_w_rr": 0.0016556291390728477,
    "performed": true,
    "readout": {
      "candidate_variants": [
        "space+lowercase",
        "space+capitalized",
        "lowercase",
        "capitalized"
      ],
      "final_norm": "Qwen3RMSNorm",
      "layers": "1..n_max_reciprocal_rank",
      "position": "final_raw_context_token",
      "unembedding": "Linear"
    },
    "scope": "fixed_id_validation_first_episode",
    "used_for_checkpoint_selection": false,
    "workspace_auc": 1.0
  },
  "formal_health": {
    "all_loss_and_gradients_finite": true,
    "final_loss": 0.6341050267219543,
    "first_loss": 29.74560546875,
    "gpu_memory": {
      "allocated_bytes": 66692314112,
      "peak_allocated_bytes": 68380972544,
      "peak_reserved_bytes": 69432508416,
      "reserved_bytes": 69432508416
    },
    "maximum_gradient_norm": 126.43643951416016,
    "maximum_loss": 30.404266357421875,
    "minimum_gradient_norm": 1.203179121017456,
    "minimum_loss": 0.0964666772633791,
    "optimizer_steps_recorded": 414,
    "throughput": {
      "mean_examples_per_second": 1.6223707626538393,
      "mean_tokens_per_second": 182.6256929833226
    }
  }
}
```

## 7. ID checkpoint-selection table

| Step | ID verbal AUC | ID Yes rate | Path | Selected |
|---:|---:|---:|---|:---:|
| 0 | 0.499 | 0.986 | `checkpoints/step-000000` | no |
| 100 | 0.637 | 0.005 | `checkpoints/step-000100` | yes |
| 250 | 0.587 | 0.056 | `checkpoints/step-000250` | no |
| 414 | 0.547 | 0.367 | `checkpoints/step-000414` | no |

Selection used ID verbal AUC only; ties select the earliest checkpoint.

## 8. Locked checkpoint

- Path: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/m1/checkpoints/step-000100`
- Step: `100`
- ID verbal AUC: `0.637`
- Tree SHA-256: `8156501a1876c11880ab9010cb59c7c4d214c9f0b9b068759c10c6553e6c5eda`

## 9. Decoupled V before/after

| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | Before within-episode | After within-episode | Yes rate before | Yes rate after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decoupled | V | 0.337 | 0.432 | +0.095 | 0.196 | 0.407 | 0.997 | 0.021 |
| Decoupled | W_rr | 0.692 | 0.693 | +0.001 | 0.705 | 0.701 | — | — |
| Compositional | V | 0.332 | 0.314 | -0.018 | 0.191 | 0.315 | 1.000 | 0.011 |
| Compositional | W_rr | 0.553 | 0.555 | +0.001 | 0.584 | 0.591 | — | — |

## 10. Decoupled W before/after

The combined table above reports the preregistered W_rr before/after and delta.

## 11. Compositional V/W before/after

The combined table above reports both Compositional channels without tuning or reruns.

## 12. Within-episode metrics

Within-episode AUCs are shown alongside pooled AUCs in the combined table.

## 13. Yes-rate before/after

Yes rates use the fixed `V >= 0.5` decision and are shown in the combined table.

## 14. Full-context QA before/after

| Condition | Before accuracy | After accuracy | Delta | Drop (pp) |
|---|---:|---:|---:|---:|
| Decoupled | 0.750 | 0.750 | +0.000 | +0.000 |
| Compositional | 0.827 | 0.827 | +0.000 | +0.000 |

## 15. Bootstrap CIs

All paired AUC intervals use 4000 whole-episode cluster draws.

| Condition | Channel | Delta estimate | 95% CI | Effective draws |
|---|---|---:|---:|---:|
| Decoupled | V | +0.095 | [+0.024, +0.176] | 4000 |
| Decoupled | W_rr | +0.001 | [-0.003, +0.005] | 4000 |
| Compositional | V | -0.018 | [-0.085, +0.055] | 4000 |
| Compositional | W_rr | +0.001 | [-0.003, +0.006] | 4000 |

## 16. GREEN / AMBER / RED decision

Decision: **AMBER**. Strong GREEN: `False`.
Controlled AMBER branch authorized: `True`.

```json
{
  "decision_reasons": [
    "directional gain with no-harm passed, below GREEN effect size"
  ]
}
```

## 17. Artifact paths and hashes

| Artifact | SHA-256 |
|---|---|
| `canary/` | `02e85e71199f998d16bcd95f096cd90abc9b26fad00cd73300ac30fe08a35961` |
| `canary/canary_manifest.json` | `acf5494525365703b8c83c8e74bc46cd92dd7206a3af3d35a5f64e83bb0bc4a2` |
| `canary/canary_workspace_id.jsonl` | `104b644fd6ebbf9096c441c6644ba6906c4e07ba9606feda44faa5e99959d779` |
| `canary/checkpoints/step-000000/README.md` | `f86a9e7fe1b0419caf47c240b6cc4c8eeefb3e285a26d947fe3fccc40094387a` |
| `canary/checkpoints/step-000000/adapter_config.json` | `bc4e9c9620bc32f0ee312442cdf3c2f38cc0adc8c0e9319cefaf846347bc66e2` |
| `canary/checkpoints/step-000000/adapter_model.safetensors` | `a7dc5fcb526a4dfb1d04ce4df5d7c65235aecc14ed3a0d24072bd449ffaddf6e` |
| `canary/checkpoints/step-000000/training_state.json` | `b12d9a35fbf6af5b841ed4d7c2f31e2c93ae6f09ac1f1dc28cd311777562bd6f` |
| `canary/checkpoints/step-000000/validation_metrics.json` | `06579100657befba11a569d36369a03249969b5147250800c32861ea57d9030b` |
| `canary/checkpoints/step-000010/README.md` | `f86a9e7fe1b0419caf47c240b6cc4c8eeefb3e285a26d947fe3fccc40094387a` |
| `canary/checkpoints/step-000010/adapter_config.json` | `bc4e9c9620bc32f0ee312442cdf3c2f38cc0adc8c0e9319cefaf846347bc66e2` |
| `canary/checkpoints/step-000010/adapter_model.safetensors` | `dc7ea983e50e5cd2830d96f5ebe875522677f06ce8ce2739a26fcd4402cbe557` |
| `canary/checkpoints/step-000010/training_state.json` | `e189cff7aaa3d6d48fdfa70451fe98322e4b4aeb326679761c7ae9814ed05128` |
| `canary/checkpoints/step-000010/validation_metrics.json` | `66d05da57575efdb595df008ea180159cf37f1a9b502f781842a09cf86d77aea` |
| `canary/provenance.json` | `a6074914124645842607bcd4a53f3fd432684bb6059b866e4db68bc819fe83aa` |
| `canary/run_config.json` | `315c84a73420de835c7d27ce5ed1167bb5ae0fdc8efcafba4c9e451d5edd34cb` |
| `canary/split_manifest.json` | `ed4bfa34aa3674c2d5df924969cf9bb98b39b3dcf3ded2d874a928939b5a9670` |
| `canary/summary.json` | `f516b896ea8effb47f2b6850278ffaeb12bfcfa28e465cc0812e49d5a36b92d8` |
| `canary/teacher_label_audit.json` | `7be5339d5ff18c984de97c583107d224b551a3070e38596ddacb3a742233d3d3` |
| `canary/teacher_labels.jsonl` | `281ab4f47754a2b23e7c06aba91f5f75354b6cd90296fb253193a353fb33a0d0` |
| `canary/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `canary/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `canary/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `canary/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `canary/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `canary/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `canary/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `canary/training_metrics.jsonl` | `3454c8e4d68a5d5e472bcae9f3055ddb5b44d6ba9fc0e7d85fc83d0c5864744e` |
| `canary/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `canary/validation_metrics.jsonl` | `b5c4c295f2fb1c3d1c3631b195e9f79861909344e3b1ba61a49d1328ebae4fa5` |
| `canary/validation_scores/step-000000.jsonl` | `aceec8129c7ea1d2a1a532959554897b8e47ac9b69d3ca9d062bea37fe3d333e` |
| `canary/validation_scores/step-000010.jsonl` | `3e0956b131383a19adf9a9991088dd15925be854ea1ba0f35929569ef518f8f2` |
| `id_lock/lock_manifest.json` | `c8f4be63167b41be6c520a8cb238cafe19b0c95eb56cdc0c90de9a9b01474d44` |
| `m0/compositional.json` | `aeed59eac1a1683cc6ad9e9a72f8f403b7a2791da75b13030896e1eaefde820b` |
| `m0/compositional.json.metadata` | `3eeb540ad56bc5bea85d6ec19bbd4d68c76f6ced8a50ec038637d92a3363c4c7` |
| `m0/decoupled.json` | `f1a054e588388de12dd39cfb62b8adaeb6dc90874bc5c2a9a2ad473374dfd932` |
| `m0/decoupled.json.metadata` | `32c9fb34c1fe58d7dfd6bcb79ebee70d2bcb71831f2c0229e0c6cef4cf5eb38f` |
| `m0/evoked.json` | `3e6fec2b2c364e11d523fc8d6a8b9e334bcc46b3be8a9e46a3dec24dad9d0611` |
| `m0/evoked.json.metadata` | `3ca57f8f024dac4f0e8d6377b7215c24807be0b8f8430b1f1a6cc14109047f93` |
| `m0/evoked_g2.json` | `b26f17e09e758df3a0aabc14b4798872aaee114591c0d31f5602ad357c10a10a` |
| `m0/evoked_g2.json.metadata` | `385500013cdef46c338031475b59a8ea164f546fe35636033f9d12d5721fcd2f` |
| `m0/explicit.json` | `7a4e98ea01965d44d3fadd115af55bb6b66596b3a6a3e777be6c7a7f2219a638` |
| `m0/explicit.json.metadata` | `637f9767701d14d236510ee8ad6f16721b3a2333b4a37c5edf242888d0a12338` |
| `m0/gate.json` | `f4590338b6bc6c178d3a1f34a1205fdfc61df80153f9e472e76de8d5fdc1e4dd` |
| `m0/gate.md` | `32575d48b72cb2ff0a5bb06357862f018e32c7ecd8ef7d5618035b2f41193d30` |
| `m1/` | `4be64be3d3ea9feadc1afaa3e3902e6d2b8796b05ed6bcb5098f55313caffc78` |
| `m1/checkpoints/step-000000/README.md` | `f86a9e7fe1b0419caf47c240b6cc4c8eeefb3e285a26d947fe3fccc40094387a` |
| `m1/checkpoints/step-000000/adapter_config.json` | `126e4cd4b401249e2fdfd78ccac5140c44f599261631624cb84b4842503a2c05` |
| `m1/checkpoints/step-000000/adapter_model.safetensors` | `a7dc5fcb526a4dfb1d04ce4df5d7c65235aecc14ed3a0d24072bd449ffaddf6e` |
| `m1/checkpoints/step-000000/training_state.json` | `2572e6789292bcc292a003d4e433ee4473285dc56abf5b69488f452474a84589` |
| `m1/checkpoints/step-000000/validation_metrics.json` | `06579100657befba11a569d36369a03249969b5147250800c32861ea57d9030b` |
| `m1/checkpoints/step-000100/README.md` | `f86a9e7fe1b0419caf47c240b6cc4c8eeefb3e285a26d947fe3fccc40094387a` |
| `m1/checkpoints/step-000100/adapter_config.json` | `126e4cd4b401249e2fdfd78ccac5140c44f599261631624cb84b4842503a2c05` |
| `m1/checkpoints/step-000100/adapter_model.safetensors` | `436c90c9e3a7b2bad9db20925119604b5d7ec6356c0eeaced30a40bcca2081b2` |
| `m1/checkpoints/step-000100/training_state.json` | `15762202c10f17e54f8e702f589362210579741efb9d172c38c8940bd4e3be2d` |
| `m1/checkpoints/step-000100/validation_metrics.json` | `1a9c845ce9aa9ad68caa06913b826c0f2baf859f303a71989975e66ca004adeb` |
| `m1/checkpoints/step-000250/README.md` | `f86a9e7fe1b0419caf47c240b6cc4c8eeefb3e285a26d947fe3fccc40094387a` |
| `m1/checkpoints/step-000250/adapter_config.json` | `126e4cd4b401249e2fdfd78ccac5140c44f599261631624cb84b4842503a2c05` |
| `m1/checkpoints/step-000250/adapter_model.safetensors` | `59b46c0b6b1a39e2f4eea18b762a11d55a6008c6472187f72025283d80cbe98e` |
| `m1/checkpoints/step-000250/training_state.json` | `06c6708e779d6e2e21d8e069e0f1e6a3498b8a653473cdfdfb035f895d08485c` |
| `m1/checkpoints/step-000250/validation_metrics.json` | `11120ed8cae55c4ae185b37ba1ec008af2cd4af14489819818b54a3154310f62` |
| `m1/checkpoints/step-000414/README.md` | `f86a9e7fe1b0419caf47c240b6cc4c8eeefb3e285a26d947fe3fccc40094387a` |
| `m1/checkpoints/step-000414/adapter_config.json` | `126e4cd4b401249e2fdfd78ccac5140c44f599261631624cb84b4842503a2c05` |
| `m1/checkpoints/step-000414/adapter_model.safetensors` | `b21ab1fe3773b462b0e96ec5e5d5afb3aae4649313e9bb8beff73ef1ab74d8b9` |
| `m1/checkpoints/step-000414/training_state.json` | `995999be15146ec4cd7bc88f49eb276b56cd1df29430fb1b2d8580b37f9eb72b` |
| `m1/checkpoints/step-000414/validation_metrics.json` | `b1e9c8509d1e505fa1fbbe9eae18144994f0e5c484d409f112ced355c6f02410` |
| `m1/lock_manifest.json` | `e0bf522c532081ba3ab46765fd56a6c61edc0c8e95d8601eb9980825221ad2bb` |
| `m1/provenance.json` | `a6074914124645842607bcd4a53f3fd432684bb6059b866e4db68bc819fe83aa` |
| `m1/run_config.json` | `ad2f9ad8f067e4d29f72421c2fabe29941b0c09b7cffb2ba4945ef11fdff97ef` |
| `m1/split_manifest.json` | `ed4bfa34aa3674c2d5df924969cf9bb98b39b3dcf3ded2d874a928939b5a9670` |
| `m1/summary.json` | `31aab975a2948c3fe97bc9baa75fd93b4b3b15486a9a21a48fae71e78ca5bb83` |
| `m1/teacher_label_audit.json` | `7be5339d5ff18c984de97c583107d224b551a3070e38596ddacb3a742233d3d3` |
| `m1/teacher_labels.jsonl` | `281ab4f47754a2b23e7c06aba91f5f75354b6cd90296fb253193a353fb33a0d0` |
| `m1/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `m1/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `m1/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `m1/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `m1/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `m1/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `m1/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `m1/training_metrics.jsonl` | `d9bada810d81e805f022d3c1889b72c219b2beb163796d941419c2791b8f977f` |
| `m1/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `m1/validation_metrics.jsonl` | `20ab78822f7bb7aac0530a5370e3f8cd306773bae28627372244c9de5749dd4c` |
| `m1/validation_scores/step-000000.jsonl` | `aceec8129c7ea1d2a1a532959554897b8e47ac9b69d3ca9d062bea37fe3d333e` |
| `m1/validation_scores/step-000100.jsonl` | `962f8bd9c4b83c2c3dd0864c163df114c3e8feabb3f5e1f312ad4885b5d2e97d` |
| `m1/validation_scores/step-000250.jsonl` | `177ecad70fe24f25ebd8676fa15b1157ff37b08411c58d8a16bdea680d94262b` |
| `m1/validation_scores/step-000414.jsonl` | `9932c24f2a2b05ed2fb18650987cca2753125dfef796bb933185f9bb78407be2` |
| `ood/result.json` | `4086d6d90b94029b5212a9ab5dfc9331d0fe9f9b040693dce46f0298a34fde85` |

## 18. No H100 job was launched

No H100 job was launched. This invocation used the single verified NVIDIA RTX A100 and stops here for manual review; it launched no M2 or later stage.
