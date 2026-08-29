# Qwen3-8B A100 Metacognitive Alignment M1 gate report

Report schema: `metacog-alignment-m1-report/v1`. Final A100 decision: **GREEN**.

## 1. M0 baseline reproduction status

M0 decision: **GREEN**. Reference and tolerance:

```json
{
  "absolute_delta": {
    "V": 0.004539986781229299,
    "W_rr": 0.0005494602335316401
  },
  "observed": {
    "V": 0.3415399867812293,
    "W_rr": 0.6545494602335317
  },
  "reference": {
    "V": 0.337,
    "W_rr": 0.654
  },
  "tolerance": 0.05
}
```

## 2. Model/tokenizer revisions

- Model: `Qwen/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Tokenizer revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Chat-template SHA-256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`

## 3. A100 memory/throughput configuration

Preflight device: `NVIDIA A100-SXM4-80GB`, total 81920 MiB, free 81152 MiB. Canary runtime summary:

```json
{
  "gpu_memory": {
    "allocated_bytes": 17132925952,
    "peak_allocated_bytes": 17429916672,
    "peak_reserved_bytes": 18146656256,
    "reserved_bytes": 18146656256
  },
  "status": "PASS",
  "throughput": {
    "mean_examples_per_second": 2.7755040674406097,
    "mean_tokens_per_second": 311.24345571287455
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
    "model": "Qwen/Qwen3-8B",
    "requested_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "resolved_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "source_artifacts": [
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json",
        "battery_sha256": "e8ce4db85d3a0c09879575850e36b91821af8be155e86495cd059ae35485e7d6",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked.json.metadata",
        "metadata_sha256": "3ff9b9c0c5adee71b40eb58dde5b914fe8418b6f184df3e2989e3ca6403a1e2f",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked.json",
        "results_sha256": "ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed",
        "source": "evoked",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json",
        "battery_sha256": "32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked_g2.json.metadata",
        "metadata_sha256": "1a0fd7e7a4764c90f53e21d292a75c588e9be188898620aa1ef0b235b5a58944",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked_g2.json",
        "results_sha256": "041028d1806f58baac05efe9f285559ad8cd7b1fc3496810f5e2864059f91fa3",
        "source": "evoked_g2",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json",
        "battery_sha256": "35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/explicit.json.metadata",
        "metadata_sha256": "4d1fcf548cb964e6acefe83b079a372ab3aed36e349805e1a2dacbde57284e8d",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/explicit.json",
        "results_sha256": "2189e6e5333d33afdd1b56d9e4a20307ce83c51ba0ed45d13cebf2877a306488",
        "source": "explicit",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
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
          "boundary_score": 0.0034602076124567475,
          "episode_id": "evoked:episode:000031",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": 0.008264462809917356,
          "episode_id": "evoked:episode:000032",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": null,
          "episode_id": "evoked:episode:000037",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked:episode:000055",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked:episode:000062",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked_g2:episode:000005",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked_g2:episode:000015",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked_g2:episode:000053",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000028",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": 0.008771929824561403,
          "episode_id": "explicit:episode:000050",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000063",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        }
      ],
      "episodes": 175,
      "episodes_with_any_tie": 11,
      "episodes_with_top_k_boundary_tie": 3,
      "tie_groups": 11,
      "tied_candidates": 22
    },
    "train_target_counts": {
      "no": 475,
      "yes": 350
    },
    "validation": {
      "episode_audit": [
        {
          "boundary_score": null,
          "episode_id": "evoked:episode:000006",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": 0.0029498525073746312,
          "episode_id": "evoked:episode:000074",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": 0.010309278350515464,
          "episode_id": "explicit:episode:000047",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        }
      ],
      "episodes": 45,
      "episodes_with_any_tie": 3,
      "episodes_with_top_k_boundary_tie": 2,
      "tie_groups": 3,
      "tied_candidates": 6
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
  "model": "Qwen/Qwen3-8B",
  "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
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
  "resolved_model_commit": "b968826d9c46dd6066d109eabc6255188de91218",
  "resolved_tokenizer_commit": "b968826d9c46dd6066d109eabc6255188de91218",
  "scheduler": "constant",
  "schema_version": 1,
  "seed": 1,
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
  "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
  "total_parameter_count": 8234382336,
  "train_candidate_count": 825,
  "train_episode_count": 175,
  "trainable_parameter_count": 43646976,
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
    "disabled_yes_probability": 0.18465577065944672,
    "enabled_logits_restored": true,
    "enabled_yes_probability": 0.34820908308029175,
    "fixed_candidate_id": "evoked:episode:000006:candidate:000",
    "logits_all_finite": true,
    "lora_layers_audited": 252,
    "max_abs_enabled_before_after": 0.0,
    "passed": true,
    "performed": true
  },
  "canary_checkpoint_save_load": {
    "all_tensors_finite": true,
    "checkpoint_tree_sha256": "6b54476ba89f08855537c1fca1c45d3e34f2c3c256e473bf68c9a58ba11fb985",
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
    "tensor_count": 504
  },
  "canary_finite_loss_and_gradients": true,
  "canary_workspace_evaluation": {
    "adapter_enabled": true,
    "all_finite": true,
    "artifact": "canary_workspace_id.jsonl",
    "candidate_rows": 5,
    "episode_id": "evoked:episode:000006",
    "expected_candidate_rows": 5,
    "max_w_rr": 0.006666666666666667,
    "min_w_rr": 0.00026946914578280785,
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
    "final_loss": 1.138826847076416,
    "first_loss": 28.156251430511475,
    "gpu_memory": {
      "allocated_bytes": 16749801984,
      "peak_allocated_bytes": 17468276736,
      "peak_reserved_bytes": 18419286016,
      "reserved_bytes": 18419286016
    },
    "maximum_gradient_norm": 140.41941833496094,
    "maximum_loss": 30.425782203674316,
    "minimum_gradient_norm": 5.232843399047852,
    "minimum_loss": 0.13042415864765644,
    "optimizer_steps_recorded": 414,
    "throughput": {
      "mean_examples_per_second": 2.824430141851838,
      "mean_tokens_per_second": 317.9334083351017
    }
  }
}
```

## 7. ID checkpoint-selection table

| Step | ID verbal AUC | ID Yes rate | Path | Selected |
|---:|---:|---:|---|:---:|
| 0 | 0.553 | 0.414 | `checkpoints/step-000000` | no |
| 100 | 0.488 | 0.874 | `checkpoints/step-000100` | no |
| 250 | 0.537 | 0.107 | `checkpoints/step-000250` | no |
| 414 | 0.575 | 0.367 | `checkpoints/step-000414` | yes |

Selection used ID verbal AUC only; ties select the earliest checkpoint.

## 8. Locked checkpoint

- Path: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m1/checkpoints/step-000414`
- Step: `414`
- ID verbal AUC: `0.575`
- Tree SHA-256: `6a669d69891d74fcc9fbe53959cdfb822d547bce12e6882133f3ecfd5c345d91`

## 9. Decoupled V before/after

| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | Before within-episode | After within-episode | Yes rate before | Yes rate after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decoupled | V | 0.342 | 0.689 | +0.347 | 0.213 | 0.754 | 0.000 | 0.373 |
| Decoupled | W_rr | 0.655 | 0.655 | +0.001 | 0.665 | 0.664 | — | — |
| Compositional | V | 0.326 | 0.599 | +0.273 | 0.189 | 0.632 | 0.000 | 0.441 |
| Compositional | W_rr | 0.547 | 0.547 | +0.001 | 0.566 | 0.572 | — | — |

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
| Decoupled | 0.706 | 0.706 | +0.000 | +0.000 |
| Compositional | 0.712 | 0.731 | +0.019 | -1.923 |

## 15. Bootstrap CIs

All paired AUC intervals use 4000 whole-episode cluster draws.

| Condition | Channel | Delta estimate | 95% CI | Effective draws |
|---|---|---:|---:|---:|
| Decoupled | V | +0.347 | [+0.276, +0.425] | 4000 |
| Decoupled | W_rr | +0.001 | [-0.002, +0.004] | 4000 |
| Compositional | V | +0.273 | [+0.180, +0.371] | 4000 |
| Compositional | W_rr | +0.001 | [-0.004, +0.006] | 4000 |

## 16. GREEN / AMBER / RED decision

Decision: **GREEN**. Strong GREEN: `True`.
Controlled AMBER branch authorized: `False`.

```json
{
  "decision_reasons": [
    "all preregistered M1 GREEN conditions passed"
  ]
}
```

## 17. Artifact paths and hashes

| Artifact | SHA-256 |
|---|---|
| `canary/` | `33be125e298b2d8556b124772fcd072422cf045542a79e588f2bb5582865de51` |
| `canary/canary_manifest.json` | `58a35564f5e3113d6a44eb3b99b3732e8c81d0aa349e6bd213c91a1f9f437727` |
| `canary/canary_workspace_id.jsonl` | `fa63c1347790ac0e2cb6e001cebe38c924aa7e27b104f6d9ef4dc8a5be304007` |
| `canary/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000000/adapter_config.json` | `08d3709914b488c72f5d3275792692fb3750bb8a85e6bde66b4eda152ccebc48` |
| `canary/checkpoints/step-000000/adapter_model.safetensors` | `f1db16dee7c5d7a1e62ab9b211925a5302d8a5e559758f4a9d0186d21be2f5f7` |
| `canary/checkpoints/step-000000/training_state.json` | `f58d33f2225b6f773e48b8a435efa2b78e107a48cc95640a1a6ccdca1c616c8b` |
| `canary/checkpoints/step-000000/validation_metrics.json` | `52f36dff258a23034874cdcafc4c846a9f78a16913b053b3deb0f9d678e66066` |
| `canary/checkpoints/step-000010/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000010/adapter_config.json` | `08d3709914b488c72f5d3275792692fb3750bb8a85e6bde66b4eda152ccebc48` |
| `canary/checkpoints/step-000010/adapter_model.safetensors` | `1718de98e3df509c23cd2554113795bb7c5ee7cd6bbb4a19ef82769baa159898` |
| `canary/checkpoints/step-000010/training_state.json` | `b0b0598d089f4aae9b525a529b0764c07d40285667b6fc1e7f0b784e41b9accc` |
| `canary/checkpoints/step-000010/validation_metrics.json` | `d0503fbb7832c11f16cba2bc3e2fc92e1828fcdd69ded587c2957c1bac21103e` |
| `canary/provenance.json` | `95d90afeea0159689538a510f8b9cdfcff5b9717134515cfe2a67cb7bffc0b97` |
| `canary/run_config.json` | `59c3ea61803e7f3935465a8b6e2e7c550f673aefbf76436c9a5e800c0540289e` |
| `canary/split_manifest.json` | `4465d688805ad16a4e728b24f34164611cd500128863eca6e57974e654458117` |
| `canary/summary.json` | `333692d4f67e52589890f0bc998b4401ef1450cedb240e307d3eeaaaf0a027d2` |
| `canary/teacher_label_audit.json` | `d44ab39b9091c15360528491c371f76b07a290f0e01e302e3080d9d10af5d08a` |
| `canary/teacher_labels.jsonl` | `17f6bf7324d53a7065943bc6a692d0f600cd3ee4401b2a0e2f90514d4a65000a` |
| `canary/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `canary/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `canary/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `canary/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `canary/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `canary/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `canary/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `canary/training_metrics.jsonl` | `4914a53eefb7e5ad1a04a30b54f0bdb0c2c237a103e1b70506425c26f39d2bd5` |
| `canary/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `canary/validation_metrics.jsonl` | `ab509632d3329f944216d72426ed8d80d5134aa482b605ec623e0310adc18074` |
| `canary/validation_scores/step-000000.jsonl` | `84f03ed078a745aa561abaf26d1ff4db0faae7beb988b51c03076d321d6c246c` |
| `canary/validation_scores/step-000010.jsonl` | `b107877cd8480ea3896f8e3ee9ace8dfaddae3b7397cc874f3f2a5ef114bd8f3` |
| `id_lock/lock_manifest.json` | `b387345110c29d8118d6e68fb095281ab6c632ee4a9420aa1bcba9e950999a1f` |
| `m0/compositional.json` | `2dbc1faedda196a4248dec6d5efc6c31a0fe94230940efe593fb84f68abd69e7` |
| `m0/compositional.json.metadata` | `d1b053df5fa11f288c8d7aa9df7f60c7479244629fb8b32c559cdc868b35f8b1` |
| `m0/decoupled.json` | `09179319ad4f255faf88f0863036be7271beb56790e665d77812e8712cbe62a8` |
| `m0/decoupled.json.metadata` | `411cb3e4e9cca0463cbda84755abcaee46fc51ff893d3c0386df4e42216efef3` |
| `m0/evoked.json` | `ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed` |
| `m0/evoked.json.metadata` | `3ff9b9c0c5adee71b40eb58dde5b914fe8418b6f184df3e2989e3ca6403a1e2f` |
| `m0/evoked_g2.json` | `041028d1806f58baac05efe9f285559ad8cd7b1fc3496810f5e2864059f91fa3` |
| `m0/evoked_g2.json.metadata` | `1a0fd7e7a4764c90f53e21d292a75c588e9be188898620aa1ef0b235b5a58944` |
| `m0/explicit.json` | `2189e6e5333d33afdd1b56d9e4a20307ce83c51ba0ed45d13cebf2877a306488` |
| `m0/explicit.json.metadata` | `4d1fcf548cb964e6acefe83b079a372ab3aed36e349805e1a2dacbde57284e8d` |
| `m0/gate.json` | `e2b67190a621dcf29e341a8e2cb2df649bd20da95a291d2a52d614db3e8847e8` |
| `m0/gate.md` | `415d5e89ed0427895a785158b8b351084841881585213dfc2c40b74c230d6b0d` |
| `m1/` | `c2de2debf7136fe4d56926320320dafff59cf757689e5ffb1d3f81851a6cb5c7` |
| `m1/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000000/adapter_config.json` | `7227d2818e826b2b633821268784c2097bbfff801f7debdbea469b8ea42d91f8` |
| `m1/checkpoints/step-000000/adapter_model.safetensors` | `f1db16dee7c5d7a1e62ab9b211925a5302d8a5e559758f4a9d0186d21be2f5f7` |
| `m1/checkpoints/step-000000/training_state.json` | `c47ea1415b7632a75ae93686fc91fbd47f7a123b4f6cbf830f63e5ef7a5f12c0` |
| `m1/checkpoints/step-000000/validation_metrics.json` | `52f36dff258a23034874cdcafc4c846a9f78a16913b053b3deb0f9d678e66066` |
| `m1/checkpoints/step-000100/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000100/adapter_config.json` | `7227d2818e826b2b633821268784c2097bbfff801f7debdbea469b8ea42d91f8` |
| `m1/checkpoints/step-000100/adapter_model.safetensors` | `375bb9f653d7b9b18442b05036cbe252e9f815ad7d9af84d11e1798c3b05a06b` |
| `m1/checkpoints/step-000100/training_state.json` | `c727da59deb91637a2a497278ae4b0c8c028346e2d99a4c77b627dce0f411fe7` |
| `m1/checkpoints/step-000100/validation_metrics.json` | `2a5c43c0ab429ac5aedd3e126a432e4ac612cbf82b08971fd735165a2ca1b628` |
| `m1/checkpoints/step-000250/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000250/adapter_config.json` | `7227d2818e826b2b633821268784c2097bbfff801f7debdbea469b8ea42d91f8` |
| `m1/checkpoints/step-000250/adapter_model.safetensors` | `90db3ade1ee267f300642599889362035e18f02f8da15aaaa0bc1ea4eff9bcf4` |
| `m1/checkpoints/step-000250/training_state.json` | `2900118dfd3f560ca8fa76cfe0c30ed43be49dc323cacc19b173040ee7a29416` |
| `m1/checkpoints/step-000250/validation_metrics.json` | `aadd1337ebd2085a200cc3371551f2c95da24c55477e8f0459265127c3c24103` |
| `m1/checkpoints/step-000414/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000414/adapter_config.json` | `7227d2818e826b2b633821268784c2097bbfff801f7debdbea469b8ea42d91f8` |
| `m1/checkpoints/step-000414/adapter_model.safetensors` | `145a064531003a56715d9a992122182f95a60476696f793bf19543d6fba0d24d` |
| `m1/checkpoints/step-000414/training_state.json` | `e36b900fb6d22cb013917c24772c1fd68da346bdf4d7fc5afa301ad73689ae9e` |
| `m1/checkpoints/step-000414/validation_metrics.json` | `6155452eaf510f66b71c6640d212e1a0675a5e546d8e5dc229334cf897c15648` |
| `m1/lock_manifest.json` | `81de4a395f3e842bc4f79746dbfde2c3c7d67423b81fa07c211c43070fdfb0aa` |
| `m1/provenance.json` | `95d90afeea0159689538a510f8b9cdfcff5b9717134515cfe2a67cb7bffc0b97` |
| `m1/run_config.json` | `886a576133879db3ab144076c0815f2819a6459930e67a45c823f945fd9b5dd5` |
| `m1/split_manifest.json` | `4465d688805ad16a4e728b24f34164611cd500128863eca6e57974e654458117` |
| `m1/summary.json` | `4010de64635e9c9f4b245d3b38bd633859ba8d3b71a358ed144232c891f3f8aa` |
| `m1/teacher_label_audit.json` | `d44ab39b9091c15360528491c371f76b07a290f0e01e302e3080d9d10af5d08a` |
| `m1/teacher_labels.jsonl` | `17f6bf7324d53a7065943bc6a692d0f600cd3ee4401b2a0e2f90514d4a65000a` |
| `m1/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `m1/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `m1/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `m1/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `m1/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `m1/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `m1/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `m1/training_metrics.jsonl` | `1234290b9ebdf16a5c2998fb585fa6d1fca703943bda43a91d1cfb50556f73d5` |
| `m1/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `m1/validation_metrics.jsonl` | `e77ab61c9fdd978283f15090c8c4a72f731cb0acd708fd2c3e6464f167705312` |
| `m1/validation_scores/step-000000.jsonl` | `84f03ed078a745aa561abaf26d1ff4db0faae7beb988b51c03076d321d6c246c` |
| `m1/validation_scores/step-000100.jsonl` | `a0f7c78e8435dda949f3ac03b74da2e982d876a6ba19f7aeb74812def8809881` |
| `m1/validation_scores/step-000250.jsonl` | `7d66333b48d46b04bcc8533be7448719a9e16edbf4405b24f07d537119d865a3` |
| `m1/validation_scores/step-000414.jsonl` | `d6e3823f941bf3dc4108f2fc110cb13bdc8f8285906f96b8c31a974cb702ebd8` |
| `ood/result.json` | `d869e06996faf53b5e295daa31a184734df73e400213fa7232d4750190098ff6` |

## 18. No H100 job was launched

No H100 job was launched. This invocation used the single verified NVIDIA RTX A100 and stops here for manual review; it launched no M2 or later stage.
