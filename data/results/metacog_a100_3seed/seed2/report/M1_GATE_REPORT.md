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
    "allocated_bytes": 17134286336,
    "peak_allocated_bytes": 17431169024,
    "peak_reserved_bytes": 18415091712,
    "reserved_bytes": 18415091712
  },
  "status": "PASS",
  "throughput": {
    "mean_examples_per_second": 2.9020371532162086,
    "mean_tokens_per_second": 328.16973449884904
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
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked.json.metadata",
        "metadata_sha256": "6b28372ab108a6e9c61a968f9a75419447e9b78902b4c375eb9d5165e3437938",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked.json",
        "results_sha256": "ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed",
        "source": "evoked",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json",
        "battery_sha256": "32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked_g2.json.metadata",
        "metadata_sha256": "647ac5096c0b0d7eb0d2521bb1264a94169a9ca8c77d586d06fbe5bf3c98ac1d",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked_g2.json",
        "results_sha256": "041028d1806f58baac05efe9f285559ad8cd7b1fc3496810f5e2864059f91fa3",
        "source": "evoked_g2",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json",
        "battery_sha256": "35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json.metadata",
        "metadata_sha256": "729955b848ee49ca998cc200120c0504be9a2fa7f1258495fcffa76195bc7180",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json",
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
  "seed": 2,
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
    "enabled_yes_probability": 0.4776788353919983,
    "fixed_candidate_id": "evoked:episode:000006:candidate:000",
    "logits_all_finite": true,
    "lora_layers_audited": 252,
    "max_abs_enabled_before_after": 0.0,
    "passed": true,
    "performed": true
  },
  "canary_checkpoint_save_load": {
    "all_tensors_finite": true,
    "checkpoint_tree_sha256": "27907eb83eee1005fd5fd0b9518b971143a7b0f01e546b5b23c16ed3ac99b195",
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
    "max_w_rr": 0.007462686567164179,
    "min_w_rr": 0.00026888948642108095,
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
    "final_loss": 1.1381504535675049,
    "first_loss": 27.625001907348633,
    "gpu_memory": {
      "allocated_bytes": 16749801984,
      "peak_allocated_bytes": 17468002816,
      "peak_reserved_bytes": 18423480320,
      "reserved_bytes": 18423480320
    },
    "maximum_gradient_norm": 121.93275451660156,
    "maximum_loss": 29.796875476837158,
    "minimum_gradient_norm": 0.9414491057395935,
    "minimum_loss": 0.014410419738851488,
    "optimizer_steps_recorded": 414,
    "throughput": {
      "mean_examples_per_second": 2.8926186135346144,
      "mean_tokens_per_second": 325.53725708889147
    }
  }
}
```

## 7. ID checkpoint-selection table

| Step | ID verbal AUC | ID Yes rate | Path | Selected |
|---:|---:|---:|---|:---:|
| 0 | 0.553 | 0.414 | `checkpoints/step-000000` | no |
| 100 | 0.498 | 0.051 | `checkpoints/step-000100` | no |
| 250 | 0.585 | 0.791 | `checkpoints/step-000250` | yes |
| 414 | 0.564 | 0.191 | `checkpoints/step-000414` | no |

Selection used ID verbal AUC only; ties select the earliest checkpoint.

## 8. Locked checkpoint

- Path: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m1/checkpoints/step-000250`
- Step: `250`
- ID verbal AUC: `0.585`
- Tree SHA-256: `d17669b04e453f5633bf5da51667bc8c5b164e34e2794539d1c6f9636ff414da`

## 9. Decoupled V before/after

| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | Before within-episode | After within-episode | Yes rate before | Yes rate after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decoupled | V | 0.342 | 0.589 | +0.248 | 0.213 | 0.639 | 0.000 | 0.934 |
| Decoupled | W_rr | 0.655 | 0.654 | -0.000 | 0.665 | 0.664 | — | — |
| Compositional | V | 0.326 | 0.468 | +0.142 | 0.189 | 0.515 | 0.000 | 0.870 |
| Compositional | W_rr | 0.547 | 0.547 | +0.000 | 0.566 | 0.573 | — | — |

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
| Decoupled | 0.706 | 0.691 | -0.015 | +1.471 |
| Compositional | 0.712 | 0.731 | +0.019 | -1.923 |

## 15. Bootstrap CIs

All paired AUC intervals use 4000 whole-episode cluster draws.

| Condition | Channel | Delta estimate | 95% CI | Effective draws |
|---|---|---:|---:|---:|
| Decoupled | V | +0.248 | [+0.187, +0.316] | 4000 |
| Decoupled | W_rr | -0.000 | [-0.003, +0.002] | 4000 |
| Compositional | V | +0.142 | [+0.060, +0.231] | 4000 |
| Compositional | W_rr | +0.000 | [-0.003, +0.004] | 4000 |

## 16. GREEN / AMBER / RED decision

Decision: **GREEN**. Strong GREEN: `False`.
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
| `canary/` | `339a9ee80db38f114b054e9869591cf49db2b0f1eeac74d761cd7f78b34604a0` |
| `canary/canary_manifest.json` | `b8da8c0f192b83bf3f96951ef4c599b426cc756e20926a7dbb75eb6fc1d2514c` |
| `canary/canary_workspace_id.jsonl` | `53496d939eb004ba86480e6d33e9acaecb802b8bd93cee347330e9f6b8c6c17c` |
| `canary/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000000/adapter_config.json` | `1817a64f405deaf3f621d0c018725bbc26821ef656fc7d9ce37edf7950e9d85a` |
| `canary/checkpoints/step-000000/adapter_model.safetensors` | `a59a8a4f42c89f6788b2069024a2c87ca66b536bbca451baaa5c00bb9f364c00` |
| `canary/checkpoints/step-000000/training_state.json` | `8a4c5db0413e62e4599b86afc119188df2e12bcf88c51aad586e8435eb43807d` |
| `canary/checkpoints/step-000000/validation_metrics.json` | `52f36dff258a23034874cdcafc4c846a9f78a16913b053b3deb0f9d678e66066` |
| `canary/checkpoints/step-000010/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000010/adapter_config.json` | `1817a64f405deaf3f621d0c018725bbc26821ef656fc7d9ce37edf7950e9d85a` |
| `canary/checkpoints/step-000010/adapter_model.safetensors` | `ddd2a1773bdeb24eb2c23ab4a9a76f4533b8184e67becd7d4447894354290553` |
| `canary/checkpoints/step-000010/training_state.json` | `d9f4e1d69f48166846de7a43509ecb44a88c4574d882ac7b4bad41c9f5deffaa` |
| `canary/checkpoints/step-000010/validation_metrics.json` | `1e637b8151daa0e0c580e3fd8c48e4df1cf96559a128a166be8402eb3f3c0a10` |
| `canary/provenance.json` | `3680e915de3b82bc0d05bce07e0bf9ecf8b7807e7e2c6720c6a7230e807105c9` |
| `canary/run_config.json` | `67f86d0dc8c0bb10ccfe8feaefc5a4f8cd9528c1bdb3e94b5d2524b33fe9029e` |
| `canary/split_manifest.json` | `b9f0ece9e9ebd507ad326043405617aa2684d128b93f89ee5e0700607b74a7ae` |
| `canary/summary.json` | `cda3d77a1af7092a04cc4bb02334b3299a984eac58e69fbe74f70dbedf3dadb3` |
| `canary/teacher_label_audit.json` | `d44ab39b9091c15360528491c371f76b07a290f0e01e302e3080d9d10af5d08a` |
| `canary/teacher_labels.jsonl` | `17f6bf7324d53a7065943bc6a692d0f600cd3ee4401b2a0e2f90514d4a65000a` |
| `canary/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `canary/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `canary/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `canary/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `canary/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `canary/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `canary/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `canary/training_metrics.jsonl` | `f8d10f0fb944e4b68bf7c55ba92e157fc1069a2e8fbd60a1c604e3c6c5e99898` |
| `canary/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `canary/validation_metrics.jsonl` | `9dde3e5ceb3d8e8f77bfb56bc09af4db367e69a692399a80dfb7127a85c8a0f4` |
| `canary/validation_scores/step-000000.jsonl` | `84f03ed078a745aa561abaf26d1ff4db0faae7beb988b51c03076d321d6c246c` |
| `canary/validation_scores/step-000010.jsonl` | `9a4bddc5b97eef29ebfbba62f17eefe922efeb3d8bb9f531a86f34d3dca73886` |
| `id_lock/lock_manifest.json` | `c00c32c6b1badce993b80d5340c75f57096c6ad766fed81b90aa201a1b93490d` |
| `m0/compositional.json` | `2dbc1faedda196a4248dec6d5efc6c31a0fe94230940efe593fb84f68abd69e7` |
| `m0/compositional.json.metadata` | `176e0977db27cce4e4423b3d78a5923529ecce011040b9a17a365cf6dc46b943` |
| `m0/decoupled.json` | `09179319ad4f255faf88f0863036be7271beb56790e665d77812e8712cbe62a8` |
| `m0/decoupled.json.metadata` | `d0d6bbaa6a74477e249d270ff8d8402253f292e79b1797dd54c1144696b55b35` |
| `m0/evoked.json` | `ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed` |
| `m0/evoked.json.metadata` | `6b28372ab108a6e9c61a968f9a75419447e9b78902b4c375eb9d5165e3437938` |
| `m0/evoked_g2.json` | `041028d1806f58baac05efe9f285559ad8cd7b1fc3496810f5e2864059f91fa3` |
| `m0/evoked_g2.json.metadata` | `647ac5096c0b0d7eb0d2521bb1264a94169a9ca8c77d586d06fbe5bf3c98ac1d` |
| `m0/explicit.json` | `2189e6e5333d33afdd1b56d9e4a20307ce83c51ba0ed45d13cebf2877a306488` |
| `m0/explicit.json.metadata` | `729955b848ee49ca998cc200120c0504be9a2fa7f1258495fcffa76195bc7180` |
| `m0/gate.json` | `001eed7ce663fcff9a61a54267e2142826f16e3e9028bd0decf97a9d08e8b4a9` |
| `m0/gate.md` | `3cf78a95985b39c51a57fcbe1f3026a8f8525e69a0582a4fb53b4dfd97ade5c6` |
| `m1/` | `39bb1be843a002c79eb05bf69da260601e205411f719c53cda462c2b448207b8` |
| `m1/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000000/adapter_config.json` | `df9c3cab976f9859a0f5f8e31f0afd4fb9e5826518df62cf2bfb028265d9cd87` |
| `m1/checkpoints/step-000000/adapter_model.safetensors` | `a59a8a4f42c89f6788b2069024a2c87ca66b536bbca451baaa5c00bb9f364c00` |
| `m1/checkpoints/step-000000/training_state.json` | `2a34ead23178894297fb3986b85c62c73f394ea995d8145fa91ee2afa11f8548` |
| `m1/checkpoints/step-000000/validation_metrics.json` | `52f36dff258a23034874cdcafc4c846a9f78a16913b053b3deb0f9d678e66066` |
| `m1/checkpoints/step-000100/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000100/adapter_config.json` | `df9c3cab976f9859a0f5f8e31f0afd4fb9e5826518df62cf2bfb028265d9cd87` |
| `m1/checkpoints/step-000100/adapter_model.safetensors` | `6246cc8341d6ced0bb668678a222c915c284fc06e623c2a36890a049f0231c61` |
| `m1/checkpoints/step-000100/training_state.json` | `41a698a10848fcf06ff480f7f6ecc27c4d0ac5545b7adced10b667723051ce8d` |
| `m1/checkpoints/step-000100/validation_metrics.json` | `7efceb4b9ac575cfbb0abb69e632ddbb5bbe5b756624637d4a410a9f5cd57add` |
| `m1/checkpoints/step-000250/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000250/adapter_config.json` | `df9c3cab976f9859a0f5f8e31f0afd4fb9e5826518df62cf2bfb028265d9cd87` |
| `m1/checkpoints/step-000250/adapter_model.safetensors` | `3042cedafe9e675a88401a27d9e47ac7efd809efd3c87a1394cf43dc4edea65e` |
| `m1/checkpoints/step-000250/training_state.json` | `2e227a2b50bacf2d737e77e98a31006889b5678b095fd1492a1c3845889ecfd8` |
| `m1/checkpoints/step-000250/validation_metrics.json` | `2e2d301b49fad9e21529c1ad3341b955b3e6bdaca536492f0639ac3b5120e2d7` |
| `m1/checkpoints/step-000414/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000414/adapter_config.json` | `df9c3cab976f9859a0f5f8e31f0afd4fb9e5826518df62cf2bfb028265d9cd87` |
| `m1/checkpoints/step-000414/adapter_model.safetensors` | `fbd50fc3b84f68dcb6f4b0808cfe46b54ca33d853cd05eab91189adaac314a37` |
| `m1/checkpoints/step-000414/training_state.json` | `a6d06b9cfd71a481236554164940db3d852acc65dd0a6f019fd1f4ae7b522cf9` |
| `m1/checkpoints/step-000414/validation_metrics.json` | `006e75b23ba54203d67347be377f36183348039c994ea755c0c7105d0d68f285` |
| `m1/lock_manifest.json` | `6d87e691f48c23c1addee45b9fa725c85343fa56228ef38214ed3915658546a6` |
| `m1/provenance.json` | `3680e915de3b82bc0d05bce07e0bf9ecf8b7807e7e2c6720c6a7230e807105c9` |
| `m1/run_config.json` | `df2eca64746d59f3d28fcb24fa4034b96f3d73fe5fc22dc8d0c1fe5ccf21b586` |
| `m1/split_manifest.json` | `b9f0ece9e9ebd507ad326043405617aa2684d128b93f89ee5e0700607b74a7ae` |
| `m1/summary.json` | `6d2f196187c41548903f3125fc7df16d869e52aedcf03fa807ff57add420f2de` |
| `m1/teacher_label_audit.json` | `d44ab39b9091c15360528491c371f76b07a290f0e01e302e3080d9d10af5d08a` |
| `m1/teacher_labels.jsonl` | `17f6bf7324d53a7065943bc6a692d0f600cd3ee4401b2a0e2f90514d4a65000a` |
| `m1/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `m1/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `m1/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `m1/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `m1/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `m1/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `m1/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `m1/training_metrics.jsonl` | `1ce6ea9c458db6f285b5a31d2cbe47b1a8113128fdb48326a77e2feb166864df` |
| `m1/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `m1/validation_metrics.jsonl` | `df35f571f74b2aa22e2da8b7051f5871de04a95322f24a750f3f705cb61f2065` |
| `m1/validation_scores/step-000000.jsonl` | `84f03ed078a745aa561abaf26d1ff4db0faae7beb988b51c03076d321d6c246c` |
| `m1/validation_scores/step-000100.jsonl` | `05377e3db6daa8d1ccfb172d7552935950e95743eaabe96332726f03ad5f7eb8` |
| `m1/validation_scores/step-000250.jsonl` | `75f1527522b7cee4f52d4af7b85a7ca9c838cf01e4dcc679102d1cd62e3c970e` |
| `m1/validation_scores/step-000414.jsonl` | `478fb0b99284a5f853ccb8d81889dcaf8ba6f28746edd3b3461dfc71450156d8` |
| `ood/result.json` | `bf846d0899e0eccadcabb6013915456c178d8b60986a7c761d2052ec77f41d03` |

## 18. No H100 job was launched

No H100 job was launched. This invocation used the single verified NVIDIA RTX A100 and stops here for manual review; it launched no M2 or later stage.
