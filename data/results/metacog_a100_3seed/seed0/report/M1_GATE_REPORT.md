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
    "allocated_bytes": 17140921856,
    "peak_allocated_bytes": 17441162240,
    "peak_reserved_bytes": 18807259136,
    "reserved_bytes": 18807259136
  },
  "status": "PASS",
  "throughput": {
    "mean_examples_per_second": 2.7383576085394226,
    "mean_tokens_per_second": 316.1502140729073
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
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked.json.metadata",
        "metadata_sha256": "4353e452b8aef0cb75fb74fed409a02f5bf6149898bafd445e922b3be87ecd66",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked.json",
        "results_sha256": "ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed",
        "source": "evoked",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json",
        "battery_sha256": "32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked_g2.json.metadata",
        "metadata_sha256": "537ca17f384f858a12caf29563a291fc8d9e34632fe19c4d5ef7b9b0e984370e",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked_g2.json",
        "results_sha256": "041028d1806f58baac05efe9f285559ad8cd7b1fc3496810f5e2864059f91fa3",
        "source": "evoked_g2",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json",
        "battery_sha256": "35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/explicit.json.metadata",
        "metadata_sha256": "5800e63fc5da30ead958f0f47cc227ecba4fea06485135d8be2f439e206f6b62",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/explicit.json",
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
    "enabled_yes_probability": 0.19898326694965363,
    "fixed_candidate_id": "evoked:episode:000006:candidate:000",
    "logits_all_finite": true,
    "lora_layers_audited": 252,
    "max_abs_enabled_before_after": 0.0,
    "passed": true,
    "performed": true
  },
  "canary_checkpoint_save_load": {
    "all_tensors_finite": true,
    "checkpoint_tree_sha256": "ae257d1241df05963675eb30f52636fa1096026b93446838fd2c684f04e8e374",
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
    "max_w_rr": 0.006756756756756757,
    "min_w_rr": 0.0002736726874657909,
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
    "final_loss": 1.0549421310424805,
    "first_loss": 29.929688453674316,
    "gpu_memory": {
      "allocated_bytes": 16749801984,
      "peak_allocated_bytes": 17467631616,
      "peak_reserved_bytes": 18813550592,
      "reserved_bytes": 18813550592
    },
    "maximum_gradient_norm": 137.93052673339844,
    "maximum_loss": 29.929688453674316,
    "minimum_gradient_norm": 4.2091450691223145,
    "minimum_loss": 0.08862085081636906,
    "optimizer_steps_recorded": 414,
    "throughput": {
      "mean_examples_per_second": 2.811248026634295,
      "mean_tokens_per_second": 316.4701903487789
    }
  }
}
```

## 7. ID checkpoint-selection table

| Step | ID verbal AUC | ID Yes rate | Path | Selected |
|---:|---:|---:|---|:---:|
| 0 | 0.553 | 0.414 | `checkpoints/step-000000` | no |
| 100 | 0.432 | 0.195 | `checkpoints/step-000100` | no |
| 250 | 0.558 | 0.000 | `checkpoints/step-000250` | no |
| 414 | 0.613 | 0.591 | `checkpoints/step-000414` | yes |

Selection used ID verbal AUC only; ties select the earliest checkpoint.

## 8. Locked checkpoint

- Path: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m1/checkpoints/step-000414`
- Step: `414`
- ID verbal AUC: `0.613`
- Tree SHA-256: `08ae9c7a2d36dfa960b64369970624969ef7e408ae3f0ffd36ef1a6bce3894b4`

## 9. Decoupled V before/after

| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | Before within-episode | After within-episode | Yes rate before | Yes rate after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decoupled | V | 0.342 | 0.566 | +0.225 | 0.213 | 0.541 | 0.000 | 0.639 |
| Decoupled | W_rr | 0.655 | 0.655 | +0.000 | 0.665 | 0.660 | — | — |
| Compositional | V | 0.326 | 0.426 | +0.100 | 0.189 | 0.416 | 0.000 | 0.705 |
| Compositional | W_rr | 0.547 | 0.550 | +0.003 | 0.566 | 0.565 | — | — |

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
| Compositional | 0.712 | 0.712 | +0.000 | +0.000 |

## 15. Bootstrap CIs

All paired AUC intervals use 4000 whole-episode cluster draws.

| Condition | Channel | Delta estimate | 95% CI | Effective draws |
|---|---|---:|---:|---:|
| Decoupled | V | +0.225 | [+0.148, +0.304] | 4000 |
| Decoupled | W_rr | +0.000 | [-0.003, +0.003] | 4000 |
| Compositional | V | +0.100 | [+0.003, +0.199] | 4000 |
| Compositional | W_rr | +0.003 | [-0.001, +0.007] | 4000 |

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
| `canary/` | `97f94a0e542276e1b0ed00937d76bb084efdf070350bde8622fe0faf925cd170` |
| `canary/canary_manifest.json` | `c9f51ed72fbaf01b542c00925d78712424664da9a5dd587d3c673b9b0cca5ae3` |
| `canary/canary_workspace_id.jsonl` | `bc2ff3d36ade2057ec8f7e723fd7f62595ba7d7c98fc6e5ff12162aa63dd5830` |
| `canary/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000000/adapter_config.json` | `9e6a1f4016881a90983bbe0ca9ac7855b731dbfed57da7e150876a6011cc4fde` |
| `canary/checkpoints/step-000000/adapter_model.safetensors` | `8c3ced3fa2bb076919a6c632630b845507d284cc8033df5bfaf769d80262af35` |
| `canary/checkpoints/step-000000/training_state.json` | `c72e27c43ea7276f39253d3c3052265339c8963151dba13c3bad35272d40d978` |
| `canary/checkpoints/step-000000/validation_metrics.json` | `52f36dff258a23034874cdcafc4c846a9f78a16913b053b3deb0f9d678e66066` |
| `canary/checkpoints/step-000010/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000010/adapter_config.json` | `9e6a1f4016881a90983bbe0ca9ac7855b731dbfed57da7e150876a6011cc4fde` |
| `canary/checkpoints/step-000010/adapter_model.safetensors` | `c3f39bf359ec4f0127eaf8a646b45c328adb16c94f313196166bac49c16da319` |
| `canary/checkpoints/step-000010/training_state.json` | `6067b1f2b0068404ba1cf5c9ac57579ec075127240484cfc210c3fe145184ed2` |
| `canary/checkpoints/step-000010/validation_metrics.json` | `98ff7fffd41c6e23e4df12326198809e77854975bf28ead08f6ed84babaf7133` |
| `canary/provenance.json` | `1dd39524167b306190c166632950a877eab7977daa1ef5fbe54b9cdc861aca75` |
| `canary/run_config.json` | `84088648176dfd800796010323c949080580f4bd364720e955713b138d824162` |
| `canary/split_manifest.json` | `aaf3f412419c15290197e7e3cdbf30a1d9d09142b8af5963b0d0eefbce981510` |
| `canary/summary.json` | `ac5831f0d4a4d415cee0f1239b983c329cd38c66f7f9f531a92fc3d3d9b17e1a` |
| `canary/teacher_label_audit.json` | `d44ab39b9091c15360528491c371f76b07a290f0e01e302e3080d9d10af5d08a` |
| `canary/teacher_labels.jsonl` | `17f6bf7324d53a7065943bc6a692d0f600cd3ee4401b2a0e2f90514d4a65000a` |
| `canary/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `canary/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `canary/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `canary/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `canary/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `canary/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `canary/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `canary/training_metrics.jsonl` | `3db7aa39c6984fb5b15467b2ba221dccdf2c8fb48bdd391500850b855c37aeac` |
| `canary/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `canary/validation_metrics.jsonl` | `a0efce3cf2c17f3e6ffc4a8bc1fa09ea9984affd74e55799c7082f062849f5a3` |
| `canary/validation_scores/step-000000.jsonl` | `84f03ed078a745aa561abaf26d1ff4db0faae7beb988b51c03076d321d6c246c` |
| `canary/validation_scores/step-000010.jsonl` | `d68c18a36d0ded09891859749a42e4a1aa5cff18b46755e5ed51b89b3328af80` |
| `id_lock/lock_manifest.json` | `52c837bb010e070433bc341f9ed469e51473b63561d898b75c3bf23a73dd7ea7` |
| `m0/compositional.json` | `2dbc1faedda196a4248dec6d5efc6c31a0fe94230940efe593fb84f68abd69e7` |
| `m0/compositional.json.metadata` | `fedbece11ef8631fee86d0add551c288a865c764caef74d55af351b7d6a0cd7a` |
| `m0/decoupled.json` | `09179319ad4f255faf88f0863036be7271beb56790e665d77812e8712cbe62a8` |
| `m0/decoupled.json.metadata` | `924cc5ce5b756171c83fc8b13bc418dab9d5ae077972bd7bb16ca154fd18e15e` |
| `m0/evoked.json` | `ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed` |
| `m0/evoked.json.metadata` | `4353e452b8aef0cb75fb74fed409a02f5bf6149898bafd445e922b3be87ecd66` |
| `m0/evoked_g2.json` | `041028d1806f58baac05efe9f285559ad8cd7b1fc3496810f5e2864059f91fa3` |
| `m0/evoked_g2.json.metadata` | `537ca17f384f858a12caf29563a291fc8d9e34632fe19c4d5ef7b9b0e984370e` |
| `m0/explicit.json` | `2189e6e5333d33afdd1b56d9e4a20307ce83c51ba0ed45d13cebf2877a306488` |
| `m0/explicit.json.metadata` | `5800e63fc5da30ead958f0f47cc227ecba4fea06485135d8be2f439e206f6b62` |
| `m0/gate.json` | `c3e7a2530c314da2c9a41ee85a2d7262cadd65c28e4c304e14aec63526d54e83` |
| `m0/gate.md` | `82922c0da1ba831201fa70047acc77c8a45c09341851f8592022939d59f14a78` |
| `m1/` | `6437aad22d3513d06bae51969b957260771abc2266d1bd0856cf464c223900eb` |
| `m1/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000000/adapter_config.json` | `e42a99f07da1032e9519c4b81845a89628dfdc7565c2afe9cf823e7722914991` |
| `m1/checkpoints/step-000000/adapter_model.safetensors` | `8c3ced3fa2bb076919a6c632630b845507d284cc8033df5bfaf769d80262af35` |
| `m1/checkpoints/step-000000/training_state.json` | `ca073b360ca0a0a3477fdc19c5d9f4d2ed3a073a067e86987af866276c5edf6b` |
| `m1/checkpoints/step-000000/validation_metrics.json` | `52f36dff258a23034874cdcafc4c846a9f78a16913b053b3deb0f9d678e66066` |
| `m1/checkpoints/step-000100/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000100/adapter_config.json` | `e42a99f07da1032e9519c4b81845a89628dfdc7565c2afe9cf823e7722914991` |
| `m1/checkpoints/step-000100/adapter_model.safetensors` | `0af783581b83b9d25735af781cad861bde1ea05b0c75cbb949424a03320b7b2f` |
| `m1/checkpoints/step-000100/training_state.json` | `70878a25e42a3d3d154ad8d94c5f7d0aae5dd38ad5ef7b67ce339ed7b3714c47` |
| `m1/checkpoints/step-000100/validation_metrics.json` | `0e508f776e3acf8223acf57630b707f579108c108fbf374c9351fd5687b38e21` |
| `m1/checkpoints/step-000250/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000250/adapter_config.json` | `e42a99f07da1032e9519c4b81845a89628dfdc7565c2afe9cf823e7722914991` |
| `m1/checkpoints/step-000250/adapter_model.safetensors` | `38f7bcae18efc3b590023180239253e13e7a594e1f2d50829b5047ff8859972f` |
| `m1/checkpoints/step-000250/training_state.json` | `c7c42042ee2d7d16b460645ac2673deb2a8e027ff60d4dda49427ef0d0e03100` |
| `m1/checkpoints/step-000250/validation_metrics.json` | `fababfcd75a1a0ee06b37c121ac40a77f44921b468b32b9991b2e848eb3d277c` |
| `m1/checkpoints/step-000414/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000414/adapter_config.json` | `e42a99f07da1032e9519c4b81845a89628dfdc7565c2afe9cf823e7722914991` |
| `m1/checkpoints/step-000414/adapter_model.safetensors` | `2ac5d2382829d5515d55513f6c81bbcfa705373ca727e3ddafb94ee9c4a90527` |
| `m1/checkpoints/step-000414/training_state.json` | `4dff8d296e1e10a42055deddb0213d4bc8fc07d8170a716879a7975a08be1aa1` |
| `m1/checkpoints/step-000414/validation_metrics.json` | `b464a3e3bc3e8c6059fac480e9bcfcc0ed63fcdbb0a11a7148f1cd3ddbc0266f` |
| `m1/lock_manifest.json` | `978f3926577f3af123d2354720a15a745ffe1e0abb0875f153ce0d0d9d136a7b` |
| `m1/provenance.json` | `1dd39524167b306190c166632950a877eab7977daa1ef5fbe54b9cdc861aca75` |
| `m1/run_config.json` | `141fa11b7612f44762008cf65e1af7fce850e40a272d7c70c4bb53262965eb97` |
| `m1/split_manifest.json` | `aaf3f412419c15290197e7e3cdbf30a1d9d09142b8af5963b0d0eefbce981510` |
| `m1/summary.json` | `8ba9cbd21f2d1a1d5b191737f7df8745b3ae865b31199847b937aafe877fc997` |
| `m1/teacher_label_audit.json` | `d44ab39b9091c15360528491c371f76b07a290f0e01e302e3080d9d10af5d08a` |
| `m1/teacher_labels.jsonl` | `17f6bf7324d53a7065943bc6a692d0f600cd3ee4401b2a0e2f90514d4a65000a` |
| `m1/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `m1/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `m1/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `m1/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `m1/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `m1/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `m1/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `m1/training_metrics.jsonl` | `7f31c5d159c9d241d3bc6401458f0cf0656f8b95bb5d46dbc89188e839b1ad7e` |
| `m1/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `m1/validation_metrics.jsonl` | `83c2c860dc131ff4520c84d8322cf88634f3c51dbdacc5398de4896504d98f71` |
| `m1/validation_scores/step-000000.jsonl` | `84f03ed078a745aa561abaf26d1ff4db0faae7beb988b51c03076d321d6c246c` |
| `m1/validation_scores/step-000100.jsonl` | `1d45c8cc60458a65f14e884dc567b03c12529e59a384b8e1ced1778f0d816c91` |
| `m1/validation_scores/step-000250.jsonl` | `53527d33fd0752117f12863143281c7b9a48a89ea4a0bed0667b0de6542a83a0` |
| `m1/validation_scores/step-000414.jsonl` | `60d3e79d009f05246c40250bf299837226644708721a108c5a1927d791b45c85` |
| `ood/result.json` | `686b13b4dca2420e6d7ce3b2b0afd32ce8f94d69b85948600387d6fe941e131c` |

## 18. No H100 job was launched

No H100 job was launched. This invocation used the single verified NVIDIA RTX A100 and stops here for manual review; it launched no M2 or later stage.
