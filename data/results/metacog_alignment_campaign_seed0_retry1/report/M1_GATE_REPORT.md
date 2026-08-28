# Qwen3-8B A5000 Metacognitive Alignment M1 gate report

Report schema: `metacog-alignment-m1-report/v1`. Final A5000 decision: **GREEN**.

## 1. M0 baseline reproduction status

M0 decision: **GREEN**. Reference and tolerance:

```json
{
  "absolute_delta": {
    "V": 0.006908349856796614,
    "W_rr": 5.640008812513031e-05
  },
  "observed": {
    "V": 0.34390834985679664,
    "W_rr": 0.6539435999118749
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

## 3. A5000 memory/throughput configuration

Preflight device: `NVIDIA RTX A5000`, total 24564 MiB, free 24098 MiB. Canary runtime summary:

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
    "mean_examples_per_second": 1.805390946949426,
    "mean_tokens_per_second": 207.92272087648558
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
        "battery_path": "/home/myid/jc43275/jspace-devtrace/data/benchmarks/battery_v2_final.json",
        "battery_sha256": "e8ce4db85d3a0c09879575850e36b91821af8be155e86495cd059ae35485e7d6",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m0/evoked.json.metadata",
        "metadata_sha256": "85da14b8265ba37ba4ba2922a859ea9e5e1879523b180a66d48ca8293d4e4351",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m0/evoked.json",
        "results_sha256": "fe095021f528cbf8b3bd3e76262fb6e2e856cc224c0ef8fc2a06690404d2a7f6",
        "source": "evoked",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/home/myid/jc43275/jspace-devtrace/data/benchmarks/battery_v2_g2.json",
        "battery_sha256": "32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m0/evoked_g2.json.metadata",
        "metadata_sha256": "7743c833bd41bad7a61a91a783ba0246e487bbde55e354d821e2efb0dd2485e5",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m0/evoked_g2.json",
        "results_sha256": "8d7e745edd7275b5f557ee77eb185338969d660b87d5565a1e8be1543a18d0da",
        "source": "evoked_g2",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218"
      },
      {
        "adapter": null,
        "battery_path": "/home/myid/jc43275/jspace-devtrace/data/benchmarks/battery_v1_final.json",
        "battery_sha256": "35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m0/explicit.json.metadata",
        "metadata_sha256": "bf8774065c5e4d0a637b40611c0868c05cd2907047e941da4cf5b0477a72d855",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "results_path": "/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m0/explicit.json",
        "results_sha256": "589a8363ffd69f6dd211827c680515396dfdbcb7c2beea16ab00b7cc4c360f12",
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
          "episode_id": "evoked_g2:episode:000053",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000012",
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
      "episodes_with_any_tie": 9,
      "episodes_with_top_k_boundary_tie": 2,
      "tie_groups": 9,
      "tied_candidates": 18
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
          "boundary_score": 0.0029585798816568047,
          "episode_id": "evoked:episode:000074",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": 0.0011086474501108647,
          "episode_id": "explicit:episode:000019",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": 0.010638297872340425,
          "episode_id": "explicit:episode:000047",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        }
      ],
      "episodes": 45,
      "episodes_with_any_tie": 4,
      "episodes_with_top_k_boundary_tie": 3,
      "tie_groups": 4,
      "tied_candidates": 8
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
      6
    ],
    "index": 0,
    "name": "NVIDIA RTX A5000",
    "total_memory_bytes": 25283526656,
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
    "accelerate": "1.10.1",
    "cuda": "12.8",
    "peft": "0.13.2",
    "python": "3.9.12",
    "safetensors": "0.7.0",
    "torch": "2.8.0",
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
    "disabled_yes_probability": 0.23210713267326355,
    "enabled_logits_restored": true,
    "enabled_yes_probability": 0.20430204272270203,
    "fixed_candidate_id": "evoked:episode:000006:candidate:000",
    "logits_all_finite": true,
    "lora_layers_audited": 252,
    "max_abs_enabled_before_after": 0.0,
    "passed": true,
    "performed": true
  },
  "canary_checkpoint_save_load": {
    "all_tensors_finite": true,
    "checkpoint_tree_sha256": "65f6c0f36c4e8f5d9327ec64dac6f0e10d651e22982781f186d7fcad050e2cde",
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
    "min_w_rr": 0.000273972602739726,
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
    "final_loss": 1.1381280422210693,
    "first_loss": 29.890625953674316,
    "gpu_memory": {
      "allocated_bytes": 16749801984,
      "peak_allocated_bytes": 17467631616,
      "peak_reserved_bytes": 18813550592,
      "reserved_bytes": 18813550592
    },
    "maximum_gradient_norm": 137.81326293945312,
    "maximum_loss": 29.890625953674316,
    "minimum_gradient_norm": 4.265919208526611,
    "minimum_loss": 0.1099916473031044,
    "optimizer_steps_recorded": 414,
    "throughput": {
      "mean_examples_per_second": 2.247968429826352,
      "mean_tokens_per_second": 252.9690033286066
    }
  }
}
```

## 7. ID checkpoint-selection table

| Step | ID verbal AUC | ID Yes rate | Path | Selected |
|---:|---:|---:|---|:---:|
| 0 | 0.552 | 0.372 | `checkpoints/step-000000` | no |
| 100 | 0.439 | 0.121 | `checkpoints/step-000100` | no |
| 250 | 0.603 | 0.005 | `checkpoints/step-000250` | yes |
| 414 | 0.542 | 0.893 | `checkpoints/step-000414` | no |

Selection used ID verbal AUC only; ties select the earliest checkpoint.

## 8. Locked checkpoint

- Path: `/home/myid/jc43275/jspace-devtrace/data/results/metacog_alignment_campaign_seed0_retry1/m1/checkpoints/step-000250`
- Step: `250`
- ID verbal AUC: `0.603`
- Tree SHA-256: `6f91e4ed635a34e1b2169ace97dad2afab28c7de8c29a4e7fa5f5326404726f0`

## 9. Decoupled V before/after

| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | Before within-episode | After within-episode | Yes rate before | Yes rate after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decoupled | V | 0.344 | 0.655 | +0.311 | 0.247 | 0.675 | 0.000 | 0.003 |
| Decoupled | W_rr | 0.654 | 0.655 | +0.001 | 0.665 | 0.664 | — | — |
| Compositional | V | 0.328 | 0.501 | +0.173 | 0.214 | 0.525 | 0.000 | 0.004 |
| Compositional | W_rr | 0.546 | 0.546 | +0.001 | 0.566 | 0.562 | — | — |

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
| Compositional | 0.731 | 0.692 | -0.038 | +3.846 |

## 15. Bootstrap CIs

All paired AUC intervals use 4000 whole-episode cluster draws.

| Condition | Channel | Delta estimate | 95% CI | Effective draws |
|---|---|---:|---:|---:|
| Decoupled | V | +0.311 | [+0.238, +0.392] | 4000 |
| Decoupled | W_rr | +0.001 | [-0.002, +0.004] | 4000 |
| Compositional | V | +0.173 | [+0.065, +0.286] | 4000 |
| Compositional | W_rr | +0.001 | [-0.003, +0.004] | 4000 |

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
| `canary/` | `10c180d0fe03c2e826b5210b9dfe08d05dd6b8ddc066c0f6d243918031b49419` |
| `canary/canary_manifest.json` | `077751660b4825046b8b0e4b9a63f9ab42792a3bdaa24548d0a494169d57294c` |
| `canary/canary_workspace_id.jsonl` | `263db5e0b0afa0d81c79d9613cc5b4e8cacb025c0ffa0b0fd7b0cef9e16aa04f` |
| `canary/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000000/adapter_config.json` | `8bc8b1b554efd7fed7c121b2012aae4664109106391165760f1218444c9da3f2` |
| `canary/checkpoints/step-000000/adapter_model.safetensors` | `8c3ced3fa2bb076919a6c632630b845507d284cc8033df5bfaf769d80262af35` |
| `canary/checkpoints/step-000000/training_state.json` | `80bdc7c6937b27b7f60fc946a907739597ec107275436706240047edf261b540` |
| `canary/checkpoints/step-000000/validation_metrics.json` | `b3e5ee9a0b15a446d6f5381ab1aba2f7cddc3a439dc625bcaf5a95548f6bea7d` |
| `canary/checkpoints/step-000010/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `canary/checkpoints/step-000010/adapter_config.json` | `8bc8b1b554efd7fed7c121b2012aae4664109106391165760f1218444c9da3f2` |
| `canary/checkpoints/step-000010/adapter_model.safetensors` | `ecaeddd34af9dfd015b2a0b574ab501ffd9ff610307ccd57c400c40644375b70` |
| `canary/checkpoints/step-000010/training_state.json` | `9f135e1a1d1b1d184456157e121950d5c8355a5b12576a12a5e19496d6f4d765` |
| `canary/checkpoints/step-000010/validation_metrics.json` | `f624960811b1022e3960d38ecde88de8188e53814fe794089c320c235671cf29` |
| `canary/provenance.json` | `06bf1b145697f2d9098a156e749b02974806aa267299c6138107589a06ff5c06` |
| `canary/run_config.json` | `57e30cf2034fae648c068bce2bc3634343c418e2a278bae985bb5f580221addb` |
| `canary/split_manifest.json` | `95484ccb2b9e2fd6058866e9fa0952c6553ef5e687d7428d55ddcf3b5ead4fd2` |
| `canary/summary.json` | `14e65422bd311d435912b4ccfcb42915943893ef4bb35497d3835735fd4c4917` |
| `canary/teacher_label_audit.json` | `480b5eb1ac0aada265c31c10e298bb053518f9050adb5bbbf03499effa31d0be` |
| `canary/teacher_labels.jsonl` | `de761bb43cdf3f1b195f8106acfa8b62e4cffbbdff2106c13a99044dfd3fd7b5` |
| `canary/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `canary/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `canary/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `canary/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `canary/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `canary/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `canary/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `canary/training_metrics.jsonl` | `40ddd065ea111998fb950e74ab15e63ea22b123046f2a8d9d60e7863a0b768a1` |
| `canary/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `canary/validation_metrics.jsonl` | `13d298aa17a0a9acfedceaccf17c18c94473e4e9d817aa19071a0c4a3a432bb3` |
| `canary/validation_scores/step-000000.jsonl` | `d787fbd77f52202579fe9014662f1a97ad59f7372fde88f630b3c7024942d426` |
| `canary/validation_scores/step-000010.jsonl` | `9c7dfc0b351c5c209a61c7156f168ae51e87b71ab5e7581ee3c511b54854d21a` |
| `id_lock/lock_manifest.json` | `f4bbeaccef7268a9687606abbadeb803b6d0099a14f5880013863e61837f6946` |
| `m0/compositional.json` | `6bbc5faa064334ce7e5579369384c88448e8517d3a6b8c99c82afac7cd377e5f` |
| `m0/compositional.json.metadata` | `63f28aca6367c364ea5a383714f3156d138eb25768263c1232171655b047defc` |
| `m0/decoupled.json` | `591e3ad9618ee8242c41c397e76ef3fd42f8b254c823d0e1ef0e96b5bbc11e8d` |
| `m0/decoupled.json.metadata` | `c80c582fdf435175d816bb4c88131b68f5c49e590981938d27f4ff8ae6add77d` |
| `m0/evoked.json` | `fe095021f528cbf8b3bd3e76262fb6e2e856cc224c0ef8fc2a06690404d2a7f6` |
| `m0/evoked.json.metadata` | `85da14b8265ba37ba4ba2922a859ea9e5e1879523b180a66d48ca8293d4e4351` |
| `m0/evoked_g2.json` | `8d7e745edd7275b5f557ee77eb185338969d660b87d5565a1e8be1543a18d0da` |
| `m0/evoked_g2.json.metadata` | `7743c833bd41bad7a61a91a783ba0246e487bbde55e354d821e2efb0dd2485e5` |
| `m0/explicit.json` | `589a8363ffd69f6dd211827c680515396dfdbcb7c2beea16ab00b7cc4c360f12` |
| `m0/explicit.json.metadata` | `bf8774065c5e4d0a637b40611c0868c05cd2907047e941da4cf5b0477a72d855` |
| `m0/gate.json` | `59ecb380358604f184e5e058ced1cf019b3e470275842ee0ad7af3b2dec02941` |
| `m0/gate.md` | `4ca24d9586190c45693d2a0f297b26b00feddcffd07a93f1889ff024f427f87c` |
| `m1/` | `26cf381bfbab44f99ee8f952843435f89b5faa408d2be7aaf2de0f48f8c049d3` |
| `m1/checkpoints/step-000000/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000000/adapter_config.json` | `99884e0c2d487917ac5533478616aa0cffe6fa42147845a977558621faf6042d` |
| `m1/checkpoints/step-000000/adapter_model.safetensors` | `8c3ced3fa2bb076919a6c632630b845507d284cc8033df5bfaf769d80262af35` |
| `m1/checkpoints/step-000000/training_state.json` | `cc5a154fcb2ba5ecd1261f29c069f0abedd465445a51477d100a1a45c3e87fb2` |
| `m1/checkpoints/step-000000/validation_metrics.json` | `b3e5ee9a0b15a446d6f5381ab1aba2f7cddc3a439dc625bcaf5a95548f6bea7d` |
| `m1/checkpoints/step-000100/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000100/adapter_config.json` | `99884e0c2d487917ac5533478616aa0cffe6fa42147845a977558621faf6042d` |
| `m1/checkpoints/step-000100/adapter_model.safetensors` | `5bbc7af4edb94818bb553c9a8236e3a2ffb03860d48e8ebb300391f1bc3a4d55` |
| `m1/checkpoints/step-000100/training_state.json` | `86154dd977c04cfbd8db478ad9102b30e0a3138c271a4f37614c56d2ef3502a2` |
| `m1/checkpoints/step-000100/validation_metrics.json` | `24797e4db8e9c75388162c2a1de9a8711e65ac056179c5d73f6e9ec4898c9dc0` |
| `m1/checkpoints/step-000250/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000250/adapter_config.json` | `99884e0c2d487917ac5533478616aa0cffe6fa42147845a977558621faf6042d` |
| `m1/checkpoints/step-000250/adapter_model.safetensors` | `a36d998c795aa4d439dc567aeed6c48484da0ba674da6b5b92e7167c7e526106` |
| `m1/checkpoints/step-000250/training_state.json` | `b4f314fa809d73d3fc636b3a94b6115ab7d4c7e4ff1d09ea73a247ea5aa6f921` |
| `m1/checkpoints/step-000250/validation_metrics.json` | `cc9dd15bf38093d0501d28aaf3c911ae21b6934c6a8f17e0961933d58a81c40e` |
| `m1/checkpoints/step-000414/README.md` | `d9c7d31b4105f0870dcf81e9ea8c237b4d105e4c17cfec1ffb1a62fe15254eae` |
| `m1/checkpoints/step-000414/adapter_config.json` | `99884e0c2d487917ac5533478616aa0cffe6fa42147845a977558621faf6042d` |
| `m1/checkpoints/step-000414/adapter_model.safetensors` | `38605ad31bf322c35adc7f5ea9dba2eebe04aa88b4c7ce2425aebfadcceee3b8` |
| `m1/checkpoints/step-000414/training_state.json` | `59c31861dde5f03c2b58b390af35198cbb03f8ff522c22b9fe6b873de2dacf7a` |
| `m1/checkpoints/step-000414/validation_metrics.json` | `6659b636c33d3de3c8e1f7e9eb2e281fce87b4e525455f87d00b13bbacc2a4c1` |
| `m1/lock_manifest.json` | `a8cae80ff713ce5bc013b18f7391dad1abf8d736e83a614a5cedc9c49812c741` |
| `m1/provenance.json` | `06bf1b145697f2d9098a156e749b02974806aa267299c6138107589a06ff5c06` |
| `m1/run_config.json` | `a54197d65dfbe8d05082fa20c4b25aaf8f03a7f10eaebc9ba8eed1ba0cd5c2ca` |
| `m1/split_manifest.json` | `95484ccb2b9e2fd6058866e9fa0952c6553ef5e687d7428d55ddcf3b5ead4fd2` |
| `m1/summary.json` | `e517c5ceefcf9dc8c58ff762c11c441b082bf9e8968fcc43a0cf7b82e50192f2` |
| `m1/teacher_label_audit.json` | `480b5eb1ac0aada265c31c10e298bb053518f9050adb5bbbf03499effa31d0be` |
| `m1/teacher_labels.jsonl` | `de761bb43cdf3f1b195f8106acfa8b62e4cffbbdff2106c13a99044dfd3fd7b5` |
| `m1/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `m1/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `m1/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `m1/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `m1/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `m1/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `m1/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `m1/training_metrics.jsonl` | `7d26203cb04943eb6ac1065b8daf7c4083cc10c4b34873d8b4d54f21bddce6b1` |
| `m1/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `m1/validation_metrics.jsonl` | `82a43afb9b8c3b646f3621c250a531a053d797597453db522766d3822f6ef12b` |
| `m1/validation_scores/step-000000.jsonl` | `d787fbd77f52202579fe9014662f1a97ad59f7372fde88f630b3c7024942d426` |
| `m1/validation_scores/step-000100.jsonl` | `663ee9aca4c10b20ce35a757fb4234726cf49cd7d63244d48a0d7e7a6a5835ae` |
| `m1/validation_scores/step-000250.jsonl` | `fdbeb0b0db2a5f35fb5329cba0ab338158b34eea5dd742f379c85e66877cb741` |
| `m1/validation_scores/step-000414.jsonl` | `fda44f6fc6dc6e61d7425b8b7a2e8634ffcc549ae08029034076bd3ecf114735` |
| `ood/result.json` | `fc6092301153be7b9649f6c0efb3b0b49bf936d0d32abafcccbd145ffe1634ea` |

## 18. No H100 job was launched

No H100 job was launched. This invocation used the single verified NVIDIA RTX A5000 and stops here for manual review; it launched no M2 or later stage.
