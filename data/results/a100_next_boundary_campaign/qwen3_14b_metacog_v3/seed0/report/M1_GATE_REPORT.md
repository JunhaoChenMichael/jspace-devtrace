# Qwen3-8B A100 Metacognitive Alignment M1 gate report

Report schema: `metacog-alignment-m1-report/v1`. Final A100 decision: **GREEN**.

## 1. M0 baseline reproduction status

M0 decision: **GREEN**. Reference and tolerance:

```json
{
  "absolute_delta": null,
  "observed": {
    "V": 0.3480392156862745,
    "W_rr": 0.6676580744657413
  },
  "reference": null,
  "tolerance": null
}
```

## 2. Model/tokenizer revisions

- Model: `Qwen/Qwen3-14B`
- Model revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Tokenizer revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Chat-template SHA-256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`

## 3. A100 memory/throughput configuration

Preflight device: `NVIDIA A100-SXM4-80GB`, total 81920 MiB, free 81152 MiB. Canary runtime summary:

```json
{
  "gpu_memory": {
    "allocated_bytes": 30651291136,
    "peak_allocated_bytes": 30989561344,
    "peak_reserved_bytes": 31792824320,
    "reserved_bytes": 31792824320
  },
  "status": "PASS",
  "throughput": {
    "mean_examples_per_second": 2.5911422677910716,
    "mean_tokens_per_second": 298.9571365596821
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
    "model": "Qwen/Qwen3-14B",
    "requested_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "resolved_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "source_artifacts": [
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json",
        "battery_sha256": "e8ce4db85d3a0c09879575850e36b91821af8be155e86495cd059ae35485e7d6",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m0/evoked.json.metadata",
        "metadata_sha256": "0539edf88405356dae908457b2a2db7bfd88ed23abf28056e1db642b76af5921",
        "model": "Qwen/Qwen3-14B",
        "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m0/evoked.json",
        "results_sha256": "37ad6d1ad4b3d39a1c1ac04adfce98470aa9e383f99298b547eed42f0ebd54ef",
        "source": "evoked",
        "tokenizer_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json",
        "battery_sha256": "32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m0/evoked_g2.json.metadata",
        "metadata_sha256": "2fc0b42a9a4eb17ad6439696e1ad8753e6ad34889b238841cad2ce2dc04360ca",
        "model": "Qwen/Qwen3-14B",
        "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m0/evoked_g2.json",
        "results_sha256": "53c9ab4cf46437bb3030c8e60e552ac6e0b8d2d3a2e615cc5b09dea94b4a188e",
        "source": "evoked_g2",
        "tokenizer_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18"
      },
      {
        "adapter": null,
        "battery_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json",
        "battery_sha256": "35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "metadata_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m0/explicit.json.metadata",
        "metadata_sha256": "6e709dc2daa10eab006f50ec116926143dacfbada0f843b6e68e3d7663796c5d",
        "model": "Qwen/Qwen3-14B",
        "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "results_path": "/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m0/explicit.json",
        "results_sha256": "9711fdb145934a9cd36e7b4fcbea9ae9832cf82fbea0345d0db54448a0cb062c",
        "source": "explicit",
        "tokenizer_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18"
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
          "episode_id": "evoked:episode:000055",
          "source": "evoked",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked_g2:episode:000001",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": null,
          "episode_id": "evoked_g2:episode:000014",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        },
        {
          "boundary_score": 0.006097560975609756,
          "episode_id": "evoked_g2:episode:000053",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000046",
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
          "boundary_score": 0.005494505494505495,
          "episode_id": "evoked_g2:episode:000046",
          "source": "evoked_g2",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": true
        },
        {
          "boundary_score": null,
          "episode_id": "explicit:episode:000047",
          "source": "explicit",
          "tie_break": "candidate_index_ascending_original_order",
          "tie_groups": 1,
          "tied_candidates": 2,
          "top_k_boundary_tie": false
        }
      ],
      "episodes": 45,
      "episodes_with_any_tie": 2,
      "episodes_with_top_k_boundary_tie": 1,
      "tie_groups": 2,
      "tied_candidates": 4
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
  "model": "Qwen/Qwen3-14B",
  "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
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
  "resolved_model_commit": "40c069824f4251a91eefaf281ebe4c544efd3e18",
  "resolved_tokenizer_commit": "40c069824f4251a91eefaf281ebe4c544efd3e18",
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
  "tokenizer_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
  "total_parameter_count": 14832532480,
  "train_candidate_count": 825,
  "train_episode_count": 175,
  "trainable_parameter_count": 64225280,
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
    "disabled_yes_probability": 0.001273752306587994,
    "enabled_logits_restored": true,
    "enabled_yes_probability": 0.0004530390433501452,
    "fixed_candidate_id": "evoked:episode:000006:candidate:000",
    "logits_all_finite": true,
    "lora_layers_audited": 280,
    "max_abs_enabled_before_after": 0.0,
    "passed": true,
    "performed": true
  },
  "canary_checkpoint_save_load": {
    "all_tensors_finite": true,
    "checkpoint_tree_sha256": "f06c266a3653a15006d4234fc460ec07a279201b375136cb72fec2671e71d33b",
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
    "tensor_count": 560
  },
  "canary_finite_loss_and_gradients": true,
  "canary_workspace_evaluation": {
    "adapter_enabled": true,
    "all_finite": true,
    "artifact": "canary_workspace_id.jsonl",
    "candidate_rows": 5,
    "episode_id": "evoked:episode:000006",
    "expected_candidate_rows": 5,
    "max_w_rr": 0.004081632653061225,
    "min_w_rr": 0.0002881014116969173,
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
    "workspace_auc": 0.75
  },
  "formal_health": {
    "all_loss_and_gradients_finite": true,
    "final_loss": 0.9759061932563782,
    "first_loss": 27.82031536102295,
    "gpu_memory": {
      "allocated_bytes": 30098914816,
      "peak_allocated_bytes": 31031703552,
      "peak_reserved_bytes": 31799115776,
      "reserved_bytes": 31799115776
    },
    "maximum_gradient_norm": 202.34986877441406,
    "maximum_loss": 27.82031536102295,
    "minimum_gradient_norm": 3.9913065433502197,
    "minimum_loss": 0.1410484965890646,
    "optimizer_steps_recorded": 414,
    "throughput": {
      "mean_examples_per_second": 2.623173073805992,
      "mean_tokens_per_second": 295.25274947020426
    }
  }
}
```

## 7. ID checkpoint-selection table

| Step | ID verbal AUC | ID Yes rate | Path | Selected |
|---:|---:|---:|---|:---:|
| 0 | 0.413 | 0.000 | `checkpoints/step-000000` | no |
| 100 | 0.470 | 0.033 | `checkpoints/step-000100` | no |
| 250 | 0.575 | 0.000 | `checkpoints/step-000250` | yes |
| 414 | 0.523 | 0.893 | `checkpoints/step-000414` | no |

Selection used ID verbal AUC only; ties select the earliest checkpoint.

## 8. Locked checkpoint

- Path: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/m1/checkpoints/step-000250`
- Step: `250`
- ID verbal AUC: `0.575`
- Tree SHA-256: `b6bae4c71329ae7035564b9d2b2eaaca724bd908b6c96fa070f2852e1e5a1c6c`

## 9. Decoupled V before/after

| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | Before within-episode | After within-episode | Yes rate before | Yes rate after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decoupled | V | 0.348 | 0.731 | +0.383 | 0.333 | 0.833 | 0.000 | 0.000 |
| Decoupled | W_rr | 0.668 | 0.667 | -0.000 | 0.685 | 0.678 | — | — |
| Compositional | V | 0.322 | 0.648 | +0.326 | 0.225 | 0.697 | 0.000 | 0.000 |
| Compositional | W_rr | 0.509 | 0.511 | +0.002 | 0.502 | 0.498 | — | — |

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
| Decoupled | 0.691 | 0.691 | +0.000 | +0.000 |
| Compositional | 0.750 | 0.750 | +0.000 | +0.000 |

## 15. Bootstrap CIs

All paired AUC intervals use 4000 whole-episode cluster draws.

| Condition | Channel | Delta estimate | 95% CI | Effective draws |
|---|---|---:|---:|---:|
| Decoupled | V | +0.383 | [+0.303, +0.470] | 4000 |
| Decoupled | W_rr | -0.000 | [-0.003, +0.002] | 4000 |
| Compositional | V | +0.326 | [+0.217, +0.439] | 4000 |
| Compositional | W_rr | +0.002 | [-0.001, +0.005] | 4000 |

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
| `canary/` | `798b78821ec9c0b534c75c49b4acef3fae956a556987736b735c58308dde96bf` |
| `canary/canary_manifest.json` | `c66282a8256c8990cf4ce0ec27987c9ff28c8bd0a87c909fe96face18b59ff13` |
| `canary/canary_workspace_id.jsonl` | `8c39ed2e75f592e85990448ebbc4796d1d3b517a3b1c2fe7c69de7fb22e83138` |
| `canary/checkpoints/step-000000/README.md` | `44960cb51da5d402bf889b9c052a53ef139e30bcf307440b8f1ed99d2d1e954b` |
| `canary/checkpoints/step-000000/adapter_config.json` | `1594f0ba8295e942d644b7970271a46fee947eb4e9cfe2d6465d9b6777445e30` |
| `canary/checkpoints/step-000000/adapter_model.safetensors` | `2e6e92b1e6f2afc5dced8d21007407efbff4c9a8b21b7198dac5f3e6286333d2` |
| `canary/checkpoints/step-000000/training_state.json` | `cbe3fb86bdb889c0e0ea06b82f24566d4e96fd83af0f058f3113c2b9b71eb35b` |
| `canary/checkpoints/step-000000/validation_metrics.json` | `fd55434ba8e8040caf19b13f644b10a5c6500fd85995dc058bbcc9f8f63b89a3` |
| `canary/checkpoints/step-000010/README.md` | `44960cb51da5d402bf889b9c052a53ef139e30bcf307440b8f1ed99d2d1e954b` |
| `canary/checkpoints/step-000010/adapter_config.json` | `1594f0ba8295e942d644b7970271a46fee947eb4e9cfe2d6465d9b6777445e30` |
| `canary/checkpoints/step-000010/adapter_model.safetensors` | `d71167217aefa8ac600783ca8dd82d5e7be73df871c10b80b5a412010940c80d` |
| `canary/checkpoints/step-000010/training_state.json` | `b20d3f3aa7f5ee5717778980c6ca1b93106229b313ee729e1101399a51dc659b` |
| `canary/checkpoints/step-000010/validation_metrics.json` | `2f8bb7f8c274ff4d5e6218cbbe4fdde860dd0a9828446e873b370fda93f589bc` |
| `canary/provenance.json` | `f4c0a468ab99beb92a06fe32f69434237154a300b2a890832c4b7549ae6e36c1` |
| `canary/run_config.json` | `7b7bfe77cd34a955083fee2b43bf95edad5c3c5a95d7e1659fb19a7c85d1f5a9` |
| `canary/split_manifest.json` | `dbd96fdc8c679927b712acd7f98a22e42a233dfbcd8d573fca9f39bbe271faae` |
| `canary/summary.json` | `44cdcfd38d9f609cc59a5118e1dd1df5454fe8a9e104dd63a3b876e93a23de63` |
| `canary/teacher_label_audit.json` | `18222dff686e587e101213e8503b3f6f412e2df09e3d7a4b2809ae1403878464` |
| `canary/teacher_labels.jsonl` | `a4a4a9b19979ac2c05be236991a981152674523a34b32154dd04744a684cd7f3` |
| `canary/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `canary/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `canary/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `canary/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `canary/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `canary/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `canary/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `canary/training_metrics.jsonl` | `8ab5147811e6de04de4588ab58c76d13189abc1290dfd5453d91ba61cee4dfe0` |
| `canary/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `canary/validation_metrics.jsonl` | `2fac5048d8deea2cb3bdb924435a1afa3509f842fdb7e834397bb63c0b447651` |
| `canary/validation_scores/step-000000.jsonl` | `ad5212fcef71316206965f4d7e47986cae57ec72a357437bea0f99d23a54dad9` |
| `canary/validation_scores/step-000010.jsonl` | `e022f4d254bbc0d4cf78a058accfb4fb94790b6fb5b02d7127105f4167a7d7a3` |
| `id_lock/lock_manifest.json` | `fd63c2b6ccbafcb925114659d05fd246dfb8be1d5fbd2f0a21bba3a4ef1840ab` |
| `m0/compositional.json` | `cb15f78cde58a92aecfb85d9bb822c03bc50645f2e0f782a140f318a96218f30` |
| `m0/compositional.json.metadata` | `97569f66a4d1363a4f4cec62131e047702663d3aa41ce6eea80f39f8a292eec1` |
| `m0/decoupled.json` | `e93ab75d2cfd48b3d8b423ca6684d6c211c3a02919175491eacd384544f7c56d` |
| `m0/decoupled.json.metadata` | `eb1b8d585728278d555f4a9f96da44009d226af3cca0be094f452c871cd97977` |
| `m0/evoked.json` | `37ad6d1ad4b3d39a1c1ac04adfce98470aa9e383f99298b547eed42f0ebd54ef` |
| `m0/evoked.json.metadata` | `0539edf88405356dae908457b2a2db7bfd88ed23abf28056e1db642b76af5921` |
| `m0/evoked_g2.json` | `53c9ab4cf46437bb3030c8e60e552ac6e0b8d2d3a2e615cc5b09dea94b4a188e` |
| `m0/evoked_g2.json.metadata` | `2fc0b42a9a4eb17ad6439696e1ad8753e6ad34889b238841cad2ce2dc04360ca` |
| `m0/explicit.json` | `9711fdb145934a9cd36e7b4fcbea9ae9832cf82fbea0345d0db54448a0cb062c` |
| `m0/explicit.json.metadata` | `6e709dc2daa10eab006f50ec116926143dacfbada0f843b6e68e3d7663796c5d` |
| `m0/gate.json` | `a7764ec4a481e812aa821c36c7a2dde7be689f9f03f9f776645e64d0ea150a44` |
| `m0/gate.md` | `511e03df934a882e35332c303e140597c2e9ef514a62679d61f5ad4b372e9e38` |
| `m1/` | `4d8b16156831bf94f469839ae72438d6f1077e79d644febe873688f014b75c60` |
| `m1/checkpoints/step-000000/README.md` | `44960cb51da5d402bf889b9c052a53ef139e30bcf307440b8f1ed99d2d1e954b` |
| `m1/checkpoints/step-000000/adapter_config.json` | `6b5dd43e702f49c16536d1fb7509101214b75bb0b1ae0a38ce217afe9cc8819b` |
| `m1/checkpoints/step-000000/adapter_model.safetensors` | `2e6e92b1e6f2afc5dced8d21007407efbff4c9a8b21b7198dac5f3e6286333d2` |
| `m1/checkpoints/step-000000/training_state.json` | `d0a80cc929d6612fe19139313efc97b62ba24104f056abd00463c3641920f314` |
| `m1/checkpoints/step-000000/validation_metrics.json` | `fd55434ba8e8040caf19b13f644b10a5c6500fd85995dc058bbcc9f8f63b89a3` |
| `m1/checkpoints/step-000100/README.md` | `44960cb51da5d402bf889b9c052a53ef139e30bcf307440b8f1ed99d2d1e954b` |
| `m1/checkpoints/step-000100/adapter_config.json` | `6b5dd43e702f49c16536d1fb7509101214b75bb0b1ae0a38ce217afe9cc8819b` |
| `m1/checkpoints/step-000100/adapter_model.safetensors` | `11ed535324ec1caf1f478e7c24c9bb91f02da2511bf83fcb1b2001ab3d80aed0` |
| `m1/checkpoints/step-000100/training_state.json` | `3e001644cced42a2e5dda770ef6c9b4279ef1ed2789570cdf469153fe7fec1fb` |
| `m1/checkpoints/step-000100/validation_metrics.json` | `8327480460aaea0092a56ad16670dc0a5eea71efbf99d3638a17c16d7ca991c4` |
| `m1/checkpoints/step-000250/README.md` | `44960cb51da5d402bf889b9c052a53ef139e30bcf307440b8f1ed99d2d1e954b` |
| `m1/checkpoints/step-000250/adapter_config.json` | `6b5dd43e702f49c16536d1fb7509101214b75bb0b1ae0a38ce217afe9cc8819b` |
| `m1/checkpoints/step-000250/adapter_model.safetensors` | `3a2113045205223a0ac7a46c55f18ecb060379b2d4fe240854bc9117ca8ad0ae` |
| `m1/checkpoints/step-000250/training_state.json` | `bf3912c63b5017df7c68c25922c6a2eb4b22dfbc1a062827c8d5ff6a1d31ce9b` |
| `m1/checkpoints/step-000250/validation_metrics.json` | `bfed31ddc3bcf1c7fd130d81380dfd35567ce4515888c47dc752dcc4ae84ddcc` |
| `m1/checkpoints/step-000414/README.md` | `44960cb51da5d402bf889b9c052a53ef139e30bcf307440b8f1ed99d2d1e954b` |
| `m1/checkpoints/step-000414/adapter_config.json` | `6b5dd43e702f49c16536d1fb7509101214b75bb0b1ae0a38ce217afe9cc8819b` |
| `m1/checkpoints/step-000414/adapter_model.safetensors` | `090dfb2cb840b28ca5dc6958c40d893a61d6e20e791552d56bae1fc67a62f347` |
| `m1/checkpoints/step-000414/training_state.json` | `be9876ff3d69e0032d7411bc5194403d58911f922ddaa80aabd004b7abafc6f7` |
| `m1/checkpoints/step-000414/validation_metrics.json` | `0efe246f47c525ba0cdab1e3824cd8b30daf1559b0b04408e4f4b2a7c9877457` |
| `m1/lock_manifest.json` | `0cc1ae4bcca1ccbbd73bdd4239b22d52b985006f80f08016a7db47c7a3e8ca67` |
| `m1/provenance.json` | `f4c0a468ab99beb92a06fe32f69434237154a300b2a890832c4b7549ae6e36c1` |
| `m1/run_config.json` | `3b4c9cda14624bfdfa9bd5adbe70f38ed7c827f5f366f2389311e2f830d167e6` |
| `m1/split_manifest.json` | `dbd96fdc8c679927b712acd7f98a22e42a233dfbcd8d573fca9f39bbe271faae` |
| `m1/summary.json` | `f1217cb864efdc1f28415fef219ab9502fec179ff72dc7b11e28024858ac71ea` |
| `m1/teacher_label_audit.json` | `18222dff686e587e101213e8503b3f6f412e2df09e3d7a4b2809ae1403878464` |
| `m1/teacher_labels.jsonl` | `a4a4a9b19979ac2c05be236991a981152674523a34b32154dd04744a684cd7f3` |
| `m1/tokenizer/added_tokens.json` | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `m1/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `m1/tokenizer/merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `m1/tokenizer/special_tokens_map.json` | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `m1/tokenizer/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `m1/tokenizer/tokenizer_config.json` | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `m1/tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `m1/training_metrics.jsonl` | `bbeb8d596a4e979c69dc53d519f07ed13c64e2c2fe148e979037e9615f72f399` |
| `m1/truncation_stats.json` | `ba74cb8659729796c2fc82bf692546760fc0dcbb34b3daebb650f3f80711b87f` |
| `m1/validation_metrics.jsonl` | `40f9b41925a7a7b7001ff59341bef2d1bb20f2a951a19890143ce31862226086` |
| `m1/validation_scores/step-000000.jsonl` | `ad5212fcef71316206965f4d7e47986cae57ec72a357437bea0f99d23a54dad9` |
| `m1/validation_scores/step-000100.jsonl` | `a881a24b11e4e614684c6e4084b343b6e1088aaff3356ed5cbbf5b8b7df9a640` |
| `m1/validation_scores/step-000250.jsonl` | `68d6f2935c3075dd7dd0ea5ac3b3c116bd3b52b654f261397de6597bd004d889` |
| `m1/validation_scores/step-000414.jsonl` | `f1dc683e0ac9faebddec322ff64b955ebd922cae40d39f4b1ea36738b5173a5e` |
| `ood/result.json` | `b30169b2f20445c3c0f8d046a01d47f00f54d4e309489016c912b7cae7dda880` |

## 18. No H100 job was launched

No H100 job was launched. This invocation used the single verified NVIDIA RTX A100 and stops here for manual review; it launched no M2 or later stage.
