from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments import train_metacog_m1 as m1  # noqa: E402
from memory_rl.data import ForbiddenTrainingSourceError, TrainingSpec  # noqa: E402


class WordTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "yes":
            return {"input_ids": [901]}
        if text == "no":
            return {"input_ids": [902]}
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def _candidate(
    episode_id: str,
    index: int,
    concept: str,
    score: float,
    fingerprint: str,
    label: str = "distractor",
):
    return SimpleNamespace(
        uid=f"{episode_id}:candidate:{index:03d}",
        candidate_index=index,
        concept=concept,
        w_ref=score,
        fingerprint_sha256=fingerprint,
        label=label,
    )


def _episode(source="explicit"):
    episode_id = f"{source}:episode:000000"
    return SimpleNamespace(
        uid=episode_id,
        source=source,
        context="one two three four five six seven eight nine ten",
        candidates=(
            _candidate(episode_id, 0, "baking", 0.9, "b", "load_bearing"),
            _candidate(episode_id, 1, "baking", 0.5, "a", "load_bearing"),
            _candidate(episode_id, 2, "noise", 0.5, "c"),
            _candidate(episode_id, 3, "other", 0.1, "d"),
        ),
    )


def test_teacher_labels_use_candidate_ids_and_deterministic_top_two():
    examples = m1.build_supervised_examples([_episode()])

    assert len(examples) == 4
    assert len({example.candidate_id for example in examples}) == 4
    assert [example.concept for example in examples[:2]] == ["baking", "baking"]
    assert [example.target for example in examples] == ["yes", "yes", "no", "no"]
    # Equal W_rr at the top-2 boundary preserves original candidate order.
    assert examples[0].teacher_rank == 1
    assert examples[1].teacher_rank == 2
    assert examples[2].teacher_rank == 3
    audit = m1.teacher_tie_audit([_episode()])
    assert audit["episodes_with_top_k_boundary_tie"] == 1
    assert audit["episode_audit"][0]["tie_break"] == (
        "candidate_index_ascending_original_order"
    )


def test_teacher_label_construction_rejects_non_id_source():
    with pytest.raises(m1.M1ProtocolError, match="non-ID source"):
        m1.build_supervised_examples([_episode("decoupled")])


def test_specs_require_exact_id_source_set_and_reject_ood_before_io():
    specs = tuple(
        TrainingSpec(source, Path(f"{source}.battery"), Path(f"{source}.results"))
        for source in ("explicit", "evoked", "evoked_g2")
    )
    assert m1.validate_training_specs(specs) == specs

    with pytest.raises(m1.M1ProtocolError, match="missing"):
        m1.validate_training_specs(specs[:2])
    with pytest.raises(ForbiddenTrainingSourceError):
        m1.validate_training_specs(
            specs[:2]
            + (TrainingSpec("decoupled", Path("missing-a"), Path("missing-b")),)
        )


def test_length_ladder_and_left_truncating_supervised_encoding():
    tokenizer = WordTokenizer()
    example = m1.build_supervised_examples([_episode()])[0]
    prompt_tokens, target_tokens = m1.token_lengths(tokenizer, example)

    assert m1.choose_effective_max_length(1024, 1024) == 1024
    assert m1.choose_effective_max_length(1024, 1025) == 1536
    assert m1.choose_effective_max_length(1024, 1537) == 2048
    with pytest.raises(m1.M1ProtocolError, match="exceeding"):
        m1.choose_effective_max_length(1024, 2049)

    max_length = target_tokens + 4
    input_ids, labels, attention, audit = m1.encode_supervised_example(
        tokenizer, example, max_length
    )
    assert len(input_ids) == max_length
    assert len(labels) == max_length
    assert attention.tolist() == [1] * max_length
    assert labels[:-target_tokens].tolist() == [-100] * 4
    assert labels[-target_tokens:].tolist() == input_ids[-target_tokens:].tolist()
    assert input_ids[-target_tokens:].tolist() == [901]
    assert audit["truncated"] is (prompt_tokens > 4)


def test_truncation_stats_record_requested_and_effective_behavior():
    examples = m1.build_supervised_examples([_episode()])
    stats = m1.truncation_statistics(
        WordTokenizer(),
        examples,
        requested_max_length=8,
        effective_max_length=128,
    )
    assert stats["auto_increased"] is True
    assert stats["would_truncate_at_requested_count"] == len(examples)
    assert stats["actual_truncated_count"] == 0


def test_auc_validation_summary_and_earliest_tie_selection():
    assert m1.binary_roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.5, 0.5]) == 0.875
    assert m1.binary_roc_auc([1, 1], [0.1, 0.2]) is None
    rows = [
        {
            "episode_id": "e0",
            "source": "explicit",
            "utility_label": 1,
            "teacher_target": "yes",
            "yes_probability": 0.9,
        },
        {
            "episode_id": "e0",
            "source": "explicit",
            "utility_label": 0,
            "teacher_target": "no",
            "yes_probability": 0.1,
        },
    ]
    summary = m1.summarize_validation_scores(rows)
    assert summary["verbal_auc"] == 1.0
    assert summary["teacher_alignment_auc"] == 1.0
    assert summary["yes_rate"] == 0.5

    selected = m1.select_best_checkpoint(
        [
            {"step": 250, "verbal_auc": 0.8},
            {"step": 100, "verbal_auc": 0.8},
            {"step": 500, "verbal_auc": 0.79},
        ]
    )
    assert selected["step"] == 100


def test_checkpoint_schedule_has_preregistered_steps_and_actual_terminal():
    assert m1.checkpoint_schedule(500) == (0, 100, 250, 500)
    assert m1.checkpoint_schedule(800) == (0, 100, 250, 500, 800)
    assert m1.checkpoint_schedule(417) == (0, 100, 250, 417)
    assert m1.checkpoint_schedule(7) == (0, 7)
    assert m1.planned_optimizer_steps(
        833, epochs=2, gradient_accumulation=4, max_steps=500
    ) == 418
    assert m1.planned_optimizer_steps(
        833, epochs=2, gradient_accumulation=4, max_steps=500, canary_steps=7
    ) == 7


def test_checkpoint_tree_hash_contract_and_symlink_rejection(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "z.txt").write_text("z", encoding="utf-8")
    (checkpoint / "nested" / "a.txt").write_text("a", encoding="utf-8")

    expected = sha256()
    for relative in ("nested/a.txt", "z.txt"):
        path = checkpoint / relative
        expected.update(relative.encode())
        expected.update(b"\0")
        expected.update(sha256(path.read_bytes()).hexdigest().encode("ascii"))
        expected.update(b"\n")
    assert m1.checkpoint_tree_sha256(checkpoint) == expected.hexdigest()

    (checkpoint / "link").symlink_to(checkpoint / "z.txt")
    with pytest.raises(m1.M1ProtocolError, match="symlink"):
        m1.checkpoint_tree_sha256(checkpoint)


def test_lock_manifest_is_id_only_recomputable_selection_table(tmp_path):
    (tmp_path / "validation_metrics.jsonl").write_text("{}\n")
    records = []
    for step, auc in ((0, 0.8), (100, 0.8), (250, 0.7)):
        relative = Path("checkpoints") / f"step-{step:06d}"
        checkpoint = tmp_path / relative
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.bin").write_bytes(str(step).encode())
        records.append(
            {
                "step": step,
                "verbal_auc": auc,
                "verbal_within_episode_auc": auc - 0.01,
                "yes_rate": 0.5,
                "checkpoint_path": relative.as_posix(),
                "checkpoint_tree_sha256": m1.checkpoint_tree_sha256(checkpoint),
            }
        )
    lock = m1.build_lock_manifest(
        run_dir=tmp_path,
        checkpoint_records=records,
        split_manifest_sha256="1" * 64,
        run_config_sha256="2" * 64,
        provenance_sha256="3" * 64,
    )

    assert lock["selection_scope"] == "id_validation"
    assert lock["selection_metric"] == "verbal_auc"
    assert lock["tie_break"] == "earliest_step"
    assert lock["step"] == 0
    assert lock["candidate_steps"] == [0, 100, 250]
    assert all(
        row["selection_scope"] == "id_validation"
        for row in lock["id_selection_table"]
    )
    assert lock["ood_evaluated"] is False


def test_parser_campaign_defaults():
    parser = m1.create_parser()
    revision = "a" * 40
    args = parser.parse_args(
        ["--out-dir", "out", "--model-revision", revision]
    )
    m1.validate_args(parser, args)

    assert args.model == m1.PRIMARY_MODEL
    assert args.seed == 0
    assert args.lora_rank == 16
    assert args.gradient_accumulation == 4
    assert args.learning_rate == 1e-5
    assert args.epochs == 2
    assert args.max_steps == 500
    assert args.max_sequence_length == 1024
    assert args.canary_steps == 0


def test_explicit_commit_is_effective_when_transformers_reports_none():
    revision = "b" * 40
    assert m1.effective_resolved_revision(revision, None, None) == revision
    m1._verify_pinned_revision("tokenizer", revision, None)
    assert (
        m1._snapshot_commit_from_path(
            f"/cache/models--Qwen--Qwen3-8B/snapshots/{revision}/tokenizer.json"
        )
        == revision
    )


def test_verbal_readout_uses_exact_historical_measure_variants():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            mapping = {value: 100 + index for index, value in enumerate(m1.NO_VARIANTS)}
            mapping.update(
                {value: 200 + index for index, value in enumerate(m1.YES_VARIANTS)}
            )
            return [mapping[text]]

    no_ids, yes_ids = m1.verbal_action_token_ids(Tokenizer())
    assert m1.NO_VARIANTS == ("no", " no", "No", " No", "NO")
    assert m1.YES_VARIANTS == ("yes", " yes", "Yes", " Yes", "YES")
    assert no_ids == list(range(100, 105))
    assert yes_ids == list(range(200, 205))


class TinyWorkspaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1 if "baking" in text.lower() else 2]

    def __call__(self, text, **kwargs):
        del text, kwargs
        return {
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


class TinyWorkspaceBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
            )

    def get_output_embeddings(self):
        return self.head


class TinyWorkspaceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyWorkspaceBase()

    def get_base_model(self):
        return self.base

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
        del input_ids, attention_mask, output_hidden_states, use_cache
        hidden = torch.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]]])
        return SimpleNamespace(hidden_states=(hidden, hidden * 1.1, hidden * 1.2))


def test_canary_workspace_readout_is_real_finite_id_measurement(tmp_path):
    examples = m1.build_supervised_examples([_episode()])
    model = TinyWorkspaceModel()
    model.train()
    output = tmp_path / "workspace.jsonl"
    summary = m1.evaluate_canary_workspace_id(
        model,
        TinyWorkspaceTokenizer(),
        examples,
        device="cpu",
        max_length=16,
        output_path=output,
    )

    assert summary["performed"] is True
    assert summary["all_finite"] is True
    assert summary["candidate_rows"] == len(examples)
    assert summary["used_for_checkpoint_selection"] is False
    assert len(output.read_text().splitlines()) == len(examples)
    assert model.training is True


def _write_source_fixture(
    root: Path, source: str, revision: str, chat_hash: str
) -> tuple[Path, Path]:
    battery = []
    results = []
    for episode_index in range(2):
        concepts = ["baking", "baking", "noise"] if source == "explicit" else [
            "high",
            "middle",
            "low",
        ]
        labels = ["load_bearing", "load_bearing", "distractor"]
        battery.append(
            {
                "context": f"context {source} {episode_index}",
                "probe_question": "question?",
                "answer": "answer",
                "items": [
                    {"concept": concept, "label": label, "role": "fixture"}
                    for concept, label in zip(concepts, labels)
                ],
            }
        )
        for candidate_index, (concept, label) in enumerate(zip(concepts, labels)):
            results.append(
                {
                    "episode": episode_index,
                    "concept": concept,
                    "label": label,
                    "W_rr": 1.0 - candidate_index * 0.2,
                    "V": 0.5,
                }
            )
    battery_path = root / f"{source}-battery.json"
    results_path = root / f"{source}-results.json"
    battery_path.write_text(json.dumps(battery), encoding="utf-8")
    results_path.write_text(json.dumps(results), encoding="utf-8")
    metadata = {
        "schema_version": "workspace_measurement_metadata.v3",
        "model": m1.PRIMARY_MODEL,
        "adapter": None,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "chat_template_sha256": chat_hash,
        "hashes": {
            "raw_output_sha256": sha256(results_path.read_bytes()).hexdigest(),
            "battery_file_sha256": sha256(battery_path.read_bytes()).hexdigest(),
        },
    }
    Path(f"{results_path}.metadata").write_text(json.dumps(metadata), encoding="utf-8")
    return results_path, battery_path


def test_cpu_dry_run_persists_split_labels_and_provenance_without_gpu(tmp_path):
    revision = "c" * 40
    chat_hash = "d" * 64
    arguments = [
        "--out-dir",
        str(tmp_path / "run"),
        "--model-revision",
        revision,
        "--dry-run",
    ]
    for source in ("explicit", "evoked", "evoked_g2"):
        results, battery = _write_source_fixture(tmp_path, source, revision, chat_hash)
        arguments.extend(
            ["--train-spec", f"{source}={results}::{battery}"]
        )

    assert m1.main(arguments) == 0
    run = tmp_path / "run"
    split = json.loads((run / "split_manifest.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    labels = [json.loads(line) for line in (run / "teacher_labels.jsonl").read_text().splitlines()]

    assert split["train_episode_count"] == 3
    assert split["validation_episode_count"] == 3
    assert set(split["sources"]) == {"explicit", "evoked", "evoked_g2"}
    assert len(labels) == 18
    assert len({row["candidate_id"] for row in labels}) == 18
    assert provenance["teacher"]["frozen_original"] is True
    assert provenance["teacher"]["metadata_sidecars_verified"] is True
    assert provenance["teacher"]["student_workspace_used_for_labels"] is False
    assert provenance["data_isolation"]["ood_loaded"] is False
    assert summary["status"] == "DRY_RUN_COMPLETE"
    assert summary["gpu_used"] is False
    assert not (run / "lock_manifest.json").exists()


def test_teacher_sidecars_fail_closed_on_revision_and_chat_template(tmp_path):
    revision = "e" * 40
    specs = []
    for index, source in enumerate(("explicit", "evoked", "evoked_g2")):
        chat_hash = ("f" if index < 2 else "0") * 64
        results, battery = _write_source_fixture(
            tmp_path, source, revision, chat_hash
        )
        specs.append(TrainingSpec(source, battery, results))
    with pytest.raises(m1.M1ProtocolError, match="chat-template"):
        m1.validate_teacher_artifacts(
            specs, model_revision=revision, tokenizer_revision=revision
        )

    metadata_path = Path(f"{specs[0].results_path}.metadata")
    metadata = json.loads(metadata_path.read_text())
    metadata["model_revision"] = "1" * 40
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(m1.M1ProtocolError, match="model revision mismatch"):
        m1.validate_teacher_artifacts(
            specs, model_revision=revision, tokenizer_revision=revision
        )
