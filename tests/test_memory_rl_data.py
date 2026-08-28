from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.memory_rl.data import (
    ForbiddenTrainingSourceError,
    JoinValidationError,
    TrainingSpec,
    build_training_bundle,
    canonicalize_source,
    default_training_specs,
    load_episode_specs,
    load_source_dataset,
    split_episodes,
    verify_split_manifest,
    write_split_manifest,
    within_episode_percentiles,
)


def _episode(index: int, *, duplicate: bool = False) -> dict:
    items = [
        {"concept": f"bridge{index}", "label": "load_bearing", "role": "useful"},
        {"concept": f"noise{index}", "label": "distractor", "role": "irrelevant"},
    ]
    if duplicate:
        items.append(
            {"concept": f"noise{index}", "label": "distractor", "role": "duplicate"}
        )
    return {
        "context": f"context {index}",
        "probe_question": f"question {index}?",
        "answer": f"answer {index}",
        "items": items,
    }


def _write_source(
    directory: Path,
    source: str,
    *,
    number_of_episodes: int = 10,
    duplicate_first: bool = False,
) -> TrainingSpec:
    battery = [
        _episode(index, duplicate=duplicate_first and index == 0)
        for index in range(number_of_episodes)
    ]
    rows = []
    for episode_index, episode in enumerate(battery):
        episode_rows = []
        for candidate_index, item in enumerate(episode["items"]):
            episode_rows.append(
                {
                    "episode": episode_index,
                    "concept": item["concept"],
                    "label": item["label"],
                    "W_rr": 0.1 if item["label"] == "load_bearing" else 0.5,
                    "marker": f"{episode_index}:{candidate_index}",
                }
            )
        # Verify that joins use keys/occurrences, not global row position.
        rows.extend(reversed(episode_rows))

    battery_path = directory / f"{source}_battery.json"
    results_path = directory / f"{source}_results.json"
    battery_path.write_text(json.dumps(battery), encoding="utf-8")
    results_path.write_text(json.dumps(rows), encoding="utf-8")
    return TrainingSpec(source, battery_path, results_path)


def test_source_allowlist_rejects_every_ood_alias_before_io(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"
    for source in [
        "decoupled",
        "v4f",
        "v4xl",
        "v3d",
        "v4_relabs",
        "v4_locomo_wrapped",
    ]:
        with pytest.raises(ForbiddenTrainingSourceError):
            load_source_dataset(TrainingSpec(source, missing, missing))

    assert canonicalize_source("v1f") == "explicit"
    assert canonicalize_source("Evoked-G2") == "evoked_g2"
    with pytest.raises(ValueError, match="unknown source"):
        canonicalize_source("new_unreviewed_generator")


def test_occurrence_aware_join_ids_and_tied_percentiles(tmp_path: Path) -> None:
    spec = _write_source(
        tmp_path, "explicit", number_of_episodes=2, duplicate_first=True
    )
    dataset = load_source_dataset(spec)
    episode = dataset.episodes[0]

    assert episode.episode_id == "explicit:episode:000000"
    assert episode.uid == "explicit:episode:000000"
    assert episode.source_episode == 0
    assert [candidate.candidate_id for candidate in episode.candidates] == [
        "explicit:episode:000000:candidate:000",
        "explicit:episode:000000:candidate:001",
        "explicit:episode:000000:candidate:002",
    ]
    assert [candidate.result["marker"] for candidate in episode.candidates] == [
        "0:0",
        "0:2",
        "0:1",
    ]
    assert [candidate.w_percentile for candidate in episode.candidates] == [
        0.0,
        0.75,
        0.75,
    ]
    assert [candidate.workspace_target for candidate in episode.candidates] == [
        False,
        True,
        True,
    ]
    assert all(candidate.context == episode.context for candidate in episode.candidates)
    assert not hasattr(episode.candidates[0], "probe_question")
    assert episode.candidates[0].w_ref == episode.candidates[0].w_rr
    assert episode.candidates[0].v_ref is None
    assert dataset.battery_sha256 != dataset.results_sha256
    assert load_source_dataset(spec).episodes == dataset.episodes


def test_join_rejects_missing_extra_or_mislabeled_rows(tmp_path: Path) -> None:
    spec = _write_source(tmp_path, "evoked", number_of_episodes=2)
    rows = json.loads(spec.results_path.read_text(encoding="utf-8"))

    spec.results_path.write_text(json.dumps(rows[:-1]), encoding="utf-8")
    with pytest.raises(JoinValidationError, match="battery items"):
        load_source_dataset(spec)

    rows[-1]["label"] = "filler"
    spec.results_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(JoinValidationError, match="no result row matches"):
        load_source_dataset(spec)

    rows[-1]["label"] = "distractor"
    rows.append(dict(rows[-1], episode=99))
    spec.results_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(JoinValidationError, match="outside battery range"):
        load_source_dataset(spec)


def test_split_is_episode_level_stratified_and_order_invariant(tmp_path: Path) -> None:
    explicit = load_source_dataset(_write_source(tmp_path, "explicit"))
    evoked = load_source_dataset(_write_source(tmp_path, "evoked"))
    episodes = explicit.episodes + evoked.episodes

    train, validation = split_episodes(episodes, seed=7)
    assert len(train) == 16
    assert len(validation) == 4
    assert {episode.episode_id for episode in train}.isdisjoint(
        episode.episode_id for episode in validation
    )
    assert {episode.source for episode in validation} == {"explicit", "evoked"}
    assert split_episodes(reversed(episodes), seed=7) == (train, validation)
    assert split_episodes(episodes, seed=8) != (train, validation)


def test_bundle_manifest_is_self_hashed_and_detects_tampering(tmp_path: Path) -> None:
    specs = (
        _write_source(tmp_path, "explicit"),
        _write_source(tmp_path, "evoked"),
        _write_source(tmp_path, "evoked_g2"),
    )
    bundle = build_training_bundle(specs, seed=11)

    assert len(bundle.train_episodes) == 24
    assert len(bundle.validation_episodes) == 6
    assert verify_split_manifest(bundle.split_manifest)
    assert set(bundle.split_manifest["sources"]) == {
        "explicit",
        "evoked",
        "evoked_g2",
    }

    tampered = copy.deepcopy(bundle.split_manifest)
    tampered["seed"] = 12
    assert not verify_split_manifest(tampered)

    manifest_path = tmp_path / "manifests" / "split.json"
    written = write_split_manifest(manifest_path, bundle.split_manifest)
    assert written == bundle.split_manifest
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == written
    assert len(load_episode_specs(specs)) == 30


def test_percentiles_and_default_training_specs(tmp_path: Path) -> None:
    assert within_episode_percentiles([2.0]) == (0.5,)
    assert within_episode_percentiles([3.0, 1.0, 2.0]) == (1.0, 0.0, 0.5)
    assert within_episode_percentiles([1.0, 2.0, 2.0, 4.0]) == (
        0.0,
        0.5,
        0.5,
        1.0,
    )

    specs = default_training_specs("teacher", tmp_path)
    assert [spec.source for spec in specs] == ["explicit", "evoked", "evoked_g2"]
    assert [spec.battery_path.name for spec in specs] == [
        "battery_v1_final.json",
        "battery_v2_final.json",
        "battery_v2_g2.json",
    ]
    assert [spec.results_path.name for spec in specs] == [
        "results_v1f_teacher.json",
        "results_v2f_teacher.json",
        "results_v2g2_teacher.json",
    ]
