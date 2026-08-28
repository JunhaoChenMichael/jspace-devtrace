"""Validated data loading for workspace-guided memory-admission training.

The original experiment artifacts identify rows only by an integer episode
index and a concept string.  That is not sufficient for RL runs: source files
can be mixed, and one released Explicit episode contains the same concept
twice.  This module therefore assigns source-qualified episode and candidate
IDs, joins measurement rows to battery candidates with occurrence-aware
validation, and creates a deterministic episode-level train/validation split.

Only Explicit, Evoked, and Evoked-G2 are admitted to a training bundle.  The
construct-valid Decoupled and Compositional families (including their derived
foil and LoCoMo variants) are rejected before any files are read.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_TRAIN_SOURCES = frozenset({"explicit", "evoked", "evoked_g2"})

FORBIDDEN_TRAIN_SOURCES = frozenset(
    {
        "compositional",
        "decoupled",
        "decoupled_l",
        "confusable_absent",
        "unrelated_absent",
        "locomo_wrapped_decoupled",
    }
)

_ALLOWED_ALIASES = {
    "explicit": "explicit",
    "v1": "explicit",
    "v1f": "explicit",
    "evoked": "evoked",
    "v2": "evoked",
    "v2f": "evoked",
    "evoked_g2": "evoked_g2",
    "v2_g2": "evoked_g2",
    "v2g2": "evoked_g2",
}

_FORBIDDEN_ALIASES = {
    "compositional": "compositional",
    "v3": "compositional",
    "v3d": "compositional",
    "v3f": "compositional",
    "decoupled": "decoupled",
    "v4": "decoupled",
    "v4f": "decoupled",
    "v4_g2": "decoupled",
    "v4g2": "decoupled",
    "v4_g56": "decoupled",
    "v4g56": "decoupled",
    "decoupled_l": "decoupled_l",
    "v4xl": "decoupled_l",
    "confusable_absent": "confusable_absent",
    "v4_relabs": "confusable_absent",
    "v4ra": "confusable_absent",
    "v4xl_relabs": "confusable_absent",
    "v4xlra": "confusable_absent",
    "unrelated_absent": "unrelated_absent",
    "v4_multiabs": "unrelated_absent",
    "v4ma": "unrelated_absent",
    "locomo_wrapped_decoupled": "locomo_wrapped_decoupled",
    "v4_locomo_wrapped": "locomo_wrapped_decoupled",
    "v4_locomo_wrapped_mid": "locomo_wrapped_decoupled",
    "v4wrap": "locomo_wrapped_decoupled",
    "v4wrapmid": "locomo_wrapped_decoupled",
}


class MemoryRLDataError(ValueError):
    """Base class for data-contract violations."""


class ForbiddenTrainingSourceError(MemoryRLDataError):
    """Raised when an OOD evaluation source is supplied for training."""


class JoinValidationError(MemoryRLDataError):
    """Raised when a results file cannot be joined exactly to its battery."""


class SplitValidationError(MemoryRLDataError):
    """Raised when an episode-level split or its manifest is invalid."""


def _normalized_source_name(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise MemoryRLDataError("source must be a non-empty string")
    return source.strip().lower().replace("-", "_")


def canonicalize_source(source: str, *, for_training: bool = True) -> str:
    """Return a canonical source name and enforce the training allow-list.

    Aliases mirror the repository's historical v1/v2/v3/v4 filenames.  Unknown
    sources are rejected rather than optimistically treated as training data.
    """

    key = _normalized_source_name(source)
    if key in _ALLOWED_ALIASES:
        return _ALLOWED_ALIASES[key]
    if key in _FORBIDDEN_ALIASES:
        canonical = _FORBIDDEN_ALIASES[key]
        if for_training:
            raise ForbiddenTrainingSourceError(
                f"source {source!r} resolves to held-out {canonical!r}; "
                "only Explicit, Evoked, and Evoked-G2 may be used for training"
            )
        return canonical
    raise MemoryRLDataError(
        f"unknown source {source!r}; allowed training sources are "
        f"{sorted(ALLOWED_TRAIN_SOURCES)}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingSpec:
    """A battery and its fixed-reference measurement rows."""

    source: str
    battery_path: Path
    results_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "battery_path", Path(self.battery_path))
        object.__setattr__(self, "results_path", Path(self.results_path))

    @property
    def canonical_source(self) -> str:
        return canonicalize_source(self.source, for_training=True)


# Backward-compatible descriptive alias; new callers should use TrainingSpec.
SourceSpec = TrainingSpec


@dataclass(frozen=True)
class CandidateRecord:
    uid: str
    context: str
    concept: str
    label: str
    w_ref: float
    v_ref: float | None
    w_percentile: float
    workspace_target: bool
    source: str
    source_episode: int
    candidate_index: int
    episode_uid: str
    role: str | None
    fingerprint_sha256: str
    result: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return self.uid

    @property
    def episode_id(self) -> str:
        return self.episode_uid

    @property
    def episode_index(self) -> int:
        return self.source_episode

    @property
    def w_rr(self) -> float:
        return self.w_ref


@dataclass(frozen=True)
class EpisodeRecord:
    uid: str
    source: str
    source_episode: int
    context: str
    probe_question: str
    answer: str
    candidates: tuple[CandidateRecord, ...]
    fingerprint_sha256: str

    @property
    def episode_id(self) -> str:
        return self.uid

    @property
    def episode_index(self) -> int:
        return self.source_episode


@dataclass(frozen=True)
class SourceDataset:
    spec: TrainingSpec
    source: str
    episodes: tuple[EpisodeRecord, ...]
    battery_sha256: str
    results_sha256: str


@dataclass(frozen=True)
class TrainingBundle:
    sources: tuple[SourceDataset, ...]
    train_episodes: tuple[EpisodeRecord, ...]
    validation_episodes: tuple[EpisodeRecord, ...]
    split_manifest: Mapping[str, Any]


def default_training_specs(
    teacher_tag: str = "7B-Instruct",
    repo_root: str | Path | None = None,
) -> tuple[TrainingSpec, ...]:
    """Return the predeclared MVP training sources for a reference model tag."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    benchmarks = root / "data" / "benchmarks"
    results = root / "data" / "results"
    return (
        TrainingSpec(
            "explicit",
            benchmarks / "battery_v1_final.json",
            results / f"results_v1f_{teacher_tag}.json",
        ),
        TrainingSpec(
            "evoked",
            benchmarks / "battery_v2_final.json",
            results / f"results_v2f_{teacher_tag}.json",
        ),
        TrainingSpec(
            "evoked_g2",
            benchmarks / "battery_v2_g2.json",
            results / f"results_v2g2_{teacher_tag}.json",
        ),
    )


def _load_json_list(path: Path, description: str) -> list[Any]:
    if not path.is_file():
        raise MemoryRLDataError(f"{description} file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryRLDataError(f"cannot read {description} file {path}: {exc}") from exc
    if not isinstance(value, list):
        raise MemoryRLDataError(f"{description} file must contain a JSON list: {path}")
    return value


def within_episode_percentiles(values: Sequence[float]) -> tuple[float, ...]:
    """Return ascending [0, 1] percentile ranks with average ranks for ties.

    A singleton receives 0.5 (neutral).  Average tie ranks are important here:
    stable sorting would otherwise leak the label-correlated candidate order
    when reciprocal-rank workspace scores are equal.
    """

    if not values:
        return ()
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise JoinValidationError("W_rr values must all be finite")
    if len(numeric) == 1:
        return (0.5,)

    ordered = sorted(range(len(numeric)), key=numeric.__getitem__)
    ranks = [0.0] * len(numeric)
    start = 0
    while start < len(ordered):
        end = start + 1
        value = numeric[ordered[start]]
        while end < len(ordered) and numeric[ordered[end]] == value:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank / (len(numeric) - 1)
        start = end
    return tuple(ranks)


def _require_string(obj: Mapping[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise JoinValidationError(f"{where}.{key} must be a non-empty string")
    return value


def _validate_result_rows(
    rows: Sequence[Any], number_of_episodes: int, results_path: Path
) -> dict[int, list[dict[str, Any]]]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_index, raw_row in enumerate(rows):
        where = f"{results_path}[{row_index}]"
        if not isinstance(raw_row, dict):
            raise JoinValidationError(f"{where} must be an object")
        row = dict(raw_row)
        episode = row.get("episode")
        if isinstance(episode, bool) or not isinstance(episode, int):
            raise JoinValidationError(f"{where}.episode must be an integer")
        if episode < 0 or episode >= number_of_episodes:
            raise JoinValidationError(
                f"{where}.episode={episode} is outside battery range "
                f"[0, {number_of_episodes})"
            )
        _require_string(row, "concept", where)
        _require_string(row, "label", where)
        try:
            w_rr = float(row["W_rr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JoinValidationError(f"{where}.W_rr must be numeric") from exc
        if not math.isfinite(w_rr) or w_rr < 0:
            raise JoinValidationError(f"{where}.W_rr must be finite and non-negative")
        row["W_rr"] = w_rr
        if "V" in row and row["V"] is not None:
            try:
                v_ref = float(row["V"])
            except (TypeError, ValueError) as exc:
                raise JoinValidationError(f"{where}.V must be numeric or null") from exc
            if not math.isfinite(v_ref) or not 0.0 <= v_ref <= 1.0:
                raise JoinValidationError(f"{where}.V must be finite and in [0, 1]")
            row["V"] = v_ref
        by_episode[episode].append(row)
    return dict(by_episode)


def load_source_dataset(spec: TrainingSpec, top_k: int = 2) -> SourceDataset:
    """Load one allowed source and perform an exact battery/results join.

    Rows are matched by ``(concept, label, occurrence)`` within each episode,
    not by concept alone.  This preserves the known duplicate ``baking`` item
    in Explicit while still rejecting missing, extra, or mislabeled rows.
    """

    source = spec.canonical_source  # Reject OOD before touching either path.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise MemoryRLDataError("top_k must be a positive integer")
    battery = _load_json_list(spec.battery_path, "battery")
    rows = _load_json_list(spec.results_path, "results")
    by_episode = _validate_result_rows(rows, len(battery), spec.results_path)

    expected_episode_indices = set(range(len(battery)))
    actual_episode_indices = set(by_episode)
    if actual_episode_indices != expected_episode_indices:
        missing = sorted(expected_episode_indices - actual_episode_indices)
        extra = sorted(actual_episode_indices - expected_episode_indices)
        raise JoinValidationError(
            f"results/battery episode mismatch for {source}: "
            f"missing={missing}, extra={extra}"
        )

    episodes: list[EpisodeRecord] = []
    for episode_index, raw_episode in enumerate(battery):
        where = f"{spec.battery_path}[{episode_index}]"
        if not isinstance(raw_episode, dict):
            raise JoinValidationError(f"{where} must be an object")
        context = _require_string(raw_episode, "context", where)
        probe_question = _require_string(raw_episode, "probe_question", where)
        answer = _require_string(raw_episode, "answer", where)
        raw_items = raw_episode.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise JoinValidationError(f"{where}.items must be a non-empty list")

        result_rows = by_episode[episode_index]
        if len(result_rows) != len(raw_items):
            raise JoinValidationError(
                f"episode {episode_index} of {source} has {len(raw_items)} battery "
                f"items but {len(result_rows)} result rows"
            )

        queues: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for row in result_rows:
            queues[(row["concept"], row["label"])].append(row)

        joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for candidate_index, raw_item in enumerate(raw_items):
            item_where = f"{where}.items[{candidate_index}]"
            if not isinstance(raw_item, dict):
                raise JoinValidationError(f"{item_where} must be an object")
            concept = _require_string(raw_item, "concept", item_where)
            label = _require_string(raw_item, "label", item_where)
            key = (concept, label)
            if not queues[key]:
                raise JoinValidationError(
                    f"no result row matches {item_where} concept={concept!r}, "
                    f"label={label!r}"
                )
            joined.append((dict(raw_item), queues[key].popleft()))

        leftovers = [
            {"concept": concept, "label": label, "count": len(queue)}
            for (concept, label), queue in queues.items()
            if queue
        ]
        if leftovers:
            raise JoinValidationError(
                f"unmatched result rows in {source} episode {episode_index}: {leftovers}"
            )

        episode_id = f"{source}:episode:{episode_index:06d}"
        episode_fingerprint = _object_sha256(raw_episode)
        percentiles = within_episode_percentiles([row["W_rr"] for _, row in joined])
        ranked_indices = sorted(
            range(len(joined)),
            key=lambda index: (
                -joined[index][1]["W_rr"],
                _object_sha256(
                    {
                        "item": joined[index][0],
                        "candidate_index": index,
                    }
                ),
            ),
        )
        workspace_targets = set(ranked_indices[: min(top_k, len(ranked_indices))])
        candidates: list[CandidateRecord] = []
        for candidate_index, ((item, row), percentile) in enumerate(
            zip(joined, percentiles)
        ):
            candidate_id = f"{episode_id}:candidate:{candidate_index:03d}"
            candidate_fingerprint = _object_sha256(
                {
                    "episode_fingerprint": episode_fingerprint,
                    "candidate_index": candidate_index,
                    "item": item,
                }
            )
            role = item.get("role")
            if role is not None and not isinstance(role, str):
                raise JoinValidationError(
                    f"{where}.items[{candidate_index}].role must be a string or null"
                )
            candidates.append(
                CandidateRecord(
                    uid=candidate_id,
                    context=context,
                    concept=item["concept"],
                    label=item["label"],
                    w_ref=row["W_rr"],
                    v_ref=row.get("V"),
                    w_percentile=percentile,
                    workspace_target=candidate_index in workspace_targets,
                    source=source,
                    source_episode=episode_index,
                    candidate_index=candidate_index,
                    episode_uid=episode_id,
                    role=role,
                    fingerprint_sha256=candidate_fingerprint,
                    result=row,
                )
            )

        episodes.append(
            EpisodeRecord(
                uid=episode_id,
                source=source,
                source_episode=episode_index,
                context=context,
                probe_question=probe_question,
                answer=answer,
                candidates=tuple(candidates),
                fingerprint_sha256=episode_fingerprint,
            )
        )

    return SourceDataset(
        spec=spec,
        source=source,
        episodes=tuple(episodes),
        battery_sha256=file_sha256(spec.battery_path),
        results_sha256=file_sha256(spec.results_path),
    )


def load_episode_specs(
    specs: Sequence[TrainingSpec], top_k: int = 2
) -> tuple[EpisodeRecord, ...]:
    """Load and concatenate allowed training specs with unique source names."""

    selected_specs = tuple(specs)
    if not selected_specs:
        raise MemoryRLDataError("at least one training source is required")
    canonical_names = [spec.canonical_source for spec in selected_specs]
    if len(set(canonical_names)) != len(canonical_names):
        raise MemoryRLDataError(
            f"training source specified more than once: {canonical_names}"
        )
    return tuple(
        episode
        for spec in selected_specs
        for episode in load_source_dataset(spec, top_k=top_k).episodes
    )


def _split_hash(seed: int, episode_id: str) -> str:
    return sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()


def split_episodes(
    episodes: Iterable[EpisodeRecord],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]]:
    """Create an exact, source-stratified, episode-level SHA-256 split."""

    if not 0.0 < val_fraction < 1.0:
        raise SplitValidationError("val_fraction must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitValidationError("seed must be an integer")

    by_source: dict[str, list[EpisodeRecord]] = defaultdict(list)
    seen_ids: set[str] = set()
    for episode in episodes:
        if episode.source not in ALLOWED_TRAIN_SOURCES:
            raise ForbiddenTrainingSourceError(
                f"episode {episode.uid!r} comes from non-training source "
                f"{episode.source!r}"
            )
        if episode.uid in seen_ids:
            raise SplitValidationError(f"duplicate episode uid: {episode.uid}")
        seen_ids.add(episode.uid)
        by_source[episode.source].append(episode)

    if not by_source:
        raise SplitValidationError("cannot split an empty episode collection")

    train: list[EpisodeRecord] = []
    validation: list[EpisodeRecord] = []
    for source in sorted(by_source):
        group = by_source[source]
        if len(group) < 2:
            raise SplitValidationError(
                f"source {source!r} needs at least two episodes for an 80/20 split"
            )
        number_validation = max(1, math.ceil(len(group) * val_fraction))
        if number_validation >= len(group):
            number_validation = len(group) - 1
        hashed = sorted(group, key=lambda episode: _split_hash(seed, episode.uid))
        validation_ids = {episode.uid for episode in hashed[:number_validation]}
        validation.extend(
            episode for episode in group if episode.uid in validation_ids
        )
        train.extend(episode for episode in group if episode.uid not in validation_ids)

    sort_key = lambda episode: (episode.source, episode.source_episode)
    train.sort(key=sort_key)
    validation.sort(key=sort_key)
    if {episode.uid for episode in train} & {episode.uid for episode in validation}:
        raise SplitValidationError("train and validation episode IDs overlap")
    return tuple(train), tuple(validation)


def _build_split_manifest(
    datasets: Sequence[SourceDataset],
    train: Sequence[EpisodeRecord],
    validation: Sequence[EpisodeRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    train_ids = {episode.episode_id for episode in train}
    validation_ids = {episode.episode_id for episode in validation}
    source_entries: dict[str, Any] = {}
    for dataset in sorted(datasets, key=lambda value: value.source):
        source_entries[dataset.source] = {
            "battery_path": str(dataset.spec.battery_path.resolve()),
            "battery_sha256": dataset.battery_sha256,
            "results_path": str(dataset.spec.results_path.resolve()),
            "results_sha256": dataset.results_sha256,
            "train_episode_ids": [
                episode.episode_id
                for episode in dataset.episodes
                if episode.episode_id in train_ids
            ],
            "validation_episode_ids": [
                episode.episode_id
                for episode in dataset.episodes
                if episode.episode_id in validation_ids
            ],
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "per-source sha256(seed:episode_id), exact ceil holdout",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "sources": source_entries,
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
    }
    payload["manifest_sha256"] = _object_sha256(payload)
    return payload


def verify_split_manifest(manifest: Mapping[str, Any]) -> bool:
    """Verify the self-hash of a split manifest."""

    if not isinstance(manifest, Mapping):
        return False
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        return False
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _object_sha256(payload) == expected


def write_split_manifest(
    path: str | Path,
    train_or_manifest: Sequence[EpisodeRecord] | Mapping[str, Any],
    validation_episodes: Sequence[EpisodeRecord] | None = None,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a canonical, self-hashed split manifest.

    ``train_or_manifest`` may be the manifest already attached to a
    :class:`TrainingBundle`, or a train episode sequence accompanied by
    ``validation_episodes``.  The latter form is useful for small synthetic or
    externally assembled datasets that do not have SourceDataset metadata.
    """

    if isinstance(train_or_manifest, Mapping):
        if validation_episodes is not None:
            raise SplitValidationError(
                "validation_episodes must be omitted when writing an existing manifest"
            )
        manifest = dict(train_or_manifest)
        if not verify_split_manifest(manifest):
            raise SplitValidationError("refusing to write a manifest with an invalid SHA-256")
    else:
        if validation_episodes is None:
            raise SplitValidationError("validation_episodes are required")
        train = tuple(train_or_manifest)
        validation = tuple(validation_episodes)
        train_ids = {episode.uid for episode in train}
        validation_ids = {episode.uid for episode in validation}
        if len(train_ids) != len(train) or len(validation_ids) != len(validation):
            raise SplitValidationError("split contains duplicate episode UIDs")
        if train_ids & validation_ids:
            raise SplitValidationError("train and validation episode UIDs overlap")
        sources: dict[str, Any] = {}
        for source in sorted({episode.source for episode in train + validation}):
            sources[source] = {
                "train_episode_ids": sorted(
                    episode.uid for episode in train if episode.source == source
                ),
                "validation_episode_ids": sorted(
                    episode.uid for episode in validation if episode.source == source
                ),
            }
        manifest = {
            "schema_version": 1,
            "algorithm": "per-source sha256(seed:episode_id), exact ceil holdout",
            "seed": seed,
            "validation_fraction": val_fraction,
            "sources": sources,
            "train_episode_count": len(train),
            "validation_episode_count": len(validation),
        }
        manifest["manifest_sha256"] = _object_sha256(manifest)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return manifest


def build_training_bundle(
    specs: Sequence[TrainingSpec] | None = None,
    *,
    repo_root: str | Path | None = None,
    teacher_tag: str = "7B-Instruct",
    val_fraction: float = 0.2,
    seed: int = 0,
    top_k: int = 2,
) -> TrainingBundle:
    """Load allowed sources, split by episode, and seal the split manifest."""

    selected_specs = tuple(
        specs
        if specs is not None
        else default_training_specs(teacher_tag, repo_root)
    )
    if not selected_specs:
        raise MemoryRLDataError("at least one training source is required")

    canonical_names = [spec.canonical_source for spec in selected_specs]
    if len(set(canonical_names)) != len(canonical_names):
        raise MemoryRLDataError(
            f"training source specified more than once: {canonical_names}"
        )

    datasets = tuple(load_source_dataset(spec, top_k=top_k) for spec in selected_specs)
    all_episodes = tuple(
        episode for dataset in datasets for episode in dataset.episodes
    )
    train, validation = split_episodes(all_episodes, val_fraction, seed)
    manifest = _build_split_manifest(
        datasets,
        train,
        validation,
        validation_fraction=val_fraction,
        seed=seed,
    )
    return TrainingBundle(
        sources=datasets,
        train_episodes=train,
        validation_episodes=validation,
        split_manifest=manifest,
    )
