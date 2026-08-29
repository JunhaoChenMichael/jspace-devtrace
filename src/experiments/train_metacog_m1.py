"""A5000 entry point for the Qwen3-8B M1 Metacognitive Alignment pilot.

This is deliberately separate from ``finetune_metacog.py``.  The historical
script remains useful for reproducing the small-model experiments, while this
entry point enforces the capable-scale campaign contract:

* only Explicit, Evoked, and Evoked-G2 may enter the data bundle;
* labels are reconstructed once from precomputed frozen-base ``W_rr`` scores
  (top two candidates in each episode are ``yes``);
* a source-stratified episode split is fingerprinted and persisted;
* the Qwen3-8B base is frozen and only a bf16 LoRA adapter is optimized;
* checkpoints are evaluated only on fixed ID validation data and the highest
  verbal AUC is locked, with the earliest step winning exact ties;
* this program has no OOD loading or evaluation path.

The default 500-step value is a cap.  Two literal post-split training epochs
normally end a little before it; that actual terminal step is checkpointed and
included in the ID-only selection table.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_metacog_alignment_campaign import EXPECTED_GPU  # noqa: E402

from memory_rl.data import (  # noqa: E402
    ALLOWED_TRAIN_SOURCES,
    TrainingSpec,
    build_training_bundle,
    default_training_specs,
    file_sha256,
    write_split_manifest,
)
from memory_rl.modeling import (  # noqa: E402
    adapter_disabled,
    binary_action_logits,
    render_admission_prompt,
)


PRIMARY_MODEL = "Qwen/Qwen3-8B"
TOP_K = 2
PER_DEVICE_BATCH_SIZE = 1
DTYPE_NAME = "bfloat16"
CHECKPOINT_STEPS = (0, 100, 250, 500)
LENGTH_LADDER = (1024, 1536, 2048)
# Optimisation seeds authorised by docs/H100_NEXT_CAMPAIGNS.md for the
# three-seed Binary Metacognitive Alignment replication; the split seed stays 0.
ALLOWED_SEEDS = (0, 1, 2)
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
YES_VARIANTS = ("yes", " yes", "Yes", " Yes", "YES")
NO_VARIANTS = ("no", " no", "No", " No", "NO")


class M1ProtocolError(ValueError):
    """Raised when a run would violate the preregistered M1 protocol."""


@dataclass(frozen=True)
class SupervisedExample:
    episode_id: str
    candidate_id: str
    source: str
    context: str
    concept: str
    target: str
    w_ref: float
    teacher_rank: int
    utility_label: int
    candidate_fingerprint_sha256: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    raise TypeError(type(value).__name__)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
            + "\n"
        )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=_json_default,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def checkpoint_tree_sha256(root: str | Path) -> str:
    """Hash a checkpoint tree using the campaign launcher's public contract.

    For each recursively discovered ordinary file, ordered by relative POSIX
    path, update the digest with ``path + NUL + file_sha256 + newline``.  A
    symlink is rejected even when it points to a regular file.
    """

    directory = Path(root)
    if not directory.is_dir() or directory.is_symlink():
        raise M1ProtocolError(f"checkpoint root is not an ordinary directory: {directory}")
    files: list[Path] = []
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            raise M1ProtocolError(f"checkpoint tree contains symlink: {candidate}")
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise M1ProtocolError(f"checkpoint tree is empty: {directory}")
    digest = sha256()
    for path in sorted(files, key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_training_spec(text: str) -> TrainingSpec:
    """Parse ``SOURCE=RESULTS_JSON::BATTERY_JSON`` without colon ambiguity."""

    if "=" not in text or "::" not in text:
        raise argparse.ArgumentTypeError(
            "train spec must be SOURCE=RESULTS_JSON::BATTERY_JSON"
        )
    source, paths = text.split("=", 1)
    results_path, battery_path = paths.split("::", 1)
    if not source or not results_path or not battery_path:
        raise argparse.ArgumentTypeError(
            "train spec must contain a source, results path, and battery path"
        )
    return TrainingSpec(
        source=source,
        results_path=results_path,
        battery_path=battery_path,
    )


def validate_training_specs(specs: Sequence[TrainingSpec]) -> tuple[TrainingSpec, ...]:
    """Require exactly the three preregistered ID sources.

    Accessing ``canonical_source`` also rejects all named OOD aliases before a
    result or battery file is opened.
    """

    selected = tuple(specs)
    canonical = [spec.canonical_source for spec in selected]
    duplicates = sorted({name for name in canonical if canonical.count(name) > 1})
    if duplicates:
        raise M1ProtocolError(f"training sources repeated: {duplicates}")
    if set(canonical) != set(ALLOWED_TRAIN_SOURCES):
        missing = sorted(set(ALLOWED_TRAIN_SOURCES) - set(canonical))
        extra = sorted(set(canonical) - set(ALLOWED_TRAIN_SOURCES))
        raise M1ProtocolError(
            "M1 requires exactly Explicit, Evoked, and Evoked-G2; "
            f"missing={missing}, extra={extra}"
        )
    return selected


def _metadata_revision(metadata: Mapping[str, Any], kind: str) -> str | None:
    if kind == "model":
        candidates = (
            metadata.get("model_revision_resolved"),
            metadata.get("model_revision"),
            metadata.get("model_revision_requested"),
        )
    elif kind == "tokenizer":
        candidates = (
            metadata.get("tokenizer_revision_resolved"),
            metadata.get("tokenizer_revision"),
            metadata.get("tokenizer_revision_effective"),
            metadata.get("tokenizer_revision_requested"),
        )
    else:
        raise ValueError(kind)
    return next((value for value in candidates if isinstance(value, str) and value), None)


def validate_teacher_artifacts(
    specs: Sequence[TrainingSpec],
    *,
    model_revision: str,
    tokenizer_revision: str,
) -> list[dict[str, Any]]:
    """Authenticate all precomputed W_rr files against measurement sidecars."""

    audited: list[dict[str, Any]] = []
    chat_hashes: set[str] = set()
    for spec in specs:
        results_path = spec.results_path.resolve()
        battery_path = spec.battery_path.resolve()
        metadata_path = Path(f"{results_path}.metadata")
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise M1ProtocolError(
                f"frozen-teacher metadata sidecar is required: {metadata_path}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise M1ProtocolError(f"invalid teacher metadata {metadata_path}: {exc}") from exc
        if not isinstance(metadata, Mapping):
            raise M1ProtocolError(f"teacher metadata must be an object: {metadata_path}")
        if metadata.get("model") != PRIMARY_MODEL:
            raise M1ProtocolError(
                f"teacher artifact {results_path} was measured with "
                f"{metadata.get('model')!r}, not {PRIMARY_MODEL!r}"
            )
        if metadata.get("adapter") is not None:
            raise M1ProtocolError(
                f"teacher artifact must use frozen original adapter=None: {metadata_path}"
            )
        observed_model_revision = _metadata_revision(metadata, "model")
        observed_tokenizer_revision = _metadata_revision(metadata, "tokenizer")
        if observed_model_revision != model_revision:
            raise M1ProtocolError(
                f"teacher model revision mismatch in {metadata_path}: "
                f"{observed_model_revision!r} != {model_revision!r}"
            )
        if observed_tokenizer_revision != tokenizer_revision:
            raise M1ProtocolError(
                f"teacher tokenizer revision mismatch in {metadata_path}: "
                f"{observed_tokenizer_revision!r} != {tokenizer_revision!r}"
            )
        hashes = metadata.get("hashes")
        if not isinstance(hashes, Mapping):
            raise M1ProtocolError(f"teacher metadata has no hashes object: {metadata_path}")
        raw_hash = file_sha256(results_path)
        if hashes.get("raw_output_sha256") != raw_hash:
            raise M1ProtocolError(
                f"teacher raw hash mismatch for {results_path}: metadata is stale or wrong"
            )
        battery_hash = file_sha256(battery_path)
        if hashes.get("battery_file_sha256") != battery_hash:
            raise M1ProtocolError(
                f"teacher battery hash mismatch for {battery_path}: metadata is stale or wrong"
            )
        chat_hash = metadata.get("chat_template_sha256")
        if not isinstance(chat_hash, str) or len(chat_hash) != 64:
            raise M1ProtocolError(
                f"teacher chat-template SHA-256 is missing/invalid: {metadata_path}"
            )
        chat_hashes.add(chat_hash)
        audited.append(
            {
                "source": spec.canonical_source,
                "results_path": str(results_path),
                "results_sha256": raw_hash,
                "battery_path": str(battery_path),
                "battery_sha256": battery_hash,
                "metadata_path": str(metadata_path),
                "metadata_sha256": file_sha256(metadata_path),
                "model": metadata["model"],
                "model_revision": observed_model_revision,
                "tokenizer_revision": observed_tokenizer_revision,
                "adapter": None,
                "chat_template_sha256": chat_hash,
            }
        )
    if len(chat_hashes) != 1:
        raise M1ProtocolError(
            f"teacher artifacts disagree on chat-template provenance: {sorted(chat_hashes)}"
        )
    return sorted(audited, key=lambda row: row["source"])


def build_supervised_examples(episodes: Sequence[Any]) -> tuple[SupervisedExample, ...]:
    """Derive deterministic top-two ``yes``/rest-``no`` labels from fixed ``W_rr``.

    No student model value is accepted by this function.  Python's historical
    stable ``sorted(..., reverse=True)`` behavior is made explicit by using
    ``candidate_index`` at ties, reproducing ``finetune_metacog.py`` exactly.
    """

    examples: list[SupervisedExample] = []
    seen_episode_ids: set[str] = set()
    for episode in episodes:
        if episode.source not in ALLOWED_TRAIN_SOURCES:
            raise M1ProtocolError(
                f"non-ID source reached label construction: {episode.source!r}"
            )
        if episode.uid in seen_episode_ids:
            raise M1ProtocolError(f"duplicate episode ID: {episode.uid}")
        seen_episode_ids.add(episode.uid)
        candidates = tuple(episode.candidates)
        if not candidates:
            raise M1ProtocolError(f"episode has no candidates: {episode.uid}")
        for candidate in candidates:
            if not math.isfinite(float(candidate.w_ref)):
                raise M1ProtocolError(
                    f"non-finite frozen W_rr for candidate {candidate.uid}"
                )
        ranked = sorted(
            candidates,
            key=lambda candidate: (-float(candidate.w_ref), candidate.candidate_index),
        )
        rank_by_id = {candidate.uid: rank for rank, candidate in enumerate(ranked, 1)}
        yes_ids = {candidate.uid for candidate in ranked[: min(TOP_K, len(ranked))]}
        for candidate in candidates:
            examples.append(
                SupervisedExample(
                    episode_id=episode.uid,
                    candidate_id=candidate.uid,
                    source=episode.source,
                    context=episode.context,
                    concept=candidate.concept,
                    target="yes" if candidate.uid in yes_ids else "no",
                    w_ref=float(candidate.w_ref),
                    teacher_rank=rank_by_id[candidate.uid],
                    utility_label=int(candidate.label == "load_bearing"),
                    candidate_fingerprint_sha256=candidate.fingerprint_sha256,
                )
            )
        observed_yes = sum(
            example.target == "yes" for example in examples[-len(candidates) :]
        )
        if observed_yes != min(TOP_K, len(candidates)):
            raise AssertionError("top-k teacher-label postcondition failed")
    return tuple(examples)


def teacher_tie_audit(episodes: Sequence[Any]) -> dict[str, Any]:
    """Count all W_rr ties and the ties that cross the top-2 boundary."""

    episode_rows = []
    for episode in episodes:
        ranked = sorted(
            episode.candidates,
            key=lambda candidate: (-float(candidate.w_ref), candidate.candidate_index),
        )
        score_counts: dict[float, int] = {}
        for candidate in ranked:
            score = float(candidate.w_ref)
            score_counts[score] = score_counts.get(score, 0) + 1
        tie_groups = sum(count > 1 for count in score_counts.values())
        tied_candidates = sum(count for count in score_counts.values() if count > 1)
        boundary_tie = len(ranked) > TOP_K and (
            float(ranked[TOP_K - 1].w_ref) == float(ranked[TOP_K].w_ref)
        )
        if tie_groups or boundary_tie:
            episode_rows.append(
                {
                    "episode_id": episode.uid,
                    "source": episode.source,
                    "tie_groups": tie_groups,
                    "tied_candidates": tied_candidates,
                    "top_k_boundary_tie": boundary_tie,
                    "boundary_score": float(ranked[TOP_K - 1].w_ref)
                    if boundary_tie
                    else None,
                    "tie_break": "candidate_index_ascending_original_order",
                }
            )
    return {
        "episodes": len(episodes),
        "episodes_with_any_tie": sum(row["tie_groups"] > 0 for row in episode_rows),
        "episodes_with_top_k_boundary_tie": sum(
            row["top_k_boundary_tie"] for row in episode_rows
        ),
        "tie_groups": sum(row["tie_groups"] for row in episode_rows),
        "tied_candidates": sum(row["tied_candidates"] for row in episode_rows),
        "episode_audit": episode_rows,
    }


def teacher_label_rows(
    train_examples: Sequence[SupervisedExample],
    validation_examples: Sequence[SupervisedExample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, examples in (
        ("train", train_examples),
        ("validation", validation_examples),
    ):
        for example in examples:
            row = asdict(example)
            row.pop("context")
            row["split"] = split
            row["label_source"] = "frozen_original_precomputed_W_rr_episode_top_2"
            rows.append(row)
    return rows


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def verbal_action_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return the exact historical measure.py action sets as (No, Yes)."""

    def variants(values: Sequence[str]) -> list[int]:
        token_ids = []
        for value in values:
            encoded = tokenizer.encode(value, add_special_tokens=False)
            if encoded:
                token_ids.append(int(encoded[0]))
        return sorted(set(token_ids))

    no_ids = variants(NO_VARIANTS)
    yes_ids = variants(YES_VARIANTS)
    if not no_ids or not yes_ids or set(no_ids) & set(yes_ids):
        raise M1ProtocolError("invalid or overlapping historical yes/no token sets")
    return no_ids, yes_ids


def token_lengths(tokenizer: Any, example: SupervisedExample) -> tuple[int, int]:
    prompt = render_admission_prompt(tokenizer, example.context, example.concept)
    prompt_ids = _token_ids(tokenizer, prompt)
    target_ids = _token_ids(tokenizer, example.target)
    if not target_ids:
        raise M1ProtocolError(f"empty target tokenization for {example.target!r}")
    return len(prompt_ids), len(target_ids)


def choose_effective_max_length(
    requested: int,
    maximum_required: int,
    ladder: Sequence[int] = LENGTH_LADDER,
) -> int:
    """Deterministically increase 1024 to 1536/2048 rather than truncate."""

    if requested not in ladder:
        raise M1ProtocolError(f"max sequence length must be one of {tuple(ladder)}")
    for length in ladder:
        if length >= requested and length >= maximum_required:
            return length
    raise M1ProtocolError(
        f"a sample needs {maximum_required} tokens, exceeding the M1 ceiling "
        f"of {max(ladder)}; review context handling before training"
    )


def truncation_statistics(
    tokenizer: Any,
    examples: Sequence[SupervisedExample],
    *,
    requested_max_length: int,
    effective_max_length: int,
) -> dict[str, Any]:
    lengths = [token_lengths(tokenizer, example) for example in examples]
    totals = [prompt + target for prompt, target in lengths]
    requested_removed = [max(0, total - requested_max_length) for total in totals]
    effective_removed = [max(0, total - effective_max_length) for total in totals]
    count = len(examples)
    return {
        "examples": count,
        "requested_max_length": requested_max_length,
        "effective_max_length": effective_max_length,
        "auto_increased": effective_max_length != requested_max_length,
        "max_original_tokens": max(totals, default=0),
        "mean_original_tokens": sum(totals) / count if count else 0.0,
        "would_truncate_at_requested_count": sum(value > 0 for value in requested_removed),
        "would_truncate_at_requested_tokens": sum(requested_removed),
        "actual_truncated_count": sum(value > 0 for value in effective_removed),
        "actual_truncated_tokens": sum(effective_removed),
    }


def encode_supervised_example(
    tokenizer: Any,
    example: SupervisedExample,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int | bool]]:
    prompt = render_admission_prompt(tokenizer, example.context, example.concept)
    prompt_ids = _token_ids(tokenizer, prompt)
    target_ids = _token_ids(tokenizer, example.target)
    if not target_ids or len(target_ids) >= max_length:
        raise M1ProtocolError(
            f"target for {example.candidate_id} cannot fit max_length={max_length}"
        )
    keep_prompt = max_length - len(target_ids)
    removed = max(0, len(prompt_ids) - keep_prompt)
    if removed:
        prompt_ids = prompt_ids[-keep_prompt:]
    input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
    labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, labels, attention_mask, {
        "truncated": bool(removed),
        "truncated_prompt_tokens": removed,
        "sequence_tokens": len(input_ids),
        "target_tokens": len(target_ids),
    }


def binary_roc_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float | None:
    """Return tie-aware binary ROC AUC without a scikit-learn dependency."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positives = [float(score) for label, score in zip(labels, scores) if bool(label)]
    negatives = [float(score) for label, score in zip(labels, scores) if not bool(label)]
    if not positives or not negatives:
        return None
    wins = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_validation_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["utility_label"]) for row in rows]
    teacher_labels = [int(row["teacher_target"] == "yes") for row in rows]
    scores = [float(row["yes_probability"]) for row in rows]
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    episode_aucs = []
    for episode_id in episode_ids:
        group = [row for row in rows if row["episode_id"] == episode_id]
        auc = binary_roc_auc(
            [int(row["utility_label"]) for row in group],
            [float(row["yes_probability"]) for row in group],
        )
        if auc is not None:
            episode_aucs.append(auc)
    per_source: dict[str, Any] = {}
    for source in sorted({str(row["source"]) for row in rows}):
        group = [row for row in rows if row["source"] == source]
        per_source[source] = {
            "verbal_auc": binary_roc_auc(
                [int(row["utility_label"]) for row in group],
                [float(row["yes_probability"]) for row in group],
            ),
            "teacher_alignment_auc": binary_roc_auc(
                [int(row["teacher_target"] == "yes") for row in group],
                [float(row["yes_probability"]) for row in group],
            ),
            "yes_rate": _mean(
                [float(float(row["yes_probability"]) >= 0.5) for row in group]
            ),
            "episodes": len({row["episode_id"] for row in group}),
            "candidates": len(group),
        }
    return {
        "verbal_auc": binary_roc_auc(labels, scores),
        "verbal_within_episode_auc": _mean(episode_aucs),
        "teacher_alignment_auc": binary_roc_auc(teacher_labels, scores),
        "yes_rate": _mean([float(score >= 0.5) for score in scores]),
        "episodes": len(episode_ids),
        "candidates": len(rows),
        "per_source": per_source,
    }


def planned_optimizer_steps(
    train_candidate_count: int,
    *,
    epochs: int,
    gradient_accumulation: int,
    max_steps: int,
    canary_steps: int = 0,
) -> int:
    if train_candidate_count < 1:
        raise M1ProtocolError("training candidate count must be positive")
    if epochs < 1 or gradient_accumulation < 1 or max_steps < 1:
        raise M1ProtocolError("epochs, gradient accumulation, and max steps must be positive")
    if canary_steps:
        if not 5 <= canary_steps <= 20:
            raise M1ProtocolError("canary_steps must be zero or between 5 and 20")
        return min(canary_steps, max_steps)
    steps_per_epoch = math.ceil(train_candidate_count / gradient_accumulation)
    return min(max_steps, epochs * steps_per_epoch)


def checkpoint_schedule(terminal_step: int) -> tuple[int, ...]:
    if terminal_step < 1:
        raise M1ProtocolError("terminal optimizer step must be positive")
    milestones = {step for step in CHECKPOINT_STEPS if step <= terminal_step}
    milestones.add(terminal_step)
    return tuple(sorted(milestones))


def select_best_checkpoint(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select maximum ID verbal AUC, resolving exact ties to earliest step."""

    eligible = []
    for record in records:
        auc = record.get("verbal_auc")
        step = record.get("step")
        if auc is None or isinstance(step, bool) or not isinstance(step, int):
            continue
        auc = float(auc)
        if math.isfinite(auc):
            eligible.append(record)
    if not eligible:
        raise M1ProtocolError("no finite ID verbal AUC is available for checkpoint lock")
    return min(eligible, key=lambda record: (-float(record["verbal_auc"]), record["step"]))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def software_versions() -> dict[str, str | None]:
    values = {
        "python": sys.version.split()[0],
        "cuda": torch.version.cuda,
    }
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def _revision_is_commit(value: str | None) -> bool:
    if value is None or len(value) != 40:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _resolved_tokenizer_commit(tokenizer: Any) -> str | None:
    effective = getattr(tokenizer, "_metacog_resolved_revision", None)
    if isinstance(effective, str):
        return effective
    direct = getattr(tokenizer, "_commit_hash", None)
    if isinstance(direct, str):
        return direct
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    value = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None
    return value if isinstance(value, str) else None


def _snapshot_commit_from_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    parts = Path(path).parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    candidate = parts[index + 1]
    return candidate if _revision_is_commit(candidate) else None


def resolve_cached_revision(
    model: str,
    revision: str | None,
    filenames: Sequence[str],
) -> str | None:
    """Resolve a Hub snapshot SHA, tolerating tokenizer metadata omissions.

    Transformers 4.51 can successfully load a tokenizer pinned to a commit yet
    expose neither ``tokenizer._commit_hash`` nor the init-kwargs equivalent.
    Inspect the already-cached snapshot path; if that also omits the value, a
    successful load at an explicit 40-character SHA makes that SHA effective.
    """

    try:
        from transformers.utils.hub import cached_file
    except ImportError:
        return revision if _revision_is_commit(revision) else None
    for filename in filenames:
        try:
            cached = cached_file(
                model,
                filename,
                revision=revision,
                local_files_only=True,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except (OSError, TypeError, ValueError):
            continue
        resolved = _snapshot_commit_from_path(cached)
        if resolved is not None:
            return resolved
    return revision if _revision_is_commit(revision) else None


def effective_resolved_revision(
    requested: str | None,
    reported: str | None,
    cached: str | None = None,
) -> str | None:
    return reported or cached or (requested if _revision_is_commit(requested) else None)


def _verify_pinned_revision(kind: str, requested: str | None, resolved: str | None) -> None:
    if _revision_is_commit(requested) and resolved is not None and resolved != requested:
        raise M1ProtocolError(
            f"{kind} resolved commit {resolved!r} does not match pinned {requested!r}"
        )


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tokenizer_revision = args.tokenizer_revision or args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=tokenizer_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_revision = resolve_cached_revision(
        args.model,
        tokenizer_revision,
        ("tokenizer_config.json", "tokenizer.json"),
    )
    resolved_revision = effective_resolved_revision(
        tokenizer_revision,
        _resolved_tokenizer_commit(tokenizer),
        cached_revision,
    )
    tokenizer._metacog_resolved_revision = resolved_revision
    _verify_pinned_revision(
        "tokenizer", tokenizer_revision, resolved_revision
    )
    return tokenizer


def load_lora_model(args: argparse.Namespace, device: str) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    reported_revision = getattr(model.config, "_commit_hash", None)
    cached_revision = resolve_cached_revision(
        args.model, args.model_revision, ("config.json",)
    )
    resolved_revision = effective_resolved_revision(
        args.model_revision, reported_revision, cached_revision
    )
    model.config._metacog_resolved_revision = resolved_revision
    _verify_pinned_revision("model", args.model_revision, resolved_revision)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha or 2 * args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, lora)
    model.train()
    non_lora_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if non_lora_trainable:
        raise M1ProtocolError(
            f"non-LoRA parameters unexpectedly trainable: {non_lora_trainable[:5]}"
        )
    return model


def validate_device(device: str) -> dict[str, Any]:
    if device != "cuda":
        raise M1ProtocolError("M1 training is bf16 CUDA-only; CPU is for unit tests/dry-run")
    if not torch.cuda.is_available():
        raise M1ProtocolError("CUDA is unavailable")
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    # One exact device name, taken from the campaign launcher's closed
    # allowlist (METACOG_EXPECTED_GPU).  This is still a hard substitution
    # guard: an unapproved accelerator fails closed exactly as before.
    if name != EXPECTED_GPU:
        raise M1ProtocolError(
            f"M1 requires {EXPECTED_GPU}, found {name!r}; do not silently substitute"
        )
    if not torch.cuda.is_bf16_supported():
        raise M1ProtocolError("the selected CUDA device does not report bf16 support")
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "index": index,
        "name": name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "bf16_supported": True,
    }


def gpu_memory_snapshot(device: str) -> dict[str, int | None]:
    if device != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _find_final_norm(model: Any) -> Any:
    for path in (
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "final_norm"),
    ):
        value = model
        for attribute in path:
            if not hasattr(value, attribute):
                break
            value = getattr(value, attribute)
        else:
            if isinstance(value, torch.nn.Module):
                return value
    raise M1ProtocolError("cannot locate final norm for workspace W_rr readout")


def _concept_token_variants(tokenizer: Any, concept: str) -> list[int]:
    values = []
    for variant in (f" {concept}", f" {concept.capitalize()}", concept, concept.capitalize()):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            values.append(int(encoded[0]))
    return list(dict.fromkeys(values))


@torch.no_grad()
def evaluate_canary_workspace_id(
    model: Any,
    tokenizer: Any,
    examples: Sequence[SupervisedExample],
    *,
    device: str,
    max_length: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run an adapter-on W_rr readout on one deterministic fixed-ID episode."""

    if not examples:
        raise M1ProtocolError("canary workspace evaluation has no ID examples")
    episode_id = min(example.episode_id for example in examples)
    selected = [example for example in examples if example.episode_id == episode_id]
    if any(example.source not in ALLOWED_TRAIN_SOURCES for example in selected):
        raise M1ProtocolError("canary workspace evaluator received an OOD source")
    contexts = {example.context for example in selected}
    if len(contexts) != 1:
        raise M1ProtocolError(f"inconsistent contexts in fixed ID episode {episode_id}")
    context = next(iter(contexts))
    tokenized = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = tokenized["input_ids"]
    if input_ids.shape[-1] > max_length:
        raise M1ProtocolError(
            f"raw workspace context has {input_ids.shape[-1]} tokens, above {max_length}"
        )
    batch = {key: value.to(device) for key, value in tokenized.items()}
    was_training = model.training
    model.eval()
    try:
        output = model(**batch, output_hidden_states=True, use_cache=False)
    finally:
        model.train(was_training)
    hidden_states = output.hidden_states
    if not hidden_states or len(hidden_states) < 2:
        raise M1ProtocolError("model returned no transformer hidden states for W_rr")
    get_base_model = getattr(model, "get_base_model", None)
    readout_model = get_base_model() if callable(get_base_model) else model
    final_norm = _find_final_norm(readout_model)
    unembedding = readout_model.get_output_embeddings()
    variant_ids = {
        example.candidate_id: _concept_token_variants(tokenizer, example.concept)
        for example in selected
    }
    missing = [candidate_id for candidate_id, ids in variant_ids.items() if not ids]
    if missing:
        raise M1ProtocolError(f"concept has no token variant for W_rr: {missing[0]}")
    reciprocal_ranks = {example.candidate_id: 0.0 for example in selected}
    for layer in hidden_states[1:]:
        hidden = layer[0, -1]
        weight = unembedding.weight
        norm_device = next(final_norm.parameters(), weight).device
        normalized = final_norm(hidden.to(device=norm_device, dtype=weight.dtype))
        logits = unembedding(normalized.to(weight.device)).float()
        for example in selected:
            best = reciprocal_ranks[example.candidate_id]
            for token_id in variant_ids[example.candidate_id]:
                rank = int((logits > logits[token_id]).sum().item()) + 1
                best = max(best, 1.0 / rank)
            reciprocal_ranks[example.candidate_id] = best
    rows = [
        {
            "episode_id": example.episode_id,
            "candidate_id": example.candidate_id,
            "source": example.source,
            "concept": example.concept,
            "utility_label": example.utility_label,
            "w_rr_adapter_on": reciprocal_ranks[example.candidate_id],
            "token_variant_ids": variant_ids[example.candidate_id],
        }
        for example in selected
    ]
    write_jsonl(output_path, rows)
    scores = [float(row["w_rr_adapter_on"]) for row in rows]
    finite = len(rows) == len(selected) and all(
        math.isfinite(score) and 0.0 < score <= 1.0 for score in scores
    )
    return {
        "performed": True,
        "scope": "fixed_id_validation_first_episode",
        "used_for_checkpoint_selection": False,
        "adapter_enabled": True,
        "episode_id": episode_id,
        "expected_candidate_rows": len(selected),
        "candidate_rows": len(rows),
        "all_finite": finite,
        "min_w_rr": min(scores, default=None),
        "max_w_rr": max(scores, default=None),
        "workspace_auc": binary_roc_auc(
            [example.utility_label for example in selected], scores
        ),
        "artifact": output_path.name,
        "readout": {
            "position": "final_raw_context_token",
            "layers": "1..n_max_reciprocal_rank",
            "final_norm": type(final_norm).__qualname__,
            "unembedding": type(unembedding).__qualname__,
            "candidate_variants": [
                "space+lowercase",
                "space+capitalized",
                "lowercase",
                "capitalized",
            ],
        },
    }


@torch.no_grad()
def evaluate_id_validation(
    model: Any,
    tokenizer: Any,
    examples: Sequence[SupervisedExample],
    *,
    device: str,
    max_length: int,
    step: int,
    score_path: Path,
) -> dict[str, Any]:
    """Evaluate only fixed ID examples; this function accepts no source specs."""

    invalid = sorted({example.source for example in examples} - set(ALLOWED_TRAIN_SOURCES))
    if invalid:
        raise M1ProtocolError(f"non-ID validation sources rejected: {invalid}")
    action_ids = verbal_action_token_ids(tokenizer)
    was_training = model.training
    model.eval()
    rows = []
    try:
        for example in examples:
            prompt = render_admission_prompt(tokenizer, example.context, example.concept)
            logits = binary_action_logits(
                model,
                tokenizer,
                [prompt],
                action_ids,
                device,
                max_length,
            )
            probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu())
            rows.append(
                {
                    "step": step,
                    "episode_id": example.episode_id,
                    "candidate_id": example.candidate_id,
                    "source": example.source,
                    "utility_label": example.utility_label,
                    "teacher_target": example.target,
                    "w_ref": example.w_ref,
                    "yes_probability": probability,
                }
            )
    finally:
        model.train(was_training)
    write_jsonl(score_path, rows)
    metrics = summarize_validation_scores(rows)
    metrics.update(
        {
            "schema_version": 1,
            "event": "id_validation",
            "step": step,
            "selection_scope": "id_validation",
            "selection_metric": "verbal_auc",
            "score_artifact": score_path.name,
        }
    )
    return metrics


def save_adapter_checkpoint(
    model: Any,
    run_dir: Path,
    step: int,
    validation_metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[Path, str]:
    relative = Path("checkpoints") / f"step-{step:06d}"
    destination = run_dir / relative
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.mkdir(parents=True)
    model.save_pretrained(destination, safe_serialization=True)
    write_json(destination / "validation_metrics.json", dict(validation_metrics))
    write_json(
        destination / "training_state.json",
        {
            "schema_version": 1,
            "step": step,
            "adapter_only": True,
            **dict(metadata),
        },
    )
    return relative, checkpoint_tree_sha256(destination)


def verify_adapter_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Read the saved adapter config and every tensor from CPU storage."""

    tree_hash = checkpoint_tree_sha256(checkpoint)
    config_path = checkpoint / "adapter_config.json"
    if not config_path.is_file():
        raise M1ProtocolError(f"missing adapter_config.json in {checkpoint}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    from peft import PeftConfig

    parsed_config = PeftConfig.from_pretrained(str(checkpoint))
    safetensors_path = checkpoint / "adapter_model.safetensors"
    binary_path = checkpoint / "adapter_model.bin"
    tensor_count = 0
    all_finite = True
    if safetensors_path.is_file():
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                tensor_count += 1
                all_finite = all_finite and bool(torch.isfinite(tensor.float()).all())
    elif binary_path.is_file():
        try:
            state = torch.load(binary_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(binary_path, map_location="cpu")
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor):
                tensor_count += 1
                all_finite = all_finite and bool(torch.isfinite(tensor.float()).all())
    else:
        raise M1ProtocolError(f"missing adapter weights in {checkpoint}")
    if tensor_count == 0 or not all_finite:
        raise M1ProtocolError(
            f"adapter serialization is empty or non-finite: tensors={tensor_count}, "
            f"all_finite={all_finite}"
        )
    return {
        "readable": True,
        "peft_config_loaded": True,
        "peft_type": str(getattr(parsed_config, "peft_type", raw_config.get("peft_type"))),
        "tensor_count": tensor_count,
        "all_tensors_finite": all_finite,
        "checkpoint_tree_sha256": tree_hash,
    }


@torch.no_grad()
def _fixed_example_action_logits(
    model: Any,
    tokenizer: Any,
    example: SupervisedExample,
    *,
    device: str,
    max_length: int,
) -> torch.Tensor:
    prompt = render_admission_prompt(tokenizer, example.context, example.concept)
    was_training = model.training
    model.eval()
    try:
        logits = binary_action_logits(
            model,
            tokenizer,
            [prompt],
            verbal_action_token_ids(tokenizer),
            device,
            max_length,
        )[0].float()
    finally:
        model.train(was_training)
    return logits.detach().cpu()


def _active_adapter_name(model: Any) -> str:
    value = getattr(model, "active_adapter", None)
    if isinstance(value, str):
        return value
    values = getattr(model, "active_adapters", None)
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return str(values[0])
    raise M1ProtocolError(f"expected one active PEFT adapter, found {values or value!r}")


def verify_live_peft_roundtrip(
    model: Any,
    tokenizer: Any,
    checkpoint: Path,
    example: SupervisedExample,
    *,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    """Reload the saved adapter in place, execute it, and restore live state.

    PEFT 0.13.2 does not reproduce weights exactly when an adapter saved as
    ``default`` is loaded under a different temporary name.  The difference is
    small at the tensor level but is large enough to make a logits equality
    check fail.  Reloading the checkpoint into its original adapter name tests
    the serialization path without exercising that adapter-renaming behavior.
    """

    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    original_adapter = _active_adapter_name(model)
    before = _fixed_example_action_logits(
        model, tokenizer, example, device=device, max_length=max_length
    )
    original_state = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(
            model, adapter_name=original_adapter
        ).items()
    }
    saved_state = load_peft_weights(str(checkpoint), device="cpu")
    if set(original_state) != set(saved_state):
        raise M1ProtocolError(
            "saved adapter keys differ from the live adapter during roundtrip"
        )
    zero_state = {key: torch.zeros_like(value) for key, value in original_state.items()}
    try:
        set_peft_model_state_dict(
            model, zero_state, adapter_name=original_adapter
        )
        set_peft_model_state_dict(
            model, saved_state, adapter_name=original_adapter
        )
        reloaded = _fixed_example_action_logits(
            model, tokenizer, example, device=device, max_length=max_length
        )
    finally:
        set_peft_model_state_dict(
            model, original_state, adapter_name=original_adapter
        )
    restored = _fixed_example_action_logits(
        model, tokenizer, example, device=device, max_length=max_length
    )
    reloaded_state = get_peft_model_state_dict(
        model, adapter_name=original_adapter
    )
    finite = bool(torch.isfinite(before).all()) and bool(torch.isfinite(reloaded).all())
    reload_difference = float(torch.max(torch.abs(before - reloaded)))
    restore_difference = float(torch.max(torch.abs(before - restored)))
    state_restore_difference = max(
        float(torch.max(torch.abs(original_state[key] - reloaded_state[key].detach().cpu())))
        for key in original_state
    )
    passed = (
        finite
        and reload_difference <= 1e-5
        and restore_difference <= 1e-5
        and state_restore_difference == 0.0
    )
    return {
        "performed": True,
        "passed": passed,
        "reload_method": "in_place_original_adapter",
        "original_adapter_restored": _active_adapter_name(model) == original_adapter,
        "logits_all_finite": finite,
        "max_abs_saved_vs_reloaded_logits": reload_difference,
        "max_abs_before_vs_restored_logits": restore_difference,
        "max_abs_before_vs_restored_state": state_restore_difference,
    }


def verify_adapter_enable_disable(
    model: Any,
    tokenizer: Any,
    example: SupervisedExample,
    *,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    """Execute fixed-ID logits with adapter on/off and audit LoRA-layer flags."""

    disable = getattr(model, "disable_adapter", None)
    if not callable(disable):
        return {"performed": False, "passed": False, "reason": "disable_adapter missing"}
    layers = [module for module in model.modules() if hasattr(module, "_disable_adapters")]
    before_flags = [bool(module._disable_adapters) for module in layers]
    enabled_before = _fixed_example_action_logits(
        model, tokenizer, example, device=device, max_length=max_length
    )
    with adapter_disabled(model):
        inside_flags = [bool(module._disable_adapters) for module in layers]
        disabled = _fixed_example_action_logits(
            model, tokenizer, example, device=device, max_length=max_length
        )
    after_flags = [bool(module._disable_adapters) for module in layers]
    enabled_after = _fixed_example_action_logits(
        model, tokenizer, example, device=device, max_length=max_length
    )
    finite = all(
        bool(torch.isfinite(values).all())
        for values in (enabled_before, disabled, enabled_after)
    )
    entered = bool(layers) and all(inside_flags)
    restored = after_flags == before_flags and torch.allclose(
        enabled_before, enabled_after, atol=1e-5, rtol=0.0
    )
    return {
        "performed": True,
        "passed": bool(finite and entered and restored),
        "fixed_candidate_id": example.candidate_id,
        "lora_layers_audited": len(layers),
        "disable_flags_entered": entered,
        "disable_flags_restored": after_flags == before_flags,
        "logits_all_finite": finite,
        "enabled_logits_restored": bool(restored),
        "enabled_yes_probability": float(torch.softmax(enabled_before, dim=-1)[1]),
        "disabled_yes_probability": float(torch.softmax(disabled, dim=-1)[1]),
        "max_abs_enabled_before_after": float(
            torch.max(torch.abs(enabled_before - enabled_after))
        ),
    }


def _checkpoint_event(
    *,
    model: Any,
    tokenizer: Any,
    validation_examples: Sequence[SupervisedExample],
    run_dir: Path,
    device: str,
    max_length: int,
    step: int,
    split_manifest_sha256: str,
    run_config_sha256: str,
) -> dict[str, Any]:
    score_relative = Path("validation_scores") / f"step-{step:06d}.jsonl"
    metrics = evaluate_id_validation(
        model,
        tokenizer,
        validation_examples,
        device=device,
        max_length=max_length,
        step=step,
        score_path=run_dir / score_relative,
    )
    metrics["score_artifact"] = score_relative.as_posix()
    checkpoint_relative, tree_hash = save_adapter_checkpoint(
        model,
        run_dir,
        step,
        metrics,
        {
            "selection_scope": "id_validation",
            "ood_evaluated": False,
            "split_manifest_sha256": split_manifest_sha256,
            "run_config_sha256": run_config_sha256,
        },
    )
    metrics["checkpoint_path"] = checkpoint_relative.as_posix()
    metrics["checkpoint_tree_sha256"] = tree_hash
    append_jsonl(run_dir / "validation_metrics.jsonl", metrics)
    print(
        f"[ID validation step={step}] verbal_auc={metrics['verbal_auc']} "
        f"yes_rate={metrics['yes_rate']}",
        flush=True,
    )
    return metrics


def _training_groups(
    examples: Sequence[SupervisedExample],
    *,
    gradient_accumulation: int,
    seed: int,
) -> Iterable[tuple[int, list[SupervisedExample]]]:
    """Yield shuffled accumulation groups forever (max_steps stops the caller)."""

    rng = random.Random(seed)
    epoch = 0
    while True:
        order = list(examples)
        rng.shuffle(order)
        for start in range(0, len(order), gradient_accumulation):
            yield epoch, order[start : start + gradient_accumulation]
        epoch += 1


def train(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[SupervisedExample],
    validation_examples: Sequence[SupervisedExample],
    run_dir: Path,
    effective_max_length: int,
    split_manifest_sha256: str,
    run_config_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_steps = planned_optimizer_steps(
        len(train_examples),
        epochs=args.epochs,
        gradient_accumulation=args.gradient_accumulation,
        max_steps=args.max_steps,
        canary_steps=args.canary_steps,
    )
    scheduled_checkpoints = set(checkpoint_schedule(target_steps))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise M1ProtocolError("LoRA model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    model.zero_grad(set_to_none=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    checkpoint_records = [
        _checkpoint_event(
            model=model,
            tokenizer=tokenizer,
            validation_examples=validation_examples,
            run_dir=run_dir,
            device=args.device,
            max_length=effective_max_length,
            step=0,
            split_manifest_sha256=split_manifest_sha256,
            run_config_sha256=run_config_sha256,
        )
    ]
    history: list[dict[str, Any]] = []
    stream = _training_groups(
        train_examples,
        gradient_accumulation=args.gradient_accumulation,
        seed=args.seed,
    )
    training_started = time.perf_counter()
    processed_examples = 0
    for step in range(1, target_steps + 1):
        epoch, group = next(stream)
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        losses: list[float] = []
        token_count = 0
        truncated_tokens = 0
        for example in group:
            input_ids, labels, attention_mask, encoding = encode_supervised_example(
                tokenizer, example, effective_max_length
            )
            token_count += int(encoding["sequence_tokens"])
            truncated_tokens += int(encoding["truncated_prompt_tokens"])
            output = model(
                input_ids=input_ids.unsqueeze(0).to(args.device),
                labels=labels.unsqueeze(0).to(args.device),
                attention_mask=attention_mask.unsqueeze(0).to(args.device),
            )
            loss = output.loss.float()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss at optimizer step {step}, candidate {example.candidate_id}"
                )
            (loss / len(group)).backward()
            losses.append(float(loss.detach().cpu()))
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at optimizer step {step}")
        optimizer.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        total_elapsed = time.perf_counter() - training_started
        processed_examples += len(group)
        record = {
            "schema_version": 1,
            "event": "optimizer_step",
            "step": step,
            "epoch_index": epoch,
            "epochs_completed": processed_examples / len(train_examples),
            "loss": sum(losses) / len(losses),
            "grad_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "microbatches": len(group),
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "sequence_tokens": token_count,
            "truncated_prompt_tokens": truncated_tokens,
            "step_seconds": elapsed,
            "examples_per_second": len(group) / elapsed,
            "tokens_per_second": token_count / elapsed,
            "cumulative_seconds": total_elapsed,
            "gpu_memory": gpu_memory_snapshot(args.device),
        }
        history.append(record)
        append_jsonl(run_dir / "training_metrics.jsonl", record)
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step}/{target_steps} loss={record['loss']:.6f} "
                f"grad={grad_norm:.4f} tok/s={record['tokens_per_second']:.1f}",
                flush=True,
            )
        if step in scheduled_checkpoints:
            checkpoint_records.append(
                _checkpoint_event(
                    model=model,
                    tokenizer=tokenizer,
                    validation_examples=validation_examples,
                    run_dir=run_dir,
                    device=args.device,
                    max_length=effective_max_length,
                    step=step,
                    split_manifest_sha256=split_manifest_sha256,
                    run_config_sha256=run_config_sha256,
                )
            )
    return history, checkpoint_records


def build_lock_manifest(
    *,
    run_dir: Path,
    checkpoint_records: Sequence[Mapping[str, Any]],
    split_manifest_sha256: str,
    run_config_sha256: str,
    provenance_sha256: str,
) -> dict[str, Any]:
    best = select_best_checkpoint(checkpoint_records)
    checkpoint_path = str(best["checkpoint_path"])
    actual_tree_hash = checkpoint_tree_sha256(run_dir / checkpoint_path)
    if actual_tree_hash != best["checkpoint_tree_sha256"]:
        raise M1ProtocolError("selected checkpoint tree changed before lock")
    selection_table = [
        {
            "step": int(record["step"]),
            "selection_scope": "id_validation",
            "selection_metric": "verbal_auc",
            "verbal_auc": record["verbal_auc"],
            "verbal_within_episode_auc": record["verbal_within_episode_auc"],
            "yes_rate": record["yes_rate"],
            "checkpoint_path": record["checkpoint_path"],
            "checkpoint_tree_sha256": record["checkpoint_tree_sha256"],
        }
        for record in sorted(checkpoint_records, key=lambda item: item["step"])
    ]
    manifest = {
        "schema_version": 1,
        "campaign_stage": "M1",
        "status": "LOCKED",
        "selection_scope": "id_validation",
        "selection_metric": "verbal_auc",
        "tie_break": "earliest_step",
        "ood_evaluated": False,
        "eligible_for_ood": True,
        "checkpoint_path": checkpoint_path,
        "step": int(best["step"]),
        "validation_auc": float(best["verbal_auc"]),
        "validation_yes_rate": best["yes_rate"],
        "checkpoint_tree_sha256": actual_tree_hash,
        "split_manifest_sha256": split_manifest_sha256,
        "run_config_sha256": run_config_sha256,
        "provenance_sha256": provenance_sha256,
        "validation_metrics_sha256": file_sha256(run_dir / "validation_metrics.jsonl"),
        "candidate_steps": [row["step"] for row in selection_table],
        "id_selection_table": selection_table,
        "candidate_checkpoints": selection_table,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def build_canary_manifest(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    validation_examples: Sequence[SupervisedExample],
    history: Sequence[Mapping[str, Any]],
    checkpoint_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
    effective_max_length: int,
) -> dict[str, Any]:
    terminal = args.canary_steps
    terminal_record = next(
        record for record in checkpoint_records if record["step"] == terminal
    )
    checkpoint_path = run_dir / terminal_record["checkpoint_path"]
    serialization = verify_adapter_checkpoint(checkpoint_path)
    roundtrip = verify_live_peft_roundtrip(
        model,
        tokenizer,
        checkpoint_path,
        validation_examples[0],
        device=args.device,
        max_length=effective_max_length,
    )
    verification = {
        **serialization,
        "live_peft_roundtrip": roundtrip,
        "passed": bool(serialization["readable"] and roundtrip["passed"]),
    }
    workspace = evaluate_canary_workspace_id(
        model,
        tokenizer,
        validation_examples,
        device=args.device,
        max_length=effective_max_length,
        output_path=run_dir / "canary_workspace_id.jsonl",
    )
    adapter_check = verify_adapter_enable_disable(
        model,
        tokenizer,
        validation_examples[0],
        device=args.device,
        max_length=effective_max_length,
    )
    finite = bool(history) and all(
        math.isfinite(float(row["loss"])) and math.isfinite(float(row["grad_norm"]))
        for row in history
    )
    passed = (
        len(history) == terminal
        and finite
        and bool(verification["passed"])
        and bool(adapter_check["passed"])
        and bool(workspace["performed"])
        and bool(workspace["all_finite"])
        and workspace["candidate_rows"] == workspace["expected_candidate_rows"]
        and all(float(row["tokens_per_second"]) > 0 for row in history)
    )
    return {
        "schema_version": 1,
        "campaign_stage": "M1_engineering_canary",
        "status": "PASS" if passed else "FAIL",
        "canary_passed": passed,
        "requested_optimizer_steps": terminal,
        "completed_optimizer_steps": len(history),
        "finite_loss_and_gradients": finite,
        "checkpoint_save_load": verification,
        "adapter_enable_disable_check": adapter_check,
        "throughput": {
            "mean_tokens_per_second": _mean(
                [float(row["tokens_per_second"]) for row in history]
            ),
            "mean_examples_per_second": _mean(
                [float(row["examples_per_second"]) for row in history]
            ),
        },
        "gpu_memory": history[-1]["gpu_memory"] if history else gpu_memory_snapshot(args.device),
        "workspace_post_training_evaluation": workspace,
        "selection_scope": "id_validation_canary_only",
        "ood_evaluated": False,
        "eligible_for_ood": False,
        "formal_lock_created": False,
        "terminal_checkpoint_path": terminal_record["checkpoint_path"],
        "terminal_checkpoint_tree_sha256": verification["checkpoint_tree_sha256"],
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-8B A5000 Metacognitive Alignment M1 LoRA trainer"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="required 40-character Hugging Face commit SHA",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="prefer a 40-character commit SHA; defaults to model revision",
    )
    parser.add_argument(
        "--train-spec",
        action="append",
        type=parse_training_spec,
        help="SOURCE=RESULTS_JSON::BATTERY_JSON; repeat for all three ID sources",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, choices=(16, 32), default=16)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--gradient-accumulation", type=int, choices=(4, 8), default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--max-sequence-length", type=int, choices=LENGTH_LADDER, default=1024
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--canary-steps",
        type=int,
        default=0,
        help="5-20 enables an engineering canary; zero is the formal M1 pilot",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and persist data/split/provenance without loading a model/tokenizer",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.model.rstrip("/") != PRIMARY_MODEL:
        parser.error(
            f"M1 primary target is {PRIMARY_MODEL}; fallback substitution is not allowed"
        )
    if args.seed not in ALLOWED_SEEDS:
        parser.error(
            f"M1 is preregistered for optimisation seeds {ALLOWED_SEEDS}; "
            "the three-seed replication uses 0, 1 and 2 with split seed 0"
        )
    if not _revision_is_commit(args.model_revision):
        parser.error("--model-revision must be a pinned 40-character commit SHA")
    effective_tokenizer_revision = args.tokenizer_revision or args.model_revision
    if not _revision_is_commit(effective_tokenizer_revision):
        parser.error("--tokenizer-revision must resolve to a pinned 40-character commit SHA")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be strictly between 0 and 1")
    if args.epochs != 2:
        parser.error("M1 uses exactly two nominal epochs")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.canary_steps and not 5 <= args.canary_steps <= 20:
        parser.error("--canary-steps must be zero or between 5 and 20")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("learning rate and max gradient norm must be positive")
    if args.lora_alpha < 0 or args.log_every < 1:
        parser.error("LoRA alpha must be non-negative and log interval positive")
    if args.device != "cuda" and not args.dry_run:
        parser.error("only --dry-run may run without CUDA")


def _chat_template_provenance(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    return {
        "source": "tokenizer.chat_template",
        "present": isinstance(template, str) and bool(template),
        "sha256": sha256(template.encode("utf-8")).hexdigest()
        if isinstance(template, str)
        else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    seed_everything(args.seed)

    specs = validate_training_specs(
        tuple(args.train_spec or default_training_specs("qwen3-8B"))
    )
    effective_tokenizer_revision = args.tokenizer_revision or args.model_revision
    teacher_artifacts = validate_teacher_artifacts(
        specs,
        model_revision=args.model_revision,
        tokenizer_revision=effective_tokenizer_revision,
    )
    data_bundle = build_training_bundle(
        specs,
        val_fraction=args.validation_fraction,
        seed=args.split_seed,
        top_k=TOP_K,
    )
    train_examples = build_supervised_examples(data_bundle.train_episodes)
    validation_examples = build_supervised_examples(data_bundle.validation_episodes)
    if not train_examples or not validation_examples:
        raise M1ProtocolError("train and validation candidate sets must both be non-empty")

    run_dir = Path(args.out_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = write_split_manifest(
        run_dir / "split_manifest.json", data_bundle.split_manifest
    )
    split_manifest_file_hash = file_sha256(run_dir / "split_manifest.json")
    label_path = run_dir / "teacher_labels.jsonl"
    write_jsonl(label_path, teacher_label_rows(train_examples, validation_examples))
    teacher_label_audit = {
        "schema_version": 1,
        "method": "stable W_rr descending equivalent: (-W_rr, candidate_index)",
        "target_text": {"positive": "yes", "negative": "no"},
        "top_k": TOP_K,
        "train": teacher_tie_audit(data_bundle.train_episodes),
        "validation": teacher_tie_audit(data_bundle.validation_episodes),
        "train_target_counts": {
            "yes": sum(example.target == "yes" for example in train_examples),
            "no": sum(example.target == "no" for example in train_examples),
        },
        "validation_target_counts": {
            "yes": sum(example.target == "yes" for example in validation_examples),
            "no": sum(example.target == "no" for example in validation_examples),
        },
    }
    write_json(run_dir / "teacher_label_audit.json", teacher_label_audit)

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "campaign_stage": "M1",
        "student": {
            "model": args.model,
            "requested_model_revision": args.model_revision,
            "requested_tokenizer_revision": args.tokenizer_revision or args.model_revision,
            "adapter": "LoRA",
        },
        "teacher": {
            "model": args.model,
            "requested_revision": args.model_revision,
            "frozen_original": True,
            "workspace_scores": "precomputed_W_rr",
            "workspace_scores_recomputed_during_training": False,
            "student_workspace_used_for_labels": False,
            "metadata_sidecars_verified": True,
            "label_rule": "within episode W_rr descending; original candidate order at ties; top-2 yes, rest no",
            "source_artifacts": teacher_artifacts,
        },
        "data_isolation": {
            "allowed_sources": sorted(ALLOWED_TRAIN_SOURCES),
            "observed_sources": sorted(source.source for source in data_bundle.sources),
            "ood_loaded": False,
            "ood_evaluated": False,
        },
        "split_manifest_sha256": split_manifest_file_hash,
        "split_manifest_self_hash": split_manifest["manifest_sha256"],
        "teacher_labels_sha256": file_sha256(label_path),
        "teacher_label_audit_sha256": file_sha256(
            run_dir / "teacher_label_audit.json"
        ),
    }

    target_steps = planned_optimizer_steps(
        len(train_examples),
        epochs=args.epochs,
        gradient_accumulation=args.gradient_accumulation,
        max_steps=args.max_steps,
        canary_steps=args.canary_steps,
    )
    base_config: dict[str, Any] = {
        "schema_version": 1,
        "campaign_stage": "M1",
        "mode": "engineering_canary" if args.canary_steps else "formal_seed_0_pilot",
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "training_sources": [spec.canonical_source for spec in specs],
        "train_episode_count": len(data_bundle.train_episodes),
        "validation_episode_count": len(data_bundle.validation_episodes),
        "train_candidate_count": len(train_examples),
        "validation_candidate_count": len(validation_examples),
        "teacher_top_k": TOP_K,
        "precision": DTYPE_NAME,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha or 2 * args.lora_rank,
        "lora_dropout": 0.0,
        "lora_target_modules": list(TARGET_MODULES),
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_steps_is_cap": True,
        "target_optimizer_steps": target_steps,
        "requested_max_sequence_length": args.max_sequence_length,
        "gradient_checkpointing": True,
        "max_grad_norm": args.max_grad_norm,
        "checkpoint_steps": list(checkpoint_schedule(target_steps)),
        "selection_scope": "id_validation",
        "selection_metric": "verbal_auc",
        "selection_tie_break": "earliest_step",
        "ood_evaluation_enabled": False,
        "software_versions": software_versions(),
    }

    if args.dry_run:
        write_json(
            run_dir / "truncation_stats.json",
            {
                "status": "not_computed_data_only_dry_run",
                "requested_max_sequence_length": args.max_sequence_length,
            },
        )
        provenance["resolved"] = None
        write_json(run_dir / "provenance.json", provenance)
        base_config["dry_run"] = True
        write_json(run_dir / "run_config.json", base_config)
        write_json(
            run_dir / "summary.json",
            {
                "status": "DRY_RUN_COMPLETE",
                "model_loaded": False,
                "gpu_used": False,
                "ood_evaluated": False,
                "artifact_paths": {
                    "run_config": "run_config.json",
                    "provenance": "provenance.json",
                    "split_manifest": "split_manifest.json",
                    "teacher_labels": "teacher_labels.jsonl",
                    "teacher_label_audit": "teacher_label_audit.json",
                },
            },
        )
        print(f"dry run complete; fixed split written to {run_dir}", flush=True)
        return 0

    device_provenance = validate_device(args.device)
    tokenizer = load_tokenizer(args)
    loaded_chat_template = _chat_template_provenance(tokenizer)
    teacher_chat_hash = teacher_artifacts[0]["chat_template_sha256"]
    if loaded_chat_template["sha256"] != teacher_chat_hash:
        raise M1ProtocolError(
            "loaded tokenizer chat template does not match frozen-teacher artifacts: "
            f"{loaded_chat_template['sha256']!r} != {teacher_chat_hash!r}"
        )
    no_token_ids, yes_token_ids = verbal_action_token_ids(tokenizer)
    all_examples = tuple(train_examples) + tuple(validation_examples)
    all_lengths = [sum(token_lengths(tokenizer, example)) for example in all_examples]
    effective_max_length = choose_effective_max_length(
        args.max_sequence_length, max(all_lengths, default=0)
    )
    truncation = {
        "schema_version": 1,
        "policy": "increase deterministically through 1024,1536,2048; never silently truncate",
        "train": truncation_statistics(
            tokenizer,
            train_examples,
            requested_max_length=args.max_sequence_length,
            effective_max_length=effective_max_length,
        ),
        "validation": truncation_statistics(
            tokenizer,
            validation_examples,
            requested_max_length=args.max_sequence_length,
            effective_max_length=effective_max_length,
        ),
    }
    if truncation["train"]["actual_truncated_count"] or truncation["validation"][
        "actual_truncated_count"
    ]:
        raise M1ProtocolError("effective sequence length still truncates an ID example")
    write_json(run_dir / "truncation_stats.json", truncation)
    tokenizer.save_pretrained(run_dir / "tokenizer")

    model = load_lora_model(args, args.device)
    base_model_config = model.get_base_model().config
    resolved_model_commit = effective_resolved_revision(
        args.model_revision,
        getattr(base_model_config, "_commit_hash", None),
        getattr(base_model_config, "_metacog_resolved_revision", None),
    )
    resolved_tokenizer_commit = _resolved_tokenizer_commit(tokenizer)
    provenance["resolved"] = {
        "model_name_or_path": getattr(
            model.get_base_model().config, "_name_or_path", args.model
        ),
        "model_commit": resolved_model_commit,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", args.model),
        "tokenizer_commit": resolved_tokenizer_commit,
        "tokenizer_class": type(tokenizer).__qualname__,
        "chat_template": loaded_chat_template,
        "verbal_readout": {
            "scoring": "sum next-token probability over unique first-token ids",
            "yes_variants": list(YES_VARIANTS),
            "no_variants": list(NO_VARIANTS),
            "yes_token_ids": yes_token_ids,
            "no_token_ids": no_token_ids,
        },
        "device": device_provenance,
    }
    provenance["teacher"]["resolved_revision"] = resolved_model_commit
    write_json(run_dir / "provenance.json", provenance)
    provenance_file_hash = file_sha256(run_dir / "provenance.json")

    base_config["dry_run"] = False
    base_config["effective_max_sequence_length"] = effective_max_length
    base_config["sequence_length_auto_increased"] = (
        effective_max_length != args.max_sequence_length
    )
    base_config["device"] = device_provenance
    base_config["resolved_model_commit"] = resolved_model_commit
    base_config["resolved_tokenizer_commit"] = resolved_tokenizer_commit
    base_config["yes_token_ids"] = yes_token_ids
    base_config["no_token_ids"] = no_token_ids
    base_config["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    base_config["total_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    write_json(run_dir / "run_config.json", base_config)
    run_config_file_hash = file_sha256(run_dir / "run_config.json")

    try:
        history, checkpoint_records = train(
            args=args,
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            validation_examples=validation_examples,
            run_dir=run_dir,
            effective_max_length=effective_max_length,
            split_manifest_sha256=split_manifest_file_hash,
            run_config_sha256=run_config_file_hash,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "ood_evaluated": False,
            "gpu_memory": gpu_memory_snapshot(args.device),
        }
        write_json(run_dir / "failure.json", failure)
        if args.canary_steps:
            write_json(
                run_dir / "canary_manifest.json",
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "canary_passed": False,
                    "eligible_for_ood": False,
                    "ood_evaluated": False,
                    "failure": failure,
                },
            )
        raise

    if args.canary_steps:
        try:
            canary = build_canary_manifest(
                args=args,
                model=model,
                tokenizer=tokenizer,
                validation_examples=validation_examples,
                history=history,
                checkpoint_records=checkpoint_records,
                run_dir=run_dir,
                effective_max_length=effective_max_length,
            )
        except Exception as exc:
            write_json(
                run_dir / "canary_manifest.json",
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "canary_passed": False,
                    "eligible_for_ood": False,
                    "ood_evaluated": False,
                    "failure": {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise
        write_json(run_dir / "canary_manifest.json", canary)
        status = canary["status"]
        locked_checkpoint = None
        exit_code = 0 if canary["canary_passed"] else 2
    else:
        observed_steps = {record["step"] for record in checkpoint_records}
        expected_steps = set(checkpoint_schedule(target_steps))
        missing = expected_steps - observed_steps
        if missing:
            raise M1ProtocolError(f"formal run missing preregistered checkpoints: {sorted(missing)}")
        lock = build_lock_manifest(
            run_dir=run_dir,
            checkpoint_records=checkpoint_records,
            split_manifest_sha256=split_manifest_file_hash,
            run_config_sha256=run_config_file_hash,
            provenance_sha256=provenance_file_hash,
        )
        write_json(run_dir / "lock_manifest.json", lock)
        status = "LOCKED"
        locked_checkpoint = lock["checkpoint_path"]
        exit_code = 0

    summary = {
        "schema_version": 1,
        "status": status,
        "mode": base_config["mode"],
        "completed_optimizer_steps": len(history),
        "checkpoint_steps": [record["step"] for record in checkpoint_records],
        "locked_checkpoint": locked_checkpoint,
        "ood_evaluated": False,
        "gpu_memory": gpu_memory_snapshot(args.device),
        "throughput": {
            "mean_tokens_per_second": _mean(
                [float(row["tokens_per_second"]) for row in history]
            ),
            "mean_examples_per_second": _mean(
                [float(row["examples_per_second"]) for row in history]
            ),
        },
        "artifacts": {
            "run_config": "run_config.json",
            "provenance": "provenance.json",
            "split_manifest": "split_manifest.json",
            "teacher_labels": "teacher_labels.jsonl",
            "teacher_label_audit": "teacher_label_audit.json",
            "truncation_stats": "truncation_stats.json",
            "training_metrics": "training_metrics.jsonl",
            "validation_metrics": "validation_metrics.jsonl",
            "lock_manifest": None if args.canary_steps else "lock_manifest.json",
            "canary_manifest": "canary_manifest.json" if args.canary_steps else None,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(f"M1 {status}: {run_dir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
