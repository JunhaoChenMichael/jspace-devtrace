"""Strict Decoupled workspace no-harm analysis for three RL-QA seeds.

This module performs no model inference and never mutates its raw inputs.  It
joins one locked base ``measure.py --end-only`` artifact with seed-0/1/2
adapter artifacts, recomputes pooled W_rr AUCs, and uses shared whole-episode
bootstrap draws for paired adapter-minus-base confidence intervals.

Examples::

    python src/analysis/stage_b_rlqa_workspace_noharm.py analyze \
      --base path/to/workspace_base_decoupled.json \
      --seed0 path/to/workspace_rl-qa-s0-k2_decoupled_end_only.json \
      --seed1 path/to/workspace_rl-qa-s1-k2_decoupled_end_only.json \
      --seed2 path/to/workspace_rl-qa-s2-k2_decoupled_end_only.json \
      --out path/to/workspace_noharm.json

    python src/analysis/stage_b_rlqa_workspace_noharm.py validate \
      --base path/to/workspace_base_decoupled.json \
      --seed0 path/to/workspace_rl-qa-s0-k2_decoupled_end_only.json \
      --seed1 path/to/workspace_rl-qa-s1-k2_decoupled_end_only.json \
      --seed2 path/to/workspace_rl-qa-s2-k2_decoupled_end_only.json \
      --analysis path/to/workspace_noharm.json \
      --out path/to/workspace_noharm_validation.json

Both commands create their output exclusively and refuse to overwrite an
existing artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SEEDS = (0, 1, 2)
LABELS = frozenset({"load_bearing", "distractor", "filler"})
REQUIRED_FIELDS = frozenset(
    {"episode", "concept", "label", "W_end", "W_max", "W_rr"}
)
OPTIONAL_VERBAL_FIELDS = frozenset({"V", "V_raw"})


class WorkspaceNoHarmError(ValueError):
    """A raw measurement or derived artifact violated the sealed contract."""


@dataclass(frozen=True)
class Protocol:
    source: str = "decoupled"
    n_episodes: int = 68
    n_rows: int = 335
    seeds: tuple[int, ...] = SEEDS
    bootstrap_samples: int = 4000
    bootstrap_seed: int = 0
    alert_threshold: float = -0.03
    locked_base_sha256: str | None = None


FORMAL_PROTOCOL = Protocol(
    locked_base_sha256=(
        "0b22da736998f970dad1a4613dc0c63743fcb2ab4df94234b887c7978a331a1f"
    )
)


@dataclass(frozen=True)
class MeasurementRow:
    episode: int
    concept: str
    label: str
    w_end: float
    w_max: float
    w_rr: float

    @property
    def uid(self) -> tuple[int, str]:
        return self.episode, self.concept

    @property
    def join_key(self) -> tuple[int, str, str]:
        return self.episode, self.concept, self.label


def _error(message: str) -> None:
    raise WorkspaceNoHarmError(message)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkspaceNoHarmError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise WorkspaceNoHarmError(f"non-finite JSON constant {value!r}")


def _strict_load(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except WorkspaceNoHarmError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkspaceNoHarmError(
            f"cannot read strict JSON {path}: {exc}"
        ) from exc


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WorkspaceNoHarmError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise WorkspaceNoHarmError(
            f"refusing to overwrite existing file: {path}"
        ) from exc


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(f"{where} must be an object")
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _error(f"{where} must be finite")
    return result


def _probability(value: Any, where: str, *, positive: bool = False) -> float:
    result = _finite_number(value, where)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not lower_ok or result > 1.0:
        boundary = "(0, 1]" if positive else "[0, 1]"
        _error(f"{where} must be in {boundary}")
    return result


def _parse_row(value: Any, where: str) -> MeasurementRow:
    row = _mapping(value, where)
    fields = frozenset(row)
    allowed_shapes = {
        REQUIRED_FIELDS,
        REQUIRED_FIELDS | OPTIONAL_VERBAL_FIELDS,
    }
    if fields not in allowed_shapes:
        missing = sorted(REQUIRED_FIELDS - fields)
        unexpected = sorted(fields - REQUIRED_FIELDS - OPTIONAL_VERBAL_FIELDS)
        incomplete_optional = sorted(OPTIONAL_VERBAL_FIELDS & fields)
        _error(
            f"{where} schema mismatch: missing={missing}, unexpected={unexpected}, "
            f"optional_verbal_fields_present={incomplete_optional}"
        )

    episode = row.get("episode")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        _error(f"{where}.episode must be a non-negative integer")
    concept = row.get("concept")
    if not isinstance(concept, str) or not concept:
        _error(f"{where}.concept must be a non-empty string")
    label = row.get("label")
    if not isinstance(label, str) or label not in LABELS:
        _error(f"{where}.label must be one of {sorted(LABELS)}")

    w_end = _probability(row.get("W_end"), f"{where}.W_end")
    w_max = _probability(row.get("W_max"), f"{where}.W_max")
    if w_max != 0.0:
        _error(f"{where}.W_max must be exactly 0 for an end-only measurement")
    w_rr = _probability(row.get("W_rr"), f"{where}.W_rr", positive=True)
    for optional in OPTIONAL_VERBAL_FIELDS:
        if optional in row:
            _probability(row[optional], f"{where}.{optional}")
    return MeasurementRow(episode, concept, label, w_end, w_max, w_rr)


def _load_measurement(
    path: Path, role: str, protocol: Protocol
) -> tuple[MeasurementRow, ...]:
    payload = _strict_load(path)
    if not isinstance(payload, list):
        _error(f"{role} JSON root must be an array")
    if len(payload) != protocol.n_rows:
        _error(
            f"{role} row count mismatch: expected {protocol.n_rows}, got {len(payload)}"
        )
    rows = tuple(
        _parse_row(value, f"{role}[{index}]")
        for index, value in enumerate(payload)
    )

    uids = [row.uid for row in rows]
    duplicate_uids = sorted(uid for uid, count in Counter(uids).items() if count > 1)
    if duplicate_uids:
        _error(f"{role} contains duplicate (episode, concept) UID: {duplicate_uids[0]!r}")

    expected_episodes = set(range(protocol.n_episodes))
    observed_episodes = {row.episode for row in rows}
    if observed_episodes != expected_episodes:
        _error(
            f"{role} episode IDs mismatch: expected 0..{protocol.n_episodes - 1}, "
            f"got {sorted(observed_episodes)}"
        )
    load_bearing_by_episode = Counter(
        row.episode for row in rows if row.label == "load_bearing"
    )
    invalid = [
        episode
        for episode in range(protocol.n_episodes)
        if load_bearing_by_episode[episode] != 1
    ]
    if invalid:
        _error(
            f"{role} must contain exactly one load-bearing row per episode; "
            f"invalid episodes={invalid}"
        )
    return rows


def _validate_join(
    base: Sequence[MeasurementRow],
    seeds: Mapping[int, Sequence[MeasurementRow]],
) -> str:
    base_uids = [row.uid for row in base]
    base_labels = {row.uid: row.label for row in base}
    base_keys = [row.join_key for row in base]
    for seed, rows in seeds.items():
        seed_uids = [row.uid for row in rows]
        if seed_uids != base_uids:
            for index, (expected, actual) in enumerate(zip(base_uids, seed_uids)):
                if expected != actual:
                    _error(
                        f"seed {seed} UID/order mismatch at row {index}: "
                        f"expected {expected!r}, got {actual!r}"
                    )
            _error(f"seed {seed} UID sequence length mismatch")
        for row in rows:
            expected_label = base_labels[row.uid]
            if row.label != expected_label:
                _error(
                    f"seed {seed} label mismatch for UID {row.uid!r}: "
                    f"expected {expected_label!r}, got {row.label!r}"
                )
        if [row.join_key for row in rows] != base_keys:
            _error(f"seed {seed} (episode, concept, label) join mismatch")
    encoded = json.dumps(
        base_keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        _error("AUC vectors must be non-empty and equal length")
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        _error("AUC requires both utility classes")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    positive_rank_sum = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        positive_rank_sum += average_rank * sum(
            bool(labels[order[index]]) for index in range(start, stop)
        )
        start = stop
    u_statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(u_statistic / (positives * negatives))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        _error("bootstrap distribution is empty")
    position = quantile * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _shared_episode_draws(protocol: Protocol) -> tuple[tuple[int, ...], ...]:
    if protocol.bootstrap_samples <= 0:
        _error("bootstrap_samples must be positive")
    rng = np.random.default_rng(protocol.bootstrap_seed)
    values = rng.integers(
        0,
        protocol.n_episodes,
        size=(protocol.bootstrap_samples, protocol.n_episodes),
    )
    return tuple(tuple(int(value) for value in draw) for draw in values)


def _paired_auc_bootstrap(
    base: Sequence[MeasurementRow],
    adapted: Sequence[MeasurementRow],
    draws: Sequence[Sequence[int]],
) -> dict[str, Any]:
    labels = [row.label == "load_bearing" for row in base]
    base_scores = [row.w_rr for row in base]
    adapted_scores = [row.w_rr for row in adapted]
    base_auc = _auc(labels, base_scores)
    adapted_auc = _auc(labels, adapted_scores)
    by_episode: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(base):
        by_episode[row.episode].append(index)

    distribution: list[float] = []
    for draw in draws:
        indices = [index for episode in draw for index in by_episode[episode]]
        sampled_labels = [labels[index] for index in indices]
        distribution.append(
            _auc(sampled_labels, [adapted_scores[index] for index in indices])
            - _auc(sampled_labels, [base_scores[index] for index in indices])
        )
    return {
        "direction": "seed_minus_base",
        "estimate": adapted_auc - base_auc,
        "ci_95": [
            _percentile(distribution, 0.025),
            _percentile(distribution, 0.975),
        ],
        "bootstrap_samples_effective": len(distribution),
        "probability_gt_zero": sum(value > 0.0 for value in distribution)
        / len(distribution),
    }


def _seed_summary(values: Mapping[str, float]) -> dict[str, Any]:
    expected = [str(seed) for seed in SEEDS]
    if list(values) != expected:
        _error(f"incomplete seed metric: expected {expected}, got {list(values)}")
    ordered = [_finite_number(values[seed], f"seed metric {seed}") for seed in expected]
    return {
        "status": "complete",
        "individual": dict(values),
        "mean": statistics.fmean(ordered),
        "sample_std": statistics.stdev(ordered),
        "n_seeds": len(ordered),
        "ddof": 1,
    }


def _validate_protocol(protocol: Protocol) -> None:
    if protocol.source != "decoupled":
        _error("workspace no-harm source must be decoupled")
    if protocol.seeds != SEEDS:
        _error(f"seed contract must be {SEEDS}")
    if protocol.n_episodes <= 0 or protocol.n_rows <= 0:
        _error("episode and row counts must be positive")
    if protocol.bootstrap_samples <= 0:
        _error("bootstrap_samples must be positive")
    if not math.isfinite(protocol.alert_threshold):
        _error("alert threshold must be finite")
    if protocol.locked_base_sha256 is not None and (
        len(protocol.locked_base_sha256) != 64
        or any(character not in "0123456789abcdef" for character in protocol.locked_base_sha256)
    ):
        _error("locked_base_sha256 must be a lowercase SHA-256 digest")


def analyze_files(
    base_path: Path,
    seed0_path: Path,
    seed1_path: Path,
    seed2_path: Path,
    *,
    protocol: Protocol = FORMAL_PROTOCOL,
) -> dict[str, Any]:
    """Recompute the complete three-seed Decoupled W_rr no-harm report."""
    _validate_protocol(protocol)
    paths = {
        "base": Path(base_path),
        "0": Path(seed0_path),
        "1": Path(seed1_path),
        "2": Path(seed2_path),
    }
    normalized_paths = [str(path.resolve(strict=False)) for path in paths.values()]
    if len(set(normalized_paths)) != len(normalized_paths):
        _error("base and seed measurement paths must be distinct")

    base_sha256 = _file_sha256(paths["base"])
    if (
        protocol.locked_base_sha256 is not None
        and base_sha256 != protocol.locked_base_sha256
    ):
        _error(
            "base SHA-256 mismatch: expected "
            f"{protocol.locked_base_sha256}, got {base_sha256}"
        )
    base = _load_measurement(paths["base"], "base", protocol)
    seed_rows = {
        seed: _load_measurement(paths[str(seed)], f"seed{seed}", protocol)
        for seed in protocol.seeds
    }
    join_sha256 = _validate_join(base, seed_rows)
    draws = _shared_episode_draws(protocol)
    labels = [row.label == "load_bearing" for row in base]
    base_auc = _auc(labels, [row.w_rr for row in base])

    seeds: dict[str, Any] = {}
    auc_values: dict[str, float] = {}
    delta_values: dict[str, float] = {}
    alerts: dict[str, bool] = {}
    for seed in protocol.seeds:
        name = f"rl-qa-s{seed}-k2"
        rows = seed_rows[seed]
        workspace_auc = _auc(labels, [row.w_rr for row in rows])
        paired = _paired_auc_bootstrap(base, rows, draws)
        alert = paired["estimate"] < protocol.alert_threshold
        auc_values[str(seed)] = workspace_auc
        delta_values[str(seed)] = paired["estimate"]
        alerts[str(seed)] = alert
        seeds[str(seed)] = {
            "condition": name,
            "input": {
                "path": str(paths[str(seed)]),
                "sha256": _file_sha256(paths[str(seed)]),
            },
            "workspace_w_rr_pooled_auc": workspace_auc,
            "paired_vs_base": paired,
            "no_harm": {
                "rule": (
                    "seed_minus_base W_rr pooled AUC < "
                    f"{protocol.alert_threshold:g}"
                ),
                "alert_threshold": protocol.alert_threshold,
                "alert": alert,
            },
        }

    label_counts = Counter(row.label for row in base)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "stage_b_rlqa_three_seed_decoupled_workspace_noharm",
        "source": protocol.source,
        "metric": "W_rr pooled candidate AUC",
        "protocol": {
            "seeds": list(protocol.seeds),
            "n_episodes": protocol.n_episodes,
            "n_rows": protocol.n_rows,
            "measurement_mode": "end_only",
            "alert_threshold": protocol.alert_threshold,
            "locked_base_sha256": protocol.locked_base_sha256,
            "bootstrap": {
                "method": "episode_cluster_paired_percentile",
                "samples_requested": protocol.bootstrap_samples,
                "seed": protocol.bootstrap_seed,
                "rng": "numpy.random.default_rng / PCG64",
                "draw": "sample n episode IDs with replacement",
                "shared_draws_across_seeds": True,
                "confidence": 0.95,
                "percentile_interpolation": "linear",
            },
        },
        "join": {
            "uid": "(episode, concept)",
            "key": "(episode, concept, label)",
            "exact_row_order_required": True,
            "status": "exact-match",
            "key_sequence_sha256": join_sha256,
            "n_rows": len(base),
            "n_episodes": len({row.episode for row in base}),
            "label_counts": dict(sorted(label_counts.items())),
            "load_bearing_per_episode": 1,
        },
        "base": {
            "input": {
                "path": str(paths["base"]),
                "sha256": base_sha256,
            },
            "workspace_w_rr_pooled_auc": base_auc,
        },
        "seeds": seeds,
        "summary": {
            "workspace_w_rr_pooled_auc": _seed_summary(auc_values),
            "seed_minus_base_auc": _seed_summary(delta_values),
            "alerts": {
                "individual": alerts,
                "any_alert": any(alerts.values()),
                "n_alerts": sum(alerts.values()),
            },
        },
    }


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return f"{path}: keys {sorted(left)!r} != {sorted(right)!r}"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: array length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def validate_analysis_files(
    base_path: Path,
    seed0_path: Path,
    seed1_path: Path,
    seed2_path: Path,
    analysis_path: Path,
    *,
    protocol: Protocol = FORMAL_PROTOCOL,
) -> dict[str, Any]:
    """Independently recompute every analysis field and compare it exactly."""
    try:
        expected = analyze_files(
            base_path,
            seed0_path,
            seed1_path,
            seed2_path,
            protocol=protocol,
        )
        observed = _strict_load(Path(analysis_path))
        observed = _mapping(observed, "analysis root")
        difference = _first_difference(expected, observed)
        errors = [] if difference is None else [difference]
    except (WorkspaceNoHarmError, OSError, ValueError) as exc:
        errors = [str(exc)]
    analysis_path = Path(analysis_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "validator": "stage_b_rlqa_workspace_noharm_raw_recomputation",
        "status": "pass" if not errors else "fail",
        "analysis": {
            "path": str(analysis_path),
            "sha256": _file_sha256(analysis_path)
            if analysis_path.is_file()
            else None,
        },
        "raw_inputs": {
            "base": str(base_path),
            "seed0": str(seed0_path),
            "seed1": str(seed1_path),
            "seed2": str(seed2_path),
        },
        "errors": errors,
    }


def _add_raw_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--seed0", required=True, type=Path)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="build the no-harm report")
    _add_raw_arguments(analyze_parser)
    validate_parser = subparsers.add_parser(
        "validate", help="recompute and validate an existing report"
    )
    _add_raw_arguments(validate_parser)
    validate_parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze_files(args.base, args.seed0, args.seed1, args.seed2)
            _write_json_exclusive(args.out, result)
            print(f"saved RL-QA workspace no-harm analysis -> {args.out}")
            return 0
        result = validate_analysis_files(
            args.base,
            args.seed0,
            args.seed1,
            args.seed2,
            args.analysis,
        )
        _write_json_exclusive(args.out, result)
        print(f"saved RL-QA workspace no-harm validation -> {args.out}")
        return 0 if result["status"] == "pass" else 1
    except WorkspaceNoHarmError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
