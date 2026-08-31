"""Recompute and gate the immutable M0 metacognitive-alignment baseline.

The gate consumes the four raw per-candidate JSON arrays produced by
``experiments/measure.py``.  It does not load a model and is safe to run on CPU.
Both reports are creation-only: if either destination already exists, the
command fails before reading any input artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONDITIONS = ("explicit", "evoked", "decoupled", "compositional")
ALLOWED_LABELS = frozenset({"load_bearing", "distractor", "filler"})
DEFAULT_PAPER_V = 0.337
DEFAULT_PAPER_W_RR = 0.654
DEFAULT_TOLERANCE = 0.05
# A scale point is not a reproduction: a different model has no reason to land
# within 0.05 of the paper's 8B AUCs, so the 32B seed-0 plans replace the
# reproduction criterion with a repairable-reporting-gap criterion on Decoupled.
DEFAULT_MIN_REPORTING_GAP = 0.10
GATE_MODES = ("paper_reproduction", "scale_gap")


class M0GateError(ValueError):
    """Raised when an M0 input or output violates the analysis contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_paths_absent(paths: Sequence[Path]) -> None:
    rendered = [str(path) for path in paths]
    if len(set(rendered)) != len(rendered):
        raise M0GateError("JSON and Markdown outputs must be different paths")
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
        )


def _finite_score(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M0GateError(f"{where} must be a finite numeric score")
    score = float(value)
    if not math.isfinite(score):
        raise M0GateError(f"{where} must be finite")
    if score < 0.0 or score > 1.0:
        raise M0GateError(f"{where} must be in [0, 1], got {score}")
    return score


def _roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Tie-aware binary ROC AUC, expressed as positive/negative pair wins."""
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        raise M0GateError("ROC AUC requires at least one positive and one negative")
    wins = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def _load_optional_metadata(raw_path: Path, raw_sha256: str) -> dict[str, Any] | None:
    metadata_path = Path(f"{raw_path}.metadata")
    if not metadata_path.exists() and not metadata_path.is_symlink():
        return None
    if not metadata_path.is_file():
        raise M0GateError(f"metadata sidecar is not a file: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M0GateError(f"cannot read metadata sidecar {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise M0GateError(f"metadata sidecar must contain an object: {metadata_path}")
    hashes = metadata.get("hashes")
    if hashes is not None and not isinstance(hashes, dict):
        raise M0GateError(f"metadata hashes must be an object: {metadata_path}")
    recorded_raw_hash = (hashes or {}).get("raw_output_sha256")
    if recorded_raw_hash is not None and recorded_raw_hash != raw_sha256:
        raise M0GateError(
            f"raw output hash disagrees with metadata for {raw_path}: "
            f"recorded {recorded_raw_hash}, actual {raw_sha256}"
        )
    runtime = metadata.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        raise M0GateError(f"metadata runtime must be an object: {metadata_path}")
    counts = metadata.get("counts")
    if counts is not None and not isinstance(counts, dict):
        raise M0GateError(f"metadata counts must be an object: {metadata_path}")
    return {
        "path": str(metadata_path.resolve()),
        "sha256": file_sha256(metadata_path),
        "schema_version": metadata.get("schema_version"),
        "model": metadata.get("model"),
        "adapter": metadata.get("adapter"),
        "model_revision": metadata.get("model_revision"),
        "model_revision_recorded": "model_revision" in metadata,
        "tokenizer_revision": metadata.get("tokenizer_revision"),
        "tokenizer_revision_recorded": "tokenizer_revision" in metadata,
        "chat_template_sha256": metadata.get("chat_template_sha256"),
        "dtype": metadata.get("dtype"),
        "device": metadata.get("device"),
        "limit_episodes": metadata.get("limit_episodes"),
        "end_only": metadata.get("end_only"),
        "verbal_enabled": metadata.get("verbal_enabled"),
        "policy_input_includes_probe": metadata.get("policy_input_includes_probe"),
        "workspace_readout": metadata.get("workspace_readout"),
        "yes_token_ids": metadata.get("yes_token_ids"),
        "no_token_ids": metadata.get("no_token_ids"),
        "runtime_versions": (runtime or {}).get("versions"),
        "gpu": (runtime or {}).get("gpu"),
        "candidate_rows_written": (counts or {}).get("candidate_rows_written"),
        "candidates_skipped_no_token": (counts or {}).get(
            "candidates_skipped_no_token"
        ),
        "measure_source_sha256": (hashes or {}).get("measure_source_sha256"),
        "workspace_lens_source_sha256": (hashes or {}).get(
            "workspace_lens_source_sha256"
        ),
        "raw_output_hash_verified": recorded_raw_hash == raw_sha256,
    }


def load_rows(path: Path, condition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise M0GateError(f"{condition} raw input is not a file: {path}")
    raw_sha256 = file_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M0GateError(f"cannot read {condition} raw input {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise M0GateError(f"{condition} raw input must be a non-empty JSON array")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[Any, int]] = set()
    legacy_candidate_counts: dict[Any, int] = defaultdict(int)
    candidate_index_presence: set[bool] = set()
    for index, raw_row in enumerate(payload):
        where = f"{condition}[{index}]"
        if not isinstance(raw_row, dict):
            raise M0GateError(f"{where} must be an object")
        missing = {"episode", "concept", "label", "V", "W_rr"} - raw_row.keys()
        if missing:
            raise M0GateError(f"{where} is missing required keys: {sorted(missing)}")
        episode = raw_row["episode"]
        if isinstance(episode, bool) or not isinstance(episode, (int, str)):
            raise M0GateError(f"{where}.episode must be an integer or string")
        concept = raw_row["concept"]
        if not isinstance(concept, str) or not concept:
            raise M0GateError(f"{where}.concept must be a non-empty string")
        label = raw_row["label"]
        if label not in ALLOWED_LABELS:
            raise M0GateError(
                f"{where}.label must be one of {sorted(ALLOWED_LABELS)}, got {label!r}"
            )
        has_candidate_index = "candidate_index" in raw_row
        candidate_index_presence.add(has_candidate_index)
        if has_candidate_index:
            candidate_index = raw_row["candidate_index"]
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index < 0
            ):
                raise M0GateError(
                    f"{where}.candidate_index must be a non-negative integer"
                )
        else:
            # Released pre-M0 artifacts did not carry candidate_index and some
            # episodes legitimately repeat a concept.  Preserve their row order
            # as the only safe legacy identity instead of treating concept text
            # as a key.
            candidate_index = legacy_candidate_counts[episode]
            legacy_candidate_counts[episode] += 1
        identity = (episode, candidate_index)
        if identity in identities:
            raise M0GateError(f"{where} duplicates candidate identity {identity!r}")
        identities.add(identity)
        rows.append(
            {
                "episode": episode,
                "candidate_index": candidate_index,
                "concept": concept,
                "label": label,
                "V": _finite_score(raw_row["V"], where=f"{where}.V"),
                "W_rr": _finite_score(raw_row["W_rr"], where=f"{where}.W_rr"),
            }
        )

    if len(candidate_index_presence) > 1:
        raise M0GateError(
            f"{condition} raw input mixes rows with and without candidate_index"
        )

    metadata = _load_optional_metadata(path, raw_sha256)
    if (
        metadata is not None
        and metadata["candidate_rows_written"] is not None
        and metadata["candidate_rows_written"] != len(rows)
    ):
        raise M0GateError(
            f"candidate count disagrees with metadata for {path}: recorded "
            f"{metadata['candidate_rows_written']}, actual {len(rows)}"
        )
    source = {
        "path": str(path.resolve()),
        "sha256": raw_sha256,
        "candidate_identity": (
            "episode+candidate_index"
            if True in candidate_index_presence
            else "episode+legacy_row_order"
        ),
        "metadata": metadata,
    }
    return rows, source


def _signal_metrics(rows: Sequence[Mapping[str, Any]], signal: str) -> dict[str, Any]:
    labels = [row["label"] == "load_bearing" for row in rows]
    scores = [float(row[signal]) for row in rows]
    pooled_auc = _roc_auc(labels, scores)

    by_episode: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)
    episode_aucs = []
    for episode_rows in by_episode.values():
        episode_labels = [row["label"] == "load_bearing" for row in episode_rows]
        if not any(episode_labels) or all(episode_labels):
            continue
        episode_aucs.append(
            _roc_auc(episode_labels, [float(row[signal]) for row in episode_rows])
        )
    if not episode_aucs:
        raise M0GateError(f"no episode is evaluable for within-episode {signal} AUC")
    result = {
        "pooled_auc": pooled_auc,
        "within_episode_auc": float(sum(episode_aucs) / len(episode_aucs)),
        "within_episode_auc_episode_count": len(episode_aucs),
        "mean_score": float(sum(scores) / len(scores)),
    }
    if signal == "V":
        result["yes_rate"] = float(sum(score >= 0.5 for score in scores) / len(scores))
        result["yes_threshold"] = 0.5
    return result


def summarize_condition(
    condition: str, rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    v_metrics = _signal_metrics(rows, "V")
    w_metrics = _signal_metrics(rows, "W_rr")
    episode_count = len({row["episode"] for row in rows})
    labels = Counter(str(row["label"]) for row in rows)
    counts = {
        "episodes": episode_count,
        "candidates": len(rows),
        "labels": {label: labels.get(label, 0) for label in sorted(ALLOWED_LABELS)},
        "within_episode_auc_episodes": v_metrics["within_episode_auc_episode_count"],
    }
    return {
        "condition": condition,
        "source": dict(source),
        "signals": {"V": v_metrics, "W_rr": w_metrics},
        # These direct views keep the gate artifact easy to consume from shell,
        # notebooks, and future campaign orchestration without schema guessing.
        "pooled_auc": {
            "V": v_metrics["pooled_auc"],
            "W_rr": w_metrics["pooled_auc"],
        },
        "within_episode_auc": {
            "V": v_metrics["within_episode_auc"],
            "W_rr": w_metrics["within_episode_auc"],
        },
        "yes_rate": v_metrics["yes_rate"],
        "counts": counts,
    }


def validate_provenance(
    conditions: Mapping[str, Mapping[str, Any]], *, require_metadata: bool
) -> dict[str, Any]:
    metadata_by_condition = {
        condition: conditions[condition]["source"]["metadata"]
        for condition in CONDITIONS
    }
    missing = [
        condition
        for condition, metadata in metadata_by_condition.items()
        if metadata is None
    ]
    if require_metadata and missing:
        raise M0GateError(
            "strict M0 gate requires measurement metadata sidecars; missing for: "
            + ", ".join(missing)
        )

    present = {
        condition: metadata
        for condition, metadata in metadata_by_condition.items()
        if metadata is not None
    }
    for condition, metadata in present.items():
        if metadata["schema_version"] == "workspace_measurement_metadata.v2":
            raise M0GateError(
                f"{condition} was measured under schema v2, whose verbal score used "
                "py/(py+pn+1e-9): the guard epsilon dominated whenever the yes/no "
                "mass fell below 1e-9, so the value ranked absolute yes-probability "
                "rather than the yes-versus-no ratio. Re-measure under v3; v2 and v3 "
                "verbal scores are not comparable and must not be mixed."
            )
        if metadata["schema_version"] != "workspace_measurement_metadata.v3":
            raise M0GateError(
                f"{condition} metadata schema must be workspace_measurement_metadata.v3"
            )
        if not metadata["raw_output_hash_verified"]:
            raise M0GateError(f"{condition} metadata does not bind the raw output hash")
        if not isinstance(metadata["model"], str) or not metadata["model"]:
            raise M0GateError(f"{condition} metadata is missing the model identifier")
        if metadata["adapter"] is not None:
            raise M0GateError(f"{condition} M0 baseline must not use an adapter")
        if not metadata["model_revision_recorded"]:
            raise M0GateError(f"{condition} metadata does not record model_revision")
        if not metadata["tokenizer_revision_recorded"]:
            raise M0GateError(f"{condition} metadata does not record tokenizer_revision")
        template_hash = metadata["chat_template_sha256"]
        if (
            not isinstance(template_hash, str)
            or len(template_hash) != 64
            or any(character not in "0123456789abcdef" for character in template_hash)
        ):
            raise M0GateError(
                f"{condition} metadata has an invalid chat-template SHA-256"
            )
        if not isinstance(metadata["dtype"], str) or not metadata["dtype"]:
            raise M0GateError(f"{condition} metadata is missing dtype")
        if not isinstance(metadata["device"], str) or not metadata["device"]:
            raise M0GateError(f"{condition} metadata is missing resolved device")
        if metadata["limit_episodes"] != 0:
            raise M0GateError(f"{condition} M0 baseline must evaluate the full battery")
        if not isinstance(metadata["end_only"], bool):
            raise M0GateError(f"{condition} metadata is missing end_only")
        if metadata["verbal_enabled"] is not True:
            raise M0GateError(f"{condition} M0 baseline must include chat-template V")
        if metadata["policy_input_includes_probe"] is not False:
            raise M0GateError(f"{condition} measurement must remain probe-blind")
        if not isinstance(metadata["workspace_readout"], dict):
            raise M0GateError(f"{condition} metadata is missing workspace readout details")
        for field in ("yes_token_ids", "no_token_ids"):
            token_ids = metadata[field]
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(
                    isinstance(token_id, bool) or not isinstance(token_id, int)
                    for token_id in token_ids
                )
            ):
                raise M0GateError(f"{condition} metadata has invalid {field}")
        if set(metadata["yes_token_ids"]) & set(metadata["no_token_ids"]):
            raise M0GateError(f"{condition} metadata Yes/No token sets overlap")
        if not isinstance(metadata["runtime_versions"], dict):
            raise M0GateError(f"{condition} metadata is missing runtime versions")
        if not isinstance(metadata["gpu"], dict):
            raise M0GateError(f"{condition} metadata is missing GPU provenance")
        if (
            isinstance(metadata["candidate_rows_written"], bool)
            or not isinstance(metadata["candidate_rows_written"], int)
            or metadata["candidate_rows_written"] < 1
        ):
            raise M0GateError(f"{condition} metadata has an invalid candidate count")
        if metadata["candidates_skipped_no_token"] != 0:
            raise M0GateError(
                f"{condition} measurement skipped candidates without usable tokens"
            )
        for field in ("measure_source_sha256", "workspace_lens_source_sha256"):
            source_hash = metadata[field]
            if (
                not isinstance(source_hash, str)
                or len(source_hash) != 64
                or any(
                    character not in "0123456789abcdef" for character in source_hash
                )
            ):
                raise M0GateError(f"{condition} metadata has an invalid {field}")

    consistency_fields = (
        "model",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "dtype",
        "device",
        "end_only",
        "workspace_readout",
        "yes_token_ids",
        "no_token_ids",
        "runtime_versions",
        "gpu",
        "measure_source_sha256",
        "workspace_lens_source_sha256",
    )
    if present:
        reference_condition = next(
            condition for condition in CONDITIONS if condition in present
        )
        reference = present[reference_condition]
        for condition, metadata in present.items():
            for field in consistency_fields:
                if metadata[field] != reference[field]:
                    raise M0GateError(
                        f"measurement provenance mismatch for {field}: "
                        f"{reference_condition} != {condition}"
                    )
    return {
        "mode": "strict" if require_metadata else "allow_missing_metadata",
        "metadata_present": sorted(present),
        "metadata_missing": missing,
        "consistent_fields": list(consistency_fields) if present else [],
    }


def analyze_files(
    paths: Mapping[str, Path],
    *,
    paper_v: float = DEFAULT_PAPER_V,
    paper_w_rr: float = DEFAULT_PAPER_W_RR,
    tolerance: float = DEFAULT_TOLERANCE,
    require_metadata: bool = True,
    mode: str = "paper_reproduction",
    min_reporting_gap: float = DEFAULT_MIN_REPORTING_GAP,
) -> dict[str, Any]:
    if set(paths) != set(CONDITIONS):
        raise M0GateError(f"expected exactly these conditions: {list(CONDITIONS)}")
    for name, value in (
        ("paper_v", paper_v),
        ("paper_w_rr", paper_w_rr),
        ("tolerance", tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise M0GateError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise M0GateError(f"{name} must be finite")
    if not 0.0 <= paper_v <= 1.0 or not 0.0 <= paper_w_rr <= 1.0:
        raise M0GateError("paper reference AUCs must be in [0, 1]")
    if tolerance < 0.0 or tolerance > 1.0:
        raise M0GateError("tolerance must be in [0, 1]")
    if mode not in GATE_MODES:
        raise M0GateError(f"gate mode must be one of {list(GATE_MODES)}")
    if isinstance(min_reporting_gap, bool) or not isinstance(min_reporting_gap, (int, float)):
        raise M0GateError("min_reporting_gap must be numeric")
    if not math.isfinite(float(min_reporting_gap)) or not 0.0 <= float(min_reporting_gap) <= 1.0:
        raise M0GateError("min_reporting_gap must be finite and in [0, 1]")
    resolved_inputs = [str(paths[name].resolve()) for name in CONDITIONS]
    if len(set(resolved_inputs)) != len(CONDITIONS):
        raise M0GateError("each M0 condition must use a distinct raw input file")

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows, source = load_rows(paths[condition], condition)
        conditions[condition] = summarize_condition(condition, rows, source)
    provenance_audit = validate_provenance(
        conditions, require_metadata=require_metadata
    )

    decoupled = conditions["decoupled"]
    observed_v = float(decoupled["pooled_auc"]["V"])
    observed_w = float(decoupled["pooled_auc"]["W_rr"])
    delta_v = observed_v - float(paper_v)
    delta_w = observed_w - float(paper_w_rr)
    v_pass = abs(delta_v) <= float(tolerance)
    w_pass = abs(delta_w) <= float(tolerance)
    reporting_gap = observed_w - observed_v
    if mode == "paper_reproduction":
        decision = "GREEN" if v_pass and w_pass else "INVESTIGATE"
        gate = {
            "mode": mode,
            "condition": "decoupled",
            "criterion": "absolute pooled-AUC deviation <= tolerance for V and W_rr",
            "tolerance": float(tolerance),
            "reference": {"V": float(paper_v), "W_rr": float(paper_w_rr)},
            "observed": {"V": observed_v, "W_rr": observed_w},
            "delta": {"V": delta_v, "W_rr": delta_w},
            "absolute_delta": {"V": abs(delta_v), "W_rr": abs(delta_w)},
            "checks": {"V": v_pass, "W_rr": w_pass},
        }
    else:
        # A new scale point only has to show that there IS a reporting gap to
        # repair. Landing near the 8B numbers is not required, and must never
        # be engineered by changing prompts, readout or thinking mode.
        gap_pass = reporting_gap >= float(min_reporting_gap)
        decision = "GREEN" if gap_pass else "SCALE_BOUNDARY"
        gate = {
            "mode": mode,
            "condition": "decoupled",
            "criterion": "W_before - V_before >= min_reporting_gap",
            "min_reporting_gap": float(min_reporting_gap),
            "observed": {"V": observed_v, "W_rr": observed_w},
            "reporting_gap": reporting_gap,
            "checks": {"reporting_gap": gap_pass},
            "reference_only_paper_values": {
                "V": float(paper_v),
                "W_rr": float(paper_w_rr),
                "note": "historical 8B/paper values, reported for context; not a gate",
            },
        }
    return {
        "schema_version": "metacog_m0_gate.v1",
        "stage": "M0",
        "decision": decision,
        "gate": gate,
        "metric_definitions": {
            "positive_label": "load_bearing",
            "negative_labels": ["distractor", "filler"],
            "pooled_auc": "tie-aware ROC AUC over every candidate",
            "within_episode_auc": (
                "unweighted mean of tie-aware per-episode AUCs over episodes "
                "containing both classes"
            ),
            "yes_rate": "fraction of V scores >= 0.5",
        },
        "provenance_audit": provenance_audit,
        "conditions": conditions,
    }


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(result: Mapping[str, Any], json_path: Path) -> str:
    gate = result["gate"]
    lines = [
        "# Metacognitive Alignment M0 Baseline Gate",
        "",
        f"**Decision: {result['decision']}**",
        "",
        (
            "## Decoupled reproduction gate"
            if gate.get("mode", "paper_reproduction") == "paper_reproduction"
            else "## Decoupled reporting-gap gate"
        ),
        "",
    ]
    if gate.get("mode", "paper_reproduction") == "paper_reproduction":
        lines.extend(
            [
                "| Signal | Paper AUC | Observed AUC | Delta | |Delta| | Tolerance | Pass |",
                "|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for signal in ("V", "W_rr"):
            lines.append(
                "| {signal} | {reference} | {observed} | {delta} | {absolute} | "
                "{tolerance} | {passed} |".format(
                    signal=signal,
                    reference=_format_metric(gate["reference"][signal]),
                    observed=_format_metric(gate["observed"][signal]),
                    delta=f"{gate['delta'][signal]:+.6f}",
                    absolute=_format_metric(gate["absolute_delta"][signal]),
                    tolerance=_format_metric(gate["tolerance"]),
                    passed="yes" if gate["checks"][signal] else "no",
                )
            )
    else:
        reference = gate.get("reference_only_paper_values", {})
        lines.extend(
            [
                "| Quantity | Value |",
                "|---|---:|",
                f"| Decoupled V (before) | {_format_metric(gate['observed']['V'])} |",
                f"| Decoupled W_rr (before) | {_format_metric(gate['observed']['W_rr'])} |",
                f"| Reporting gap (W - V) | {_format_metric(gate['reporting_gap'])} |",
                f"| Required gap | {_format_metric(gate['min_reporting_gap'])} |",
                f"| Pass | {'yes' if gate['checks']['reporting_gap'] else 'no'} |",
                "",
                "Historical 8B/paper values are context only and gate nothing: "
                f"V {reference.get('V')}, W_rr {reference.get('W_rr')}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Condition metrics",
            "",
            "| Condition | Episodes | Candidates | V pooled | V within | Yes rate | "
            "W_rr pooled | W_rr within |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        summary = result["conditions"][condition]
        lines.append(
            "| {condition} | {episodes} | {candidates} | {v_pool} | {v_within} | "
            "{yes_rate} | {w_pool} | {w_within} |".format(
                condition=condition.title(),
                episodes=summary["counts"]["episodes"],
                candidates=summary["counts"]["candidates"],
                v_pool=_format_metric(summary["pooled_auc"]["V"]),
                v_within=_format_metric(summary["within_episode_auc"]["V"]),
                yes_rate=_format_metric(summary["yes_rate"]),
                w_pool=_format_metric(summary["pooled_auc"]["W_rr"]),
                w_within=_format_metric(summary["within_episode_auc"]["W_rr"]),
            )
        )

    lines.extend(["", "## Immutable inputs", ""])
    for condition in CONDITIONS:
        source = result["conditions"][condition]["source"]
        lines.append(
            f"- {condition}: `{source['path']}` (SHA-256 `{source['sha256']}`)"
        )
    lines.extend(
        [
            "",
            f"Machine-readable report: `{json_path.resolve()}`",
            "",
            "No training or GPU model execution is performed by this gate.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(result: Mapping[str, Any], out_json: Path, out_md: Path) -> None:
    ensure_output_paths_absent([out_json, out_md])
    result_with_artifacts = dict(result)
    result_with_artifacts["artifacts"] = {
        "json": str(out_json.resolve()),
        "markdown": str(out_md.resolve()),
    }
    json_payload = json.dumps(result_with_artifacts, indent=2, sort_keys=True) + "\n"
    markdown_payload = render_markdown(result_with_artifacts, out_json)
    outputs = ((out_json, json_payload), (out_md, markdown_payload))
    for path, _payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, payload in outputs:
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                handle.write(payload)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition}", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--paper-v", type=float, default=DEFAULT_PAPER_V)
    parser.add_argument("--paper-w-rr", type=float, default=DEFAULT_PAPER_W_RR)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--mode",
        choices=GATE_MODES,
        default="paper_reproduction",
        help=(
            "'paper_reproduction' reproduces the published 8B AUCs; "
            "'scale_gap' gates a new scale point on a repairable reporting gap"
        ),
    )
    parser.add_argument(
        "--min-reporting-gap",
        type=float,
        default=DEFAULT_MIN_REPORTING_GAP,
        help="scale_gap mode: required Decoupled W_before - V_before",
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="legacy-data escape hatch; strict campaign runs must not use this",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # Check all destinations before reading or hashing any raw input.
        ensure_output_paths_absent([args.out_json, args.out_md])
        result = analyze_files(
            {condition: getattr(args, condition) for condition in CONDITIONS},
            paper_v=args.paper_v,
            paper_w_rr=args.paper_w_rr,
            tolerance=args.tolerance,
            require_metadata=not args.allow_missing_metadata,
            mode=args.mode,
            min_reporting_gap=args.min_reporting_gap,
        )
        write_reports(result, args.out_json, args.out_md)
    except (FileExistsError, M0GateError, OSError) as exc:
        parser.error(str(exc))
    print(f"M0 {result['decision']} -> {args.out_json} ; {args.out_md}")
    return 0 if result["decision"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
