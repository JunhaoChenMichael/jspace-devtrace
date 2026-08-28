#!/usr/bin/env python3
"""Generate the mandatory Markdown report for the A5000 M1 gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA = "metacog-alignment-m1-report/v1"


class ReportError(RuntimeError):
    pass


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{label} must be a JSON object")
    return value


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"missing or unsafe decision ledger: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(f"invalid ledger JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ReportError(f"ledger row {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ReportError("decision ledger is empty")
    return rows


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"missing or unsafe {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(f"invalid {label} JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ReportError(f"{label} row {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ReportError(f"{label} is empty")
    return rows


def _require_fields(mapping: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in mapping or mapping[field] is None]
    if missing:
        raise ReportError(f"{label} is missing required fields: {missing}")


def _format_number(value: Any, digits: int = 3, signed: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    prefix = "+" if signed else ""
    return format(float(value), f"{prefix}.{digits}f")


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _json_block(value: Any) -> list[str]:
    return ["```json", json.dumps(value, indent=2, sort_keys=True), "```"]


def _condition_rows(ood: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Condition | Channel | Before pooled AUC | After pooled AUC | Delta | "
        "Before within-episode | After within-episode | Yes rate before | Yes rate after |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    conditions = ood.get("conditions", {})
    for condition in ("decoupled", "compositional"):
        payload = conditions.get(condition, {}) if isinstance(conditions, Mapping) else {}
        for channel in ("verbal", "workspace"):
            metrics = payload.get(channel, {}) if isinstance(payload, Mapping) else {}
            before = metrics.get("before", {}) if isinstance(metrics, Mapping) else {}
            after = metrics.get("after", {}) if isinstance(metrics, Mapping) else {}
            yes_before = _format_number(before.get("yes_rate")) if channel == "verbal" else "—"
            yes_after = _format_number(after.get("yes_rate")) if channel == "verbal" else "—"
            lines.append(
                "| "
                + " | ".join(
                    [
                        condition.title(),
                        "V" if channel == "verbal" else "W_rr",
                        _format_number(before.get("pooled_auc")),
                        _format_number(after.get("pooled_auc")),
                        _format_number(metrics.get("delta_pooled_auc"), signed=True),
                        _format_number(before.get("within_episode_auc")),
                        _format_number(after.get("within_episode_auc")),
                        yes_before,
                        yes_after,
                    ]
                )
                + " |"
            )
    return lines


def _qa_rows(ood: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Condition | Before accuracy | After accuracy | Delta | Drop (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    conditions = ood.get("conditions", {})
    for condition in ("decoupled", "compositional"):
        qa = _get(conditions, condition, "full_context_qa", default={})
        lines.append(
            "| "
            + " | ".join(
                [
                    condition.title(),
                    _format_number(qa.get("before_accuracy")),
                    _format_number(qa.get("after_accuracy")),
                    _format_number(qa.get("after_minus_before"), signed=True),
                    _format_number(qa.get("drop_percentage_points"), signed=True),
                ]
            )
            + " |"
        )
    return lines


def _bootstrap_rows(ood: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Condition | Channel | Delta estimate | 95% CI | Effective draws |",
        "|---|---|---:|---:|---:|",
    ]
    conditions = ood.get("conditions", {})
    for condition in ("decoupled", "compositional"):
        for channel in ("verbal", "workspace"):
            bootstrap = _get(
                conditions,
                condition,
                channel,
                "paired_episode_bootstrap",
                "after_minus_before",
                default={},
            )
            ci = bootstrap.get("ci_95") if isinstance(bootstrap, Mapping) else None
            rendered_ci = (
                f"[{_format_number(ci[0], signed=True)}, {_format_number(ci[1], signed=True)}]"
                if isinstance(ci, list) and len(ci) == 2
                else "N/A"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        condition.title(),
                        "V" if channel == "verbal" else "W_rr",
                        _format_number(bootstrap.get("estimate"), signed=True),
                        rendered_ci,
                        str(bootstrap.get("bootstrap_samples_effective", "N/A")),
                    ]
                )
                + " |"
            )
    return lines


def _checkpoint_rows(lock: Mapping[str, Any]) -> list[str]:
    candidates = lock.get("candidate_checkpoints", [])
    lines = [
        "| Step | ID verbal AUC | ID Yes rate | Path | Selected |",
        "|---:|---:|---:|---|:---:|",
    ]
    if not isinstance(candidates, list) or not candidates:
        lines.append(
            f"| {lock.get('step', 'N/A')} | {_format_number(lock.get('validation_auc'))} | "
            f"N/A | `{lock.get('checkpoint_path', 'N/A')}` | yes |"
        )
        return lines
    selected_step = lock.get("step")
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        step = row.get("step")
        lines.append(
            f"| {step} | {_format_number(row.get('validation_auc', row.get('verbal_auc')))} | "
            f"{_format_number(row.get('yes_rate'))} | `{row.get('checkpoint_path', row.get('path', 'N/A'))}` | "
            f"{'yes' if step == selected_step else 'no'} |"
        )
    return lines


def _artifact_rows(ledger: Iterable[Mapping[str, Any]]) -> list[str]:
    artifacts: dict[str, str] = {}
    for row in ledger:
        values = row.get("artifact_hashes")
        if isinstance(values, Mapping):
            for path, digest in values.items():
                if isinstance(path, str) and isinstance(digest, str):
                    artifacts[path] = digest
        for path_key, hash_key in (
            ("gate_path", "gate_sha256"),
            ("manifest", "manifest_sha256"),
            ("lock_manifest", "lock_manifest_sha256"),
            ("ood_result", "ood_result_sha256"),
        ):
            path, digest = row.get(path_key), row.get(hash_key)
            if isinstance(path, str) and isinstance(digest, str):
                artifacts[path] = digest
    lines = ["| Artifact | SHA-256 |", "|---|---|"]
    for path, digest in sorted(artifacts.items()):
        lines.append(f"| `{path}` | `{digest}` |")
    if len(lines) == 2:
        lines.append("| N/A | N/A |")
    return lines


def _assert_no_h100(ledger: Sequence[Mapping[str, Any]]) -> None:
    preflight = [row for row in ledger if row.get("event") == "gpu_preflight_passed"]
    if len(preflight) != 1 or _get(preflight[0], "gpu", "name") != "NVIDIA RTX A5000":
        raise ReportError("ledger does not prove exactly one passed A5000 preflight")
    for row in ledger:
        if row.get("event") != "command_started":
            continue
        argv = row.get("argv")
        if isinstance(argv, list) and any("h100" in str(token).lower() for token in argv):
            raise ReportError("ledger contains an H100 command")


def build_report(
    *,
    m0_gate: Mapping[str, Any],
    canary: Mapping[str, Any],
    m1_summary: Mapping[str, Any],
    lock: Mapping[str, Any],
    ood: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    run_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    teacher_label_audit: Mapping[str, Any],
    training_metrics: Sequence[Mapping[str, Any]],
) -> str:
    gpu_event = next(row for row in ledger if row.get("event") == "gpu_preflight_passed")
    gpu = gpu_event["gpu"]
    decision = ood.get("decision", "INVALID")
    if decision not in {"GREEN", "AMBER", "RED"}:
        raise ReportError("OOD result has no valid gate decision")
    _require_fields(
        canary,
        (
            "finite_loss_and_gradients",
            "checkpoint_save_load",
            "adapter_enable_disable_check",
            "workspace_post_training_evaluation",
            "throughput",
            "gpu_memory",
        ),
        "canary manifest",
    )
    _require_fields(
        run_config,
        (
            "model",
            "model_revision",
            "tokenizer_revision",
            "precision",
            "lora_rank",
            "gradient_accumulation",
            "learning_rate",
            "epochs",
            "target_optimizer_steps",
            "gradient_checkpointing",
        ),
        "M1 run config",
    )
    _require_fields(provenance, ("teacher", "data_isolation"), "M1 provenance")
    _require_fields(
        teacher_label_audit,
        ("method", "top_k", "train", "validation", "train_target_counts", "validation_target_counts"),
        "teacher-label audit",
    )
    gate_details = m0_gate.get("gate")
    if not isinstance(gate_details, Mapping):
        raise ReportError("M0 report has no gate details")
    losses = [float(row["loss"]) for row in training_metrics]
    gradients = [float(row["grad_norm"]) for row in training_metrics]
    formal_health = {
        "optimizer_steps_recorded": len(training_metrics),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "minimum_gradient_norm": min(gradients),
        "maximum_gradient_norm": max(gradients),
        "all_loss_and_gradients_finite": all(math.isfinite(value) for value in [*losses, *gradients]),
        "throughput": m1_summary.get("throughput"),
        "gpu_memory": m1_summary.get("gpu_memory"),
    }
    lines: list[str] = [
        "# Qwen3-8B A5000 Metacognitive Alignment M1 gate report",
        "",
        f"Report schema: `{REPORT_SCHEMA}`. Final A5000 decision: **{decision}**.",
        "",
        "## 1. M0 baseline reproduction status",
        "",
        f"M0 decision: **{m0_gate.get('decision', 'N/A')}**. Reference and tolerance:",
        "",
        *_json_block(
            {
                "reference": gate_details.get("reference"),
                "observed": gate_details.get("observed"),
                "absolute_delta": gate_details.get("absolute_delta"),
                "tolerance": gate_details.get("tolerance"),
            }
        ),
        "",
        "## 2. Model/tokenizer revisions",
        "",
        f"- Model: `{metadata.get('model', ood.get('model', 'N/A'))}`",
        f"- Model revision: `{metadata.get('model_revision', ood.get('model_revision', 'N/A'))}`",
        f"- Tokenizer revision: `{metadata.get('tokenizer_revision', ood.get('tokenizer_revision', 'N/A'))}`",
        f"- Chat-template SHA-256: `{metadata.get('chat_template_sha256', 'N/A')}`",
        "",
        "## 3. A5000 memory/throughput configuration",
        "",
        f"Preflight device: `{gpu.get('name')}`, total {gpu.get('total_mib')} MiB, "
        f"free {gpu.get('free_mib')} MiB. Canary runtime summary:",
        "",
        *_json_block(
            {
                "throughput": canary.get("throughput"),
                "gpu_memory": canary.get("gpu_memory"),
                "status": canary.get("status"),
            }
        ),
        "",
        "## 4. Teacher-label construction audit",
        "",
        "The formal trainer is restricted to Explicit, Evoked, and Evoked-G2; labels are "
        "top-2 Yes per episode from the frozen original workspace teacher.",
        "",
        *_json_block(
            {
                "teacher": provenance["teacher"],
                "data_isolation": provenance["data_isolation"],
                "teacher_label_audit": teacher_label_audit,
            }
        ),
        "",
        "## 5. Training configuration",
        "",
        *_json_block(run_config),
        "",
        "## 6. Loss/gradient health",
        "",
        *_json_block(
            {
                "canary_finite_loss_and_gradients": canary["finite_loss_and_gradients"],
                "canary_checkpoint_save_load": canary["checkpoint_save_load"],
                "canary_adapter_enable_disable": canary["adapter_enable_disable_check"],
                "canary_workspace_evaluation": canary["workspace_post_training_evaluation"],
                "formal_health": formal_health,
            }
        ),
        "",
        "## 7. ID checkpoint-selection table",
        "",
        *_checkpoint_rows(lock),
        "",
        "Selection used ID verbal AUC only; ties select the earliest checkpoint.",
        "",
        "## 8. Locked checkpoint",
        "",
        f"- Path: `{ood.get('checkpoint_path', lock.get('checkpoint_path', 'N/A'))}`",
        f"- Step: `{lock.get('step', 'N/A')}`",
        f"- ID verbal AUC: `{_format_number(lock.get('validation_auc'))}`",
        f"- Tree SHA-256: `{lock.get('checkpoint_tree_sha256', 'N/A')}`",
        "",
        "## 9. Decoupled V before/after",
        "",
        *_condition_rows(ood),
        "",
        "## 10. Decoupled W before/after",
        "",
        "The combined table above reports the preregistered W_rr before/after and delta.",
        "",
        "## 11. Compositional V/W before/after",
        "",
        "The combined table above reports both Compositional channels without tuning or reruns.",
        "",
        "## 12. Within-episode metrics",
        "",
        "Within-episode AUCs are shown alongside pooled AUCs in the combined table.",
        "",
        "## 13. Yes-rate before/after",
        "",
        "Yes rates use the fixed `V >= 0.5` decision and are shown in the combined table.",
        "",
        "## 14. Full-context QA before/after",
        "",
        *_qa_rows(ood),
        "",
        "## 15. Bootstrap CIs",
        "",
        f"All paired AUC intervals use {ood.get('bootstrap', {}).get('samples', 'N/A')} "
        "whole-episode cluster draws.",
        "",
        *_bootstrap_rows(ood),
        "",
        "## 16. GREEN / AMBER / RED decision",
        "",
        f"Decision: **{decision}**. Strong GREEN: `{ood.get('strong_green', False)}`.",
        f"Controlled AMBER branch authorized: `{ood.get('controlled_branch_authorized', False)}`.",
        "",
        *_json_block({"decision_reasons": ood.get("decision_reasons", [])}),
        "",
        "## 17. Artifact paths and hashes",
        "",
        *_artifact_rows(ledger),
        "",
        "## 18. No H100 job was launched",
        "",
        "No H100 job was launched. This invocation used the single verified NVIDIA RTX "
        "A5000 and stops here for manual review; it launched no M2 or later stage.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-gate", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--m1-summary", type=Path, required=True)
    parser.add_argument("--lock-manifest", type=Path, required=True)
    parser.add_argument("--ood-result", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    try:
        m0_gate = _load_object(args.m0_gate, "M0 gate")
        canary = _load_object(args.canary_manifest, "canary manifest")
        m1_summary = _load_object(args.m1_summary, "M1 summary")
        lock = _load_object(args.lock_manifest, "ID lock")
        ood = _load_object(args.ood_result, "OOD result")
        ledger = _load_ledger(args.decision_ledger)
        _assert_no_h100(ledger)
        metadata_path = Path(f"{args.m0_gate.parent / 'decoupled.json'}.metadata")
        metadata = _load_object(metadata_path, "M0 Decoupled metadata")
        m1_dir = args.m1_summary.parent
        run_config = _load_object(m1_dir / "run_config.json", "M1 run config")
        provenance = _load_object(m1_dir / "provenance.json", "M1 provenance")
        teacher_label_audit = _load_object(
            m1_dir / "teacher_label_audit.json", "M1 teacher-label audit"
        )
        training_metrics = _load_jsonl(
            m1_dir / "training_metrics.jsonl", "M1 training metrics"
        )
        report = build_report(
            m0_gate=m0_gate,
            canary=canary,
            m1_summary=m1_summary,
            lock=lock,
            ood=ood,
            ledger=ledger,
            metadata=metadata,
            run_config=run_config,
            provenance=provenance,
            teacher_label_audit=teacher_label_audit,
            training_metrics=training_metrics,
        )
    except ReportError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(report)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        parser.error(f"refusing to overwrite existing output: {args.out}")
    print(f"saved M1 gate report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
