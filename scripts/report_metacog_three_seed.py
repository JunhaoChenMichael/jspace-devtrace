#!/usr/bin/env python3
"""Aggregate three one-shot M1 OOD attempts into the replication report.

The predeclared gate is applied to every seed independently; the replication is
GREEN only when all three seeds are GREEN.  The pooled numbers exist to
describe the effect, never to replace a per-seed gate.

The mean paired effect uses SHARED bootstrap draws: one episode resample is
drawn and every seed is re-scored on that same resample before the seeds are
averaged.  Averaging three independently drawn intervals would understate the
correlation between seeds that evaluate identical episodes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_metacog_m1_ood import auc, _percentile  # noqa: E402

REPORT_SCHEMA = "metacog-alignment-three-seed-report/v1"
GATE = {
    "min_delta_v": 0.15,
    "min_v_after": 0.50,
    "max_abs_delta_w": 0.03,
    "max_qa_drop_pp": 2.0,
    "strong_delta_v": 0.25,
}
DEFAULT_DRAWS = 4000


class ReportError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"missing or unsafe OOD result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReportError(f"OOD result must be a JSON object: {path}")
    if payload.get("stage") != "M1_OOD":
        raise ReportError(f"not an M1 OOD result: {path}")
    return payload


def _episode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[Any, list[Mapping[str, Any]]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["episode"], []).append(row)
    return grouped


def _pooled_auc(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return auc([float(row[key]) for row in rows], [int(row["label"] == "load_bearing") for row in rows])


def seed_metrics(payload: Mapping[str, Any], condition: str = "decoupled") -> dict[str, Any]:
    block = payload["conditions"][condition]
    qa = block["full_context_qa"]
    before = qa["per_episode"]["before"]
    after = qa["per_episode"]["after"]
    qa_before = sum(int(bool(row["correct"])) for row in before) / len(before)
    qa_after = sum(int(bool(row["correct"])) for row in after) / len(after)
    delta_v = float(block["verbal"]["delta_pooled_auc"])
    v_after = float(block["verbal"]["after"]["pooled_auc"])
    delta_w = float(block["workspace"]["delta_pooled_auc"])
    qa_drop_pp = (qa_before - qa_after) * 100.0
    checks = {
        "delta_v_at_least_0.15": delta_v >= GATE["min_delta_v"],
        "v_after_above_0.50": v_after > GATE["min_v_after"],
        "abs_delta_w_below_0.03": abs(delta_w) < GATE["max_abs_delta_w"],
        "qa_drop_at_most_2pp": qa_drop_pp <= GATE["max_qa_drop_pp"],
    }
    return {
        "delta_v": delta_v,
        "v_before": float(block["verbal"]["before"]["pooled_auc"]),
        "v_after": v_after,
        "delta_w": delta_w,
        "w_before": float(block["workspace"]["before"]["pooled_auc"]),
        "w_after": float(block["workspace"]["after"]["pooled_auc"]),
        "full_context_qa_before": qa_before,
        "full_context_qa_after": qa_after,
        "full_context_qa_drop_pp": qa_drop_pp,
        "delta_v_ci_95": block["verbal"]["paired_episode_bootstrap"]["after_minus_before"]["ci_95"],
        "delta_v_probability_gt_zero": block["verbal"]["paired_episode_bootstrap"][
            "after_minus_before"
        ].get("probability_gt_zero"),
        "checks": checks,
        "gate": "GREEN" if all(checks.values()) else "NOT_GREEN",
        "strong_green": all(checks.values()) and delta_v >= GATE["strong_delta_v"],
    }


def shared_draw_mean_effect(
    payloads: Sequence[Mapping[str, Any]],
    condition: str = "decoupled",
    key: str = "V",
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean across seeds of the paired delta, recomputed on shared resamples."""

    grouped = []
    for payload in payloads:
        block = payload["conditions"][condition]
        grouped.append(
            (
                _episode_rows(block["per_item"]["before"]),
                _episode_rows(block["per_item"]["after"]),
            )
        )
    episodes = sorted(grouped[0][0])
    for before, after in grouped:
        if sorted(before) != episodes or sorted(after) != episodes:
            raise ReportError("seeds do not share one evaluation episode set")

    point = statistics.fmean(
        _pooled_auc([r for e in episodes for r in after[e]], key)
        - _pooled_auc([r for e in episodes for r in before[e]], key)
        for before, after in grouped
    )
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        resample = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        per_seed = []
        for before, after in grouped:
            before_rows = [row for episode in resample for row in before[episode]]
            after_rows = [row for episode in resample for row in after[episode]]
            try:
                per_seed.append(_pooled_auc(after_rows, key) - _pooled_auc(before_rows, key))
            except Exception:  # a degenerate resample has one utility class
                per_seed = []
                break
        if per_seed:
            estimates.append(statistics.fmean(per_seed))
    if not estimates:
        raise ReportError("every shared bootstrap draw was degenerate")
    return {
        "estimate": point,
        "ci_95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "draws_effective": len(estimates),
        "draws_requested": draws,
        "unit": "episode_cluster",
        "shared_draws_across_seeds": True,
    }


def build(payloads: Mapping[int, Mapping[str, Any]], *, draws: int = DEFAULT_DRAWS) -> dict[str, Any]:
    if sorted(payloads) != [0, 1, 2]:
        raise ReportError(f"the replication needs seeds 0, 1 and 2; got {sorted(payloads)}")
    ordered = [payloads[seed] for seed in (0, 1, 2)]

    revisions = {(p["model"], p["model_revision"], p["tokenizer_revision"]) for p in ordered}
    if len(revisions) != 1:
        raise ReportError("seeds did not share one model/tokenizer pin")
    if len({p["checkpoint_tree_sha256"] for p in ordered}) != 3:
        raise ReportError("two seeds report the same checkpoint; they are not independent runs")
    if len({p["attempt_id"] for p in ordered}) != 3:
        raise ReportError("two seeds share an OOD attempt id")
    for condition in ("decoupled", "compositional"):
        identities = {p["conditions"][condition]["candidate_identity_sha256"] for p in ordered}
        if len(identities) != 1:
            raise ReportError(f"seeds evaluated different {condition} items")

    per_seed = {seed: seed_metrics(payloads[seed]) for seed in (0, 1, 2)}
    compositional = {seed: seed_metrics(payloads[seed], "compositional") for seed in (0, 1, 2)}
    green = [per_seed[seed]["gate"] == "GREEN" for seed in (0, 1, 2)]

    def stats(field: str, block: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
        values = [block[seed][field] for seed in (0, 1, 2)]
        return {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values),
        }

    return {
        "schema_version": REPORT_SCHEMA,
        "model": ordered[0]["model"],
        "model_revision": ordered[0]["model_revision"],
        "tokenizer_revision": ordered[0]["tokenizer_revision"],
        "gate_thresholds": GATE,
        "per_seed": per_seed,
        "per_seed_compositional": compositional,
        "aggregate_decoupled": {
            field: stats(field, per_seed)
            for field in ("delta_v", "v_after", "delta_w", "full_context_qa_drop_pp")
        },
        "aggregate_compositional": {
            field: stats(field, compositional)
            for field in ("delta_v", "v_after", "delta_w", "full_context_qa_drop_pp")
        },
        "shared_draw_mean_delta_v_decoupled": shared_draw_mean_effect(ordered, draws=draws),
        "replication_decision": "GREEN" if all(green) else "NOT_GREEN",
        "strong_green_seeds": [seed for seed in (0, 1, 2) if per_seed[seed]["strong_green"]],
    }


def _table(title: str, block: Mapping[int, Mapping[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Seed | V before | V after | delta V | delta W | full-context QA before | after | drop (pp) | gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in (0, 1, 2):
        row = block[seed]
        lines.append(
            f"| {seed} | {row['v_before']:.5f} | {row['v_after']:.5f} | {row['delta_v']:+.5f} | "
            f"{row['delta_w']:+.5f} | {row['full_context_qa_before']:.4f} | "
            f"{row['full_context_qa_after']:.4f} | {row['full_context_qa_drop_pp']:+.2f} | "
            f"{row['gate']}{' (strong)' if row['strong_green'] else ''} |"
        )
    return [*lines, ""]


def operational_appendix(
    run_dirs: Mapping[int, Path], provenance: Mapping[str, Any] | None, deviations: Sequence[str]
) -> list[str]:
    """Command plan, inventory, hashes, attempt records, exceptions, raw logs."""

    lines = ["## 9. Operator command plan and run directories", ""]
    for seed in sorted(run_dirs):
        run_dir = run_dirs[seed]
        lines.append(f"### Seed {seed} — `{run_dir}`")
        lines.append("")
        plan_path = run_dir / "campaign_plan.json"
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            lines += ["Executed command plan (ID phase):", "", "```"]
            for stage, spec in (
                ("m0", plan["stages"]["m0"][0]),
                ("m0_gate", plan["stages"]["m0_gate"]),
                ("teacher_prep", plan["stages"]["teacher_prep"]),
                ("canary", plan["stages"]["canary"]),
                ("m1", plan["stages"]["m1"]),
            ):
                lines.append(f"[{stage}] " + " ".join(spec["argv"]))
            lines += ["```", ""]
        lock_path = run_dir / "id_lock" / "lock_manifest.json"
        if lock_path.is_file():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lines += [
                f"- ID-only locked step: **{lock['step']}** "
                f"(selection scope `{lock['selection_scope']}`, metric `{lock['selection_metric']}`, "
                f"tie-break `{lock['tie_break']}`)",
                f"- Locked checkpoint: `{lock['checkpoint_path']}`",
                f"- Checkpoint tree SHA-256: `{lock['checkpoint_tree_sha256']}`",
                f"- ID validation AUC at lock: {lock['validation_auc']:.5f}",
            ]
        attempt = run_dir / "ood" / "attempt_started.json"
        if attempt.is_file():
            record = json.loads(attempt.read_text(encoding="utf-8"))
            lines += [
                f"- OOD attempt id: `{record['attempt_id']}` "
                f"(limit {record['attempt_limit']}, conditions {record['conditions']})",
                f"- OOD attempt started: {record['started_at_utc']}",
            ]
        for name, label in (
            ("decision_ledger.jsonl", "decision ledger"),
            ("command_logs", "raw command logs"),
            ("report/M1_GATE_REPORT.md", "per-seed M1 gate report"),
            ("ID_LOCK_STOP.json", "ID-phase stop marker"),
            ("STOP.json", "final stop marker"),
        ):
            target = run_dir / name
            if target.exists():
                lines.append(f"- {label}: `{target}`")
        lines.append("")

    if provenance is not None:
        lines += [
            "## 10. Environment, hardware and hashes",
            "",
            f"- Repository commit: `{provenance.get('git_commit')}`",
            f"- Python {provenance.get('python')}, conda prefix `{provenance.get('conda_prefix')}`",
            "- Packages: "
            + ", ".join(f"{k} {v}" for k, v in sorted((provenance.get("packages") or {}).items()) if v),
            "",
            "GPU inventory reported by the collecting host:",
            "",
            "```",
            str(provenance.get("nvidia_smi")),
            "```",
            "",
            "Benchmark hashes:",
            "",
            "| File | SHA-256 |",
            "|---|---|",
            *[
                f"| `data/benchmarks/{name}` | `{digest}` |"
                for name, digest in sorted((provenance.get("battery_sha256") or {}).items())
            ],
            "",
            "Executed source hashes:",
            "",
            "| File | SHA-256 |",
            "|---|---|",
            *[
                f"| `{name}` | `{digest}` |"
                for name, digest in sorted((provenance.get("source_sha256") or {}).items())
            ],
            "",
        ]

    if deviations:
        lines += ["## 11. Exceptions and deviations from the controlling plan", ""]
        lines += [f"{index}. {text}" for index, text in enumerate(deviations, 1)]
        lines.append("")
    return lines


def render(
    summary: Mapping[str, Any],
    *,
    gpu: str,
    sources: Mapping[int, str],
    run_dirs: Mapping[int, Path] | None = None,
    provenance: Mapping[str, Any] | None = None,
    deviations: Sequence[str] = (),
) -> str:
    agg = summary["aggregate_decoupled"]
    shared = summary["shared_draw_mean_delta_v_decoupled"]
    lines = [
        f"# Qwen3-8B Binary Metacognitive Alignment: three-seed {gpu} replication",
        "",
        f"Report schema: `{summary['schema_version']}`. "
        f"Replication decision: **{summary['replication_decision']}**.",
        "",
        "The predeclared gate is applied to each seed independently and the "
        "replication is GREEN only if all three seeds are GREEN. Pooled values "
        "describe the effect; they never replace a per-seed gate.",
        "",
        "## 1. Model and hardware",
        "",
        f"- Model: `{summary['model']}`",
        f"- Model revision: `{summary['model_revision']}`",
        f"- Tokenizer revision: `{summary['tokenizer_revision']}`",
        f"- Accelerator: {gpu}",
        "",
        "## 2. Per-seed Decoupled gate",
        "",
        *_table("Decoupled (primary, predeclared)", summary["per_seed"]),
        "## 3. Compositional diagnostics",
        "",
        "Mandatory diagnostics; they do not alter the Decoupled primary gate.",
        "",
        *_table("Compositional (diagnostic)", summary["per_seed_compositional"]),
        "## 4. Three-seed aggregate (Decoupled)",
        "",
        "| Quantity | seed 0 | seed 1 | seed 2 | mean | sample sd |",
        "|---|---|---|---|---|---|",
    ]
    for field, label in (
        ("delta_v", "delta V"),
        ("v_after", "V after"),
        ("delta_w", "delta W"),
        ("full_context_qa_drop_pp", "full-context QA drop (pp)"),
    ):
        values = agg[field]["values"]
        lines.append(
            f"| {label} | {values[0]:+.5f} | {values[1]:+.5f} | {values[2]:+.5f} | "
            f"{agg[field]['mean']:+.5f} | {agg[field]['sample_sd']:.5f} |"
        )
    lines += [
        "",
        "## 5. Shared-draw mean paired effect",
        "",
        f"Mean Decoupled delta V across seeds: **{shared['estimate']:+.5f}**, "
        f"95% CI [{shared['ci_95'][0]:+.5f}, {shared['ci_95'][1]:+.5f}] over "
        f"{shared['draws_effective']} shared episode-cluster draws.",
        "",
        "## 6. Gate thresholds",
        "",
        "```json",
        json.dumps(summary["gate_thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "## 7. Source artifacts",
        "",
        "| Seed | one-shot OOD result |",
        "|---|---|",
        *[f"| {seed} | `{sources[seed]}` |" for seed in (0, 1, 2)],
        "",
        "## 8. Stop",
        "",
        "This report is evidence for manual review. It authorises no extension, "
        "no additional seed, no larger model, and no combined objective.",
        "",
    ]
    if run_dirs or provenance or deviations:
        lines += operational_appendix(dict(run_dirs or {}), provenance, list(deviations))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--run-dir", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--deviation", action="append", default=[])
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    sources: dict[int, str] = {}
    payloads: dict[int, dict[str, Any]] = {}
    for item in args.result:
        if "=" not in item:
            parser.error("--result must be SEED=PATH")
        seed_text, path = item.split("=", 1)
        sources[int(seed_text)] = path
        payloads[int(seed_text)] = _load(Path(path))
    try:
        summary = build(payloads, draws=args.draws)
    except (ReportError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path, text in (
        (args.out_json, json.dumps(summary, indent=2, sort_keys=True) + "\n"),
        (
            args.out_md,
            render(
                summary,
                gpu=args.gpu,
                sources=sources,
                run_dirs={
                    int(item.split("=", 1)[0]): Path(item.split("=", 1)[1])
                    for item in args.run_dir
                },
                provenance=(
                    json.loads(args.provenance.read_text(encoding="utf-8"))
                    if args.provenance
                    else None
                ),
                deviations=args.deviation,
            ),
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            print(f"error: refusing to overwrite {path}", file=sys.stderr)
            return 1
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(f"{summary['replication_decision']}: wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
