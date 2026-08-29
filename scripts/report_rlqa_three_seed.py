#!/usr/bin/env python3
"""Aggregate the three RL-QA seeds into the Track A replication report.

Success criteria come from docs/H100_NEXT_CAMPAIGNS.md, Track A:

* primary   - every seed's Decoupled QA delta versus Original is positive, and
              the three-seed mean paired effect is positive with a 95% episode
              bootstrap CI excluding zero;
* admission - every seed's Decoupled admission-AUC delta versus Original is
              positive;
* no harm   - full-context QA drops at most 2 percentage points and fresh W_rr
              drops at most 0.03, per seed.

Seed x episode observations are never pooled: the same evaluation episodes are
seen by all three policies, so pooling would be pseudoreplication. The mean
paired effect instead uses SHARED episode draws, re-scoring every seed on the
same resample before averaging.
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
from report_metacog_three_seed import operational_appendix  # noqa: E402

REPORT_SCHEMA = "rlqa-three-seed-report/v1"
SEEDS = (0, 1, 2)
GATE = {"max_qa_drop_pp": 2.0, "max_w_rr_drop": 0.03}
DEFAULT_DRAWS = 4000


class ReportError(RuntimeError):
    pass


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"missing or unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportError(f"{label} must be a JSON object: {path}")
    return value


def _condition(seed: int) -> str:
    return f"rl-qa-s{seed}"


def _scope(payload: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    scope = payload.get("metrics", {}).get("by_spec", {}).get(source)
    if not isinstance(scope, Mapping):
        raise ReportError(f"evaluation has no metrics for source {source!r}")
    return scope


def _qa_accuracy(scope: Mapping[str, Any], condition: str, budget: int) -> float:
    entry = scope["conditions"].get(condition)
    if entry is None:
        raise ReportError(f"evaluation has no condition {condition!r}")
    value = entry.get("qa", {}).get(str(budget), {}).get("accuracy")
    if value is None:
        raise ReportError(f"condition {condition!r} has no QA accuracy at budget {budget}")
    return float(value)


def _paired_auc(scope: Mapping[str, Any], condition: str) -> dict[str, Any]:
    for row in scope.get("paired_auc_differences", []):
        if row.get("a") == condition and row.get("b") == "original":
            return row["pooled_auc_difference"]
        if row.get("a") == "original" and row.get("b") == condition:
            flipped = row["pooled_auc_difference"]
            low, high = flipped["ci_95"]
            return {
                "estimate": -float(flipped["estimate"]),
                "ci_95": [-float(high), -float(low)],
                "bootstrap_samples_effective": flipped.get("bootstrap_samples_effective"),
            }
    raise ReportError(f"no paired admission-AUC difference for {condition!r} vs original")


def _mcnemar(payload: Mapping[str, Any], source: str, condition: str, budget: int) -> dict[str, Any] | None:
    scope = payload.get("mcnemar", {}).get("by_spec", {}).get(source, {})
    # The evaluator keys its comparisons by budget, and orients each pair as it
    # was generated; normalise both so a seed is found either way round.
    rows = scope.get(str(budget))
    if rows is None:
        rows = scope.get("comparisons", [])
    for row in rows:
        if {row.get("a"), row.get("b")} != {condition, "original"}:
            continue
        if row.get("a") == condition:
            return {**row, "adapter_only_correct": row.get("a_only"), "original_only_correct": row.get("b_only")}
        return {**row, "adapter_only_correct": row.get("b_only"), "original_only_correct": row.get("a_only")}
    return None


def _no_harm(payload: Mapping[str, Any], condition: str, source: str) -> dict[str, Any] | None:
    """Adapter-enabled full-context QA drop, reported per source."""

    no_harm = payload.get("no_harm", {})
    if no_harm.get("skipped"):
        return None
    scope = no_harm.get("summary", {}).get("by_spec", {}).get(source, {})
    for row in scope.get("comparisons", []):
        if row.get("adapter") == condition:
            value = row.get("adapter_minus_base_accuracy")
            return {
                "full_context_qa_drop_pp": None if value is None else -float(value) * 100.0,
                "full_context_qa_original": scope.get("conditions", {})
                .get("original", {})
                .get("accuracy"),
                "full_context_qa_adapter": scope.get("conditions", {})
                .get(condition, {})
                .get("accuracy"),
                "full_context_exact_mcnemar_p_value": row.get("exact_mcnemar_p_value"),
            }
    return None


def _episode_qa(payload: Mapping[str, Any], source: str, condition: str, budget: int) -> dict[str, bool]:
    correct: dict[str, bool] = {}
    for episode in payload.get("per_episode", []):
        if episode.get("source") != source:
            continue
        selection = (
            episode.get("policies", {}).get(condition, {}).get("selections", {}).get(str(budget), {})
        )
        qa = selection.get("qa")
        if isinstance(qa, Mapping) and "correct" in qa:
            correct[episode["uid"]] = bool(qa["correct"])
    return correct


def shared_draw_mean_qa_effect(
    payload: Mapping[str, Any], source: str, budget: int, *, draws: int = DEFAULT_DRAWS, seed: int = 0
) -> dict[str, Any]:
    base = _episode_qa(payload, source, "original", budget)
    per_seed = {s: _episode_qa(payload, source, _condition(s), budget) for s in SEEDS}
    episodes = sorted(set(base) & set.intersection(*(set(v) for v in per_seed.values())))
    if not episodes:
        raise ReportError("no shared evaluation episodes across original and the three seeds")
    point = statistics.fmean(
        statistics.fmean(per_seed[s][uid] for uid in episodes)
        - statistics.fmean(base[uid] for uid in episodes)
        for s in SEEDS
    )
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        resample = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        base_acc = statistics.fmean(base[uid] for uid in resample)
        estimates.append(
            statistics.fmean(
                statistics.fmean(per_seed[s][uid] for uid in resample) - base_acc for s in SEEDS
            )
        )
    return {
        "estimate_pp": point * 100.0,
        "ci_95_pp": [_percentile(estimates, 0.025) * 100.0, _percentile(estimates, 0.975) * 100.0],
        "draws": len(estimates),
        "n_episodes": len(episodes),
        "unit": "episode_cluster",
        "shared_draws_across_seeds": True,
    }


def workspace_no_harm(base_path: Path, seed_paths: Mapping[int, Path]) -> dict[int, dict[str, Any]]:
    """Pooled W_rr AUC before/after the adapter, from measure.py artifacts."""

    def pooled(path: Path) -> float:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return auc(
            [float(row["W_rr"]) for row in rows],
            [int(row["label"] == "load_bearing") for row in rows],
        )

    before = pooled(base_path)
    out = {}
    for seed, path in seed_paths.items():
        after = pooled(path)
        out[seed] = {"w_rr_before": before, "w_rr_after": after, "w_rr_drop": before - after}
    return out


def build(
    qa_payload: Mapping[str, Any],
    *,
    source: str = "decoupled",
    budget: int = 2,
    draws: int = DEFAULT_DRAWS,
    workspace: Mapping[int, Mapping[str, Any]] | None = None,
    compositional: str = "compositional",
) -> dict[str, Any]:
    scope = _scope(qa_payload, source)
    original_qa = _qa_accuracy(scope, "original", budget)
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        condition = _condition(seed)
        seed_qa = _qa_accuracy(scope, condition, budget)
        admission = _paired_auc(scope, condition)
        harm = _no_harm(qa_payload, condition, source) or {}
        qa_drop = harm.get("full_context_qa_drop_pp")
        ws = (workspace or {}).get(seed, {})
        checks = {
            "decoupled_qa_delta_positive": (seed_qa - original_qa) > 0,
            "admission_auc_delta_positive": float(admission["estimate"]) > 0,
            "full_context_qa_drop_at_most_2pp": qa_drop is None or qa_drop <= GATE["max_qa_drop_pp"],
            "w_rr_drop_at_most_0.03": (
                "w_rr_drop" not in ws or float(ws["w_rr_drop"]) <= GATE["max_w_rr_drop"]
            ),
        }
        per_seed[seed] = {
            "qa_original": original_qa,
            "qa_seed": seed_qa,
            "qa_delta_pp": (seed_qa - original_qa) * 100.0,
            "admission_auc_delta": float(admission["estimate"]),
            "admission_auc_delta_ci_95": admission.get("ci_95"),
            "exact_mcnemar": _mcnemar(qa_payload, source, condition, budget),
            **harm,
            **ws,
            "checks": checks,
        }

    mean_effect = shared_draw_mean_qa_effect(qa_payload, source, budget, draws=draws)

    def stats(field: str) -> dict[str, Any]:
        values = [per_seed[s][field] for s in SEEDS if per_seed[s].get(field) is not None]
        if len(values) != len(SEEDS):
            return {"values": values, "mean": None, "sample_sd": None, "incomplete": True}
        return {"values": values, "mean": statistics.fmean(values), "sample_sd": statistics.stdev(values)}

    criteria = {
        "primary_every_seed_qa_delta_positive": all(
            per_seed[s]["checks"]["decoupled_qa_delta_positive"] for s in SEEDS
        ),
        "primary_mean_effect_ci_excludes_zero": mean_effect["ci_95_pp"][0] > 0,
        "admission_every_seed_auc_delta_positive": all(
            per_seed[s]["checks"]["admission_auc_delta_positive"] for s in SEEDS
        ),
        "no_harm_full_context_qa": all(
            per_seed[s]["checks"]["full_context_qa_drop_at_most_2pp"] for s in SEEDS
        ),
        "no_harm_workspace_w_rr": all(per_seed[s]["checks"]["w_rr_drop_at_most_0.03"] for s in SEEDS),
    }
    summary: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "source": source,
        "budget": budget,
        "per_seed": per_seed,
        "aggregate": {
            field: stats(field)
            for field in ("qa_delta_pp", "admission_auc_delta", "full_context_qa_drop_pp", "w_rr_drop")
        },
        "shared_draw_mean_qa_effect": mean_effect,
        "success_criteria": criteria,
        "decision": "PASS" if all(criteria.values()) else "NOT_PASS",
    }
    try:
        comp_scope = _scope(qa_payload, compositional)
        comp_original = _qa_accuracy(comp_scope, "original", budget)
        summary["compositional_diagnostic"] = {
            f"seed{seed}": {
                "qa_delta_pp": (_qa_accuracy(comp_scope, _condition(seed), budget) - comp_original) * 100.0,
                "admission_auc_delta": float(_paired_auc(comp_scope, _condition(seed))["estimate"]),
            }
            for seed in SEEDS
        }
    except ReportError:
        summary["compositional_diagnostic"] = None
    return summary


def render(
    summary: Mapping[str, Any],
    *,
    gpu: str,
    recipe: Mapping[str, Any] | None,
    lock: Mapping[str, Any] | None,
    run_dirs: Mapping[int, Path] | None = None,
    provenance: Mapping[str, Any] | None = None,
    deviations: Sequence[str] = (),
) -> str:
    agg = summary["aggregate"]
    effect = summary["shared_draw_mean_qa_effect"]
    lines = [
        f"# Qwen3-8B RL-QA: three-seed {gpu} replication",
        "",
        f"Report schema: `{summary['schema_version']}`. Decision: **{summary['decision']}**.",
        "",
        f"Primary source `{summary['source']}`, budget {summary['budget']}. Seed x episode "
        "observations are never pooled; the mean paired effect uses shared episode draws.",
        "",
        "## 1. Success criteria",
        "",
        "| Criterion | Result |",
        "|---|---|",
    ]
    labels = {
        "primary_every_seed_qa_delta_positive": "Primary: every seed's Decoupled QA delta vs Original is positive",
        "primary_mean_effect_ci_excludes_zero": "Primary: three-seed mean paired effect positive, 95% CI excludes zero",
        "admission_every_seed_auc_delta_positive": "Admission: every seed's Decoupled admission-AUC delta positive",
        "no_harm_full_context_qa": "No harm: full-context QA drop <= 2 pp per seed",
        "no_harm_workspace_w_rr": "No harm: fresh W_rr drop <= 0.03 per seed",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {'PASS' if summary['success_criteria'][key] else 'FAIL'} |")
    lines += [
        "",
        "## 2. Per-seed Decoupled results",
        "",
        "| Seed | QA (original) | QA (RL-QA) | QA delta (pp) | admission AUC delta | CI 95% | "
        "exact McNemar p | full-context QA drop (pp) | W_rr drop |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in SEEDS:
        row = summary["per_seed"][seed]
        ci = row.get("admission_auc_delta_ci_95") or [None, None]
        mcn = row.get("exact_mcnemar") or {}
        lines.append(
            f"| {seed} | {row['qa_original']:.4f} | {row['qa_seed']:.4f} | {row['qa_delta_pp']:+.2f} | "
            f"{row['admission_auc_delta']:+.5f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{mcn.get('p_value', float('nan')):.4g} | "
            f"{'n/a' if row.get('full_context_qa_drop_pp') is None else format(row['full_context_qa_drop_pp'], '+.2f')} | "
            f"{'n/a' if row.get('w_rr_drop') is None else format(row['w_rr_drop'], '+.5f')} |"
        )
    lines += [
        "",
        "## 3. Three-seed aggregate",
        "",
        "| Quantity | seed 0 | seed 1 | seed 2 | mean | sample sd |",
        "|---|---|---|---|---|---|",
    ]
    for field, label in (
        ("qa_delta_pp", "Decoupled QA delta (pp)"),
        ("admission_auc_delta", "admission AUC delta"),
        ("full_context_qa_drop_pp", "full-context QA drop (pp)"),
        ("w_rr_drop", "W_rr drop"),
    ):
        block = agg[field]
        if block.get("incomplete") or block["mean"] is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a |")
            continue
        v = block["values"]
        lines.append(
            f"| {label} | {v[0]:+.5f} | {v[1]:+.5f} | {v[2]:+.5f} | "
            f"{block['mean']:+.5f} | {block['sample_sd']:.5f} |"
        )
    lines += [
        "",
        "## 4. Shared-draw mean paired QA effect",
        "",
        f"Mean Decoupled QA gain across seeds: **{effect['estimate_pp']:+.2f} pp**, "
        f"95% CI [{effect['ci_95_pp'][0]:+.2f}, {effect['ci_95_pp'][1]:+.2f}] pp over "
        f"{effect['draws']} shared episode-cluster draws on {effect['n_episodes']} episodes.",
        "",
    ]
    if summary.get("compositional_diagnostic"):
        lines += [
            "## 5. Compositional diagnostics",
            "",
            "| Seed | QA delta (pp) | admission AUC delta |",
            "|---|---|---|",
        ]
        for seed in SEEDS:
            row = summary["compositional_diagnostic"][f"seed{seed}"]
            lines.append(f"| {seed} | {row['qa_delta_pp']:+.2f} | {row['admission_auc_delta']:+.5f} |")
        lines.append("")
    if recipe:
        lines += [
            "## 6. Frozen recipe",
            "",
            "```json",
            json.dumps(recipe.get("recipe", recipe), indent=2, sort_keys=True),
            "```",
            "",
            "Temperature decision:",
            "",
            "```json",
            json.dumps(recipe.get("temperature_decision", {}), indent=2, sort_keys=True),
            "```",
            "",
        ]
    if lock:
        lines += [
            "## 7. Pre-OOD checkpoint lock",
            "",
            "| Seed | selected step | checkpoint tree SHA-256 |",
            "|---|---|---|",
            *[
                f"| {entry['seed']} | {entry['step']} | `{entry['checkpoint_tree_sha256']}` |"
                for entry in lock.get("seeds", [])
            ],
            "",
            f"Lock manifest self-hash: `{lock.get('manifest_sha256')}`. "
            f"Shared split manifest: `{lock.get('shared_split_manifest_sha256')}`.",
            "",
        ]
    lines += [
        "## 8. Scope and stop",
        "",
        "This allocation trained RL-QA only. RL-W, Hybrid, Soft Binary, Pairwise, "
        "Listwise, larger models and combined RL+metacognitive objectives were not run. "
        "No SFT-W baseline exists at this scale, so the SFT-W comparison named in the "
        "controlling plan is reported as unavailable rather than silently promoted or "
        "replaced. This report is evidence for manual review and authorises nothing further.",
        "",
    ]
    if run_dirs or provenance or deviations:
        lines += operational_appendix(dict(run_dirs or {}), provenance, list(deviations))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="unified evaluator JSON with QA")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--workspace-base", type=Path)
    parser.add_argument("--workspace-seed", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--run-dir", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--deviation", action="append", default=[])
    parser.add_argument("--source", default="decoupled")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        qa_payload = _load(args.qa, "unified evaluation")
        workspace = None
        if args.workspace_base and args.workspace_seed:
            seed_paths = {
                int(item.split("=", 1)[0]): Path(item.split("=", 1)[1]) for item in args.workspace_seed
            }
            workspace = workspace_no_harm(args.workspace_base, seed_paths)
        summary = build(
            qa_payload, source=args.source, budget=args.budget, draws=args.draws, workspace=workspace
        )
    except (ReportError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = render(
        summary,
        gpu=args.gpu,
        recipe=_load(args.recipe, "recipe freeze") if args.recipe else None,
        lock=_load(args.lock, "ID lock") if args.lock else None,
        run_dirs={int(i.split("=", 1)[0]): Path(i.split("=", 1)[1]) for i in args.run_dir},
        provenance=_load(args.provenance, "provenance") if args.provenance else None,
        deviations=args.deviation,
    )
    for path, content in ((args.out_json, json.dumps(summary, indent=2, sort_keys=True) + "\n"), (args.out_md, text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            print(f"error: refusing to overwrite {path}", file=sys.stderr)
            return 1
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    print(f"{summary['decision']}: wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
