"""Aggregate seeded memory-RL conditions and apply the predeclared gates.

This script consumes one unified ``evaluate_memory_rl.py`` JSON. Adapter
condition families are selected with shell-style glob patterns so names such as
``rl-w_seed0`` and ``rl-w_seed1`` can be summarized together.

Example::

    python src/analysis/memory_rl_gates.py data/results/memory_rl_eval.json \
      --sft 'sft-w_seed*' --rl-w 'rl-w_seed*' \
      --rl-qa 'rl-qa_seed*' --hybrid 'hybrid_seed*' \
      --source decoupled --budget 2 --out data/results/memory_rl_gates.json
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


RESERVED_CONDITIONS = frozenset(
    {"original", "workspace", "oracle", "rating", "embedding"}
)
FAMILY_ORDER = ("sft", "rl_w", "rl_qa", "hybrid")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nested(root: Any, *keys: str) -> Any:
    value = root
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _metric_summary(individual: Mapping[str, float | None]) -> dict[str, Any]:
    values = [value for value in individual.values() if value is not None]
    missing = [name for name, value in individual.items() if value is None]
    return {
        "individual": dict(individual),
        "mean": statistics.fmean(values) if values else None,
        # Sample standard deviation is undefined for a single seed. Reporting
        # null prevents a one-seed development result from looking variance-free.
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        "n": len(values),
        "n_matched": len(individual),
        "missing_conditions": missing,
    }


def _metric_complete(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("n_matched", 0) > 0
        and summary.get("n") == summary.get("n_matched")
        and _finite_number(summary.get("mean")) is not None
    )


def _condition_auc(condition: Any) -> float | None:
    value = _nested(condition, "classification", "pooled_auc")
    count = _nested(condition, "classification", "n_episodes")
    if count == 0:
        return None
    return _finite_number(value)


def _condition_qa(condition: Any, budget: int) -> float | None:
    value = _nested(condition, "qa", str(budget), "accuracy")
    count = _nested(condition, "qa", str(budget), "n_episodes")
    if count == 0:
        return None
    return _finite_number(value)


def _no_harm_deltas(payload: Mapping[str, Any], source: str) -> dict[str, float]:
    comparisons = _nested(
        payload, "no_harm", "summary", "by_spec", source, "comparisons"
    )
    if not isinstance(comparisons, list):
        return {}
    result: dict[str, float] = {}
    for row in comparisons:
        if not isinstance(row, Mapping) or row.get("base") != "original":
            continue
        adapter = row.get("adapter")
        value = _finite_number(row.get("adapter_minus_base_accuracy"))
        if isinstance(adapter, str) and adapter and value is not None:
            result[adapter] = value
    return result


def _match_conditions(
    eligible: Sequence[str], patterns: Sequence[str]
) -> list[str]:
    return sorted(
        name
        for name in eligible
        if any(fnmatchcase(name, pattern) for pattern in patterns)
    )


def _family_summary(
    names: Sequence[str],
    patterns: Sequence[str],
    conditions: Mapping[str, Any],
    no_harm: Mapping[str, float],
    budget: int,
) -> dict[str, Any]:
    auc = {name: _condition_auc(conditions.get(name)) for name in names}
    qa = {name: _condition_qa(conditions.get(name), budget) for name in names}
    harm = {name: _finite_number(no_harm.get(name)) for name in names}
    return {
        "patterns": list(patterns),
        "matched_conditions": list(names),
        "pooled_auc": _metric_summary(auc),
        "qa": _metric_summary(qa),
        "no_harm_adapter_minus_base_accuracy": _metric_summary(harm),
    }


def _insufficient_gate(rule: str, missing: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "insufficient-data",
        "rule": rule,
        "missing": list(missing),
    }


def _gate_a(families: Mapping[str, Any]) -> dict[str, Any]:
    rule = "RL-W mean pooled AUC minus SFT-W mean pooled AUC"
    sft = families["sft"]["pooled_auc"]
    rl_w = families["rl_w"]["pooled_auc"]
    missing = [
        name
        for name, summary in (("sft.pooled_auc", sft), ("rl_w.pooled_auc", rl_w))
        if not _metric_complete(summary)
    ]
    if missing:
        return _insufficient_gate(rule, missing)
    delta = float(rl_w["mean"] - sft["mean"])
    if delta + 1e-12 >= 0.03:
        status = "pass"
    elif abs(delta) < 0.03:
        status = "tie"
    else:
        status = "worse"
    return {
        "status": status,
        "rule": rule,
        "delta": delta,
        "pass_threshold": 0.03,
        "tie_rule": "absolute delta < 0.03",
        "rl_w_mean": rl_w["mean"],
        "sft_mean": sft["mean"],
    }


def _gate_b(families: Mapping[str, Any], original: Mapping[str, Any]) -> dict[str, Any]:
    rule = "RL-QA mean QA minus original QA >= 0.05"
    rl_qa = families["rl_qa"]["qa"]
    original_qa = original["qa"]
    missing = [
        name
        for name, summary in (("rl_qa.qa", rl_qa), ("original.qa", original_qa))
        if not _metric_complete(summary)
    ]
    if missing:
        return _insufficient_gate(rule, missing)
    delta = float(rl_qa["mean"] - original_qa["mean"])
    return {
        "status": "pass" if delta + 1e-12 >= 0.05 else "fail",
        "rule": rule,
        "delta": delta,
        "pass_threshold": 0.05,
        "rl_qa_mean": rl_qa["mean"],
        "original_mean": original_qa["mean"],
    }


def _gate_c(families: Mapping[str, Any]) -> dict[str, Any]:
    rule = "hybrid mean QA is strictly greater than both RL-W and RL-QA mean QA"
    hybrid = families["hybrid"]["qa"]
    rl_w = families["rl_w"]["qa"]
    rl_qa = families["rl_qa"]["qa"]
    missing = [
        name
        for name, summary in (
            ("hybrid.qa", hybrid),
            ("rl_w.qa", rl_w),
            ("rl_qa.qa", rl_qa),
        )
        if not _metric_complete(summary)
    ]
    if missing:
        return _insufficient_gate(rule, missing)
    beats_rl_w = hybrid["mean"] > rl_w["mean"]
    beats_rl_qa = hybrid["mean"] > rl_qa["mean"]
    return {
        "status": "pass" if beats_rl_w and beats_rl_qa else "fail",
        "rule": rule,
        "hybrid_mean": hybrid["mean"],
        "rl_w_mean": rl_w["mean"],
        "rl_qa_mean": rl_qa["mean"],
        "hybrid_minus_rl_w": hybrid["mean"] - rl_w["mean"],
        "hybrid_minus_rl_qa": hybrid["mean"] - rl_qa["mean"],
        "beats_rl_w": beats_rl_w,
        "beats_rl_qa": beats_rl_qa,
    }


def analyze_gates(
    payload: Mapping[str, Any],
    *,
    sft_patterns: Sequence[str] = ("sft*",),
    rl_w_patterns: Sequence[str] = ("rl-w*",),
    rl_qa_patterns: Sequence[str] = ("rl-qa*",),
    hybrid_patterns: Sequence[str] = ("*hybrid*",),
    source: str = "decoupled",
    budget: int = 2,
) -> dict[str, Any]:
    """Summarize seed families and evaluate Gates A/B/C.

    A gate requires every matched condition to contain its required metric. A
    partially evaluated seed family therefore becomes ``insufficient-data``
    rather than silently averaging only the successful runs.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("evaluation JSON root must be an object")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")

    scope = _nested(payload, "metrics", "by_spec", source)
    source_found = isinstance(scope, Mapping)
    conditions = _mapping(_nested(scope, "conditions")) if source_found else {}

    configured_adapters = _nested(payload, "config", "adapters")
    if isinstance(configured_adapters, Mapping):
        eligible = sorted(name for name in configured_adapters if name in conditions)
    else:
        eligible = sorted(name for name in conditions if name not in RESERVED_CONDITIONS)

    patterns = {
        "sft": tuple(sft_patterns),
        "rl_w": tuple(rl_w_patterns),
        "rl_qa": tuple(rl_qa_patterns),
        "hybrid": tuple(hybrid_patterns),
    }
    matches = {
        family: _match_conditions(eligible, family_patterns)
        for family, family_patterns in patterns.items()
    }
    owners: dict[str, list[str]] = {}
    for family, names in matches.items():
        for name in names:
            owners.setdefault(name, []).append(family)
    overlaps = {name: family_names for name, family_names in owners.items() if len(family_names) > 1}
    if overlaps:
        raise ValueError(f"condition globs overlap across families: {overlaps}")

    no_harm = _no_harm_deltas(payload, source)
    families = {
        family: _family_summary(
            matches[family], patterns[family], conditions, no_harm, budget
        )
        for family in FAMILY_ORDER
    }
    original = _family_summary(
        ["original"] if "original" in conditions else [],
        ("original",),
        conditions,
        {},
        budget,
    )
    # A base-vs-base no-harm delta is not reported; this field is meaningful
    # only for adapter families.
    original.pop("no_harm_adapter_minus_base_accuracy")

    gates = {
        "A": _gate_a(families),
        "B": _gate_b(families, original),
        "C": _gate_c(families),
    }
    if not source_found:
        for gate in gates.values():
            gate["source_missing"] = source

    return {
        "schema_version": 1,
        "source": source,
        "source_found": source_found,
        "budget": budget,
        "families": families,
        "baselines": {"original": original},
        "gates": gates,
    }


def _patterns(values: Sequence[str] | None, default: str) -> tuple[str, ...]:
    selected = tuple(values or (default,))
    if any(not value for value in selected):
        raise ValueError("condition glob patterns must be non-empty")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", help="unified evaluate_memory_rl JSON")
    parser.add_argument("--sft", action="append", help="SFT condition glob; repeatable")
    parser.add_argument("--rl-w", action="append", help="RL-W condition glob; repeatable")
    parser.add_argument("--rl-qa", action="append", help="RL-QA condition glob; repeatable")
    parser.add_argument("--hybrid", action="append", help="hybrid condition glob; repeatable")
    parser.add_argument("--source", default="decoupled")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    input_path = Path(args.evaluation_json)
    try:
        with input_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        report = analyze_gates(
            payload,
            sft_patterns=_patterns(args.sft, "sft*"),
            rl_w_patterns=_patterns(args.rl_w, "rl-w*"),
            rl_qa_patterns=_patterns(args.rl_qa, "rl-qa*"),
            hybrid_patterns=_patterns(args.hybrid, "*hybrid*"),
            source=args.source,
            budget=args.budget,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    report["input"] = str(input_path)

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
