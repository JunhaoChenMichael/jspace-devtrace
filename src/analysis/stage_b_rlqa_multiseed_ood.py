"""Fail-closed three-seed RL-QA OOD analysis for Decoupled/Compositional.

The analyzer consumes two source-isolated unified evaluator JSON files and a
pre-OOD lock.  It performs no inference.  All point estimates are recomputed
from per-item/per-episode records; uncertainty uses shared whole-episode draws.

Exact McNemar tests are reported per training seed only.  The same evaluation
episodes occur under all three policies, so pooling seed x episode observations
would be pseudoreplication and is deliberately unsupported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SEEDS = (0, 1, 2)
SFT = "sft-w-s0-k2"
RLQA = tuple(f"rl-qa-s{seed}-k2" for seed in SEEDS)
CONDITIONS = ("original", SFT, *RLQA, "workspace", "oracle")
MODEL_CONDITIONS = CONDITIONS[:5]
COMPARATORS = ("original", SFT)

SFT_ADAPTER = (
    "data/results/memory_rl_campaign_20260826/train/"
    "formal_sft-w_Qwen2.5-7B-Instruct_rank_continuous_split0_s0_"
    "beta0_k2_lq1p0_lw0p5/best-step-300"
)


class StageBRLQAMultiseedError(ValueError):
    """The sealed three-seed evaluation contract was violated."""


@dataclass(frozen=True)
class SourceContract:
    source: str
    results_path: str
    battery_path: str
    n_episodes: int
    n_items: int
    exploitable_episodes: int | None = None


@dataclass(frozen=True)
class Protocol:
    model: str
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: int
    workspace_top_k: int
    dtype: str
    admission_batch_size: int
    qa_batch_size: int
    no_harm_batch_size: int
    max_length: int
    max_new_tokens: int
    bootstrap_samples: int
    bootstrap_seed: int
    sources: tuple[SourceContract, ...]
    sft_adapter: str


FORMAL_PROTOCOL = Protocol(
    model="Qwen/Qwen2.5-7B-Instruct",
    conditions=CONDITIONS,
    seeds=SEEDS,
    budget=2,
    workspace_top_k=2,
    dtype="bfloat16",
    admission_batch_size=16,
    qa_batch_size=1,
    no_harm_batch_size=1,
    max_length=2048,
    max_new_tokens=64,
    bootstrap_samples=4000,
    bootstrap_seed=0,
    sources=(
        SourceContract(
            "decoupled",
            "data/results/results_v4f_7B-Instruct.json",
            "data/benchmarks/battery_v4_final.json",
            68,
            335,
            25,
        ),
        SourceContract(
            "compositional",
            "data/results/results_v3f_7B-Instruct.json",
            "data/benchmarks/battery_v3d.json",
            52,
            261,
        ),
    ),
    sft_adapter=SFT_ADAPTER,
)


def _error(message: str) -> None:
    raise StageBRLQAMultiseedError(message)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _error(f"{where} must be an array")
    return value


def _text(value: Any, where: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _error(f"{where} must be {'a non-empty ' if nonempty else ''}string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _error(f"{where} must be boolean")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _error(f"{where} must be finite")
    return result


def _equal(actual: Any, expected: Any, where: str) -> None:
    wrong_scalar_type = (
        isinstance(expected, bool)
        and not isinstance(actual, bool)
        or isinstance(expected, int)
        and not isinstance(expected, bool)
        and (isinstance(actual, bool) or not isinstance(actual, int))
    )
    if wrong_scalar_type or actual != expected:
        _error(f"{where} mismatch: expected {expected!r}, got {actual!r}")


def _close(actual: float, expected: float, where: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        _error(f"{where} mismatch: expected {expected!r}, got {actual!r}")


def _strict_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StageBRLQAMultiseedError(
            f"cannot read strict JSON {path}: {exc}"
        ) from exc
    return dict(_mapping(value, f"JSON root {path}"))


def _sha256(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StageBRLQAMultiseedError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise StageBRLQAMultiseedError(
            f"refusing to overwrite existing file: {path}"
        ) from exc


def _percentile(values: Sequence[float], q: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        _error("percentile distribution is empty")
    position = q * (len(finite) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _interval(estimate: float, distribution: Sequence[float]) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "ci_95": [_percentile(distribution, 0.025), _percentile(distribution, 0.975)],
        "bootstrap_samples_effective": len(distribution),
    }


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        _error("AUC vectors must be non-empty and have equal length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        _error("AUC requires both utility classes")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        rank_sum += ((start + 1 + stop) / 2.0) * sum(
            labels[order[index]] for index in range(start, stop)
        )
        start = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        _error("McNemar vectors must be non-empty and equal length")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant:
        tail = min(left_only, right_only)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(tail + 1))
            / (2**discordant),
        )
    else:
        p_value = 1.0
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "exact_two_sided_p_value": p_value,
        "n_episodes": len(left),
    }


def _draws(n: int, protocol: Protocol) -> list[list[int]]:
    if n <= 0 or protocol.bootstrap_samples <= 0:
        _error("bootstrap requires positive episode and sample counts")
    rng = random.Random(protocol.bootstrap_seed)
    return [
        [rng.randrange(n) for _ in range(n)] for _ in range(protocol.bootstrap_samples)
    ]


def _delta_interval(
    left: Sequence[bool], right: Sequence[bool], draws: Sequence[Sequence[int]]
) -> dict[str, Any]:
    estimate = sum(left) / len(left) - sum(right) / len(right)
    distribution = [
        sum(int(left[index]) - int(right[index]) for index in draw) / len(draw)
        for draw in draws
    ]
    return _interval(estimate, distribution)


def _mean_std(individual: Mapping[str, float]) -> dict[str, Any]:
    expected = [str(seed) for seed in SEEDS]
    if list(individual) != expected:
        _error(f"seed metric incomplete; expected {expected}, got {list(individual)}")
    values = [_number(individual[key], f"seed metric {key}") for key in expected]
    return {
        "status": "complete",
        "individual": dict(individual),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "n_seeds": 3,
        "ddof": 1,
    }


def _source_contract(protocol: Protocol, source: str) -> SourceContract:
    for contract in protocol.sources:
        if contract.source == source:
            return contract
    _error(f"unknown source {source!r}")


def _validate_lock(lock: Mapping[str, Any], protocol: Protocol) -> dict[str, str]:
    _equal(lock.get("schema_version"), 1, "lock.schema_version")
    _equal(lock.get("lock_status"), "locked", "lock.lock_status")
    authorization = _mapping(lock.get("authorization"), "lock.authorization")
    _equal(authorization.get("pre_ood"), True, "lock.authorization.pre_ood")
    state = _mapping(lock.get("ood_state_at_lock"), "lock.ood_state_at_lock")
    _equal(state.get("results_inspected"), False, "lock results_inspected")
    _equal(state.get("completed_artifact_exists"), False, "lock completed artifact")
    contract = _mapping(lock.get("analysis_contract"), "lock.analysis_contract")
    expected = {
        "analysis": "stage_b_rlqa_three_seed_ood_k2",
        "method": "rl-qa",
        "model": protocol.model,
        "seeds": list(protocol.seeds),
        "split_seed": 0,
        "budget": protocol.budget,
        "condition_order": list(protocol.conditions),
        "admission_batch_size": protocol.admission_batch_size,
        "qa_batch_size": protocol.qa_batch_size,
        "no_harm_batch_size": protocol.no_harm_batch_size,
        "bootstrap_samples": protocol.bootstrap_samples,
        "bootstrap_seed": protocol.bootstrap_seed,
    }
    for key, value in expected.items():
        _equal(contract.get(key), value, f"lock.analysis_contract.{key}")
    adapters = _mapping(contract.get("adapters"), "lock.analysis_contract.adapters")
    _equal(list(adapters), [SFT, *RLQA], "lock adapter order")
    _equal(adapters.get(SFT), protocol.sft_adapter, "lock SFT adapter")
    output = {SFT: protocol.sft_adapter}
    for seed, name in zip(protocol.seeds, RLQA):
        path = _text(adapters.get(name), f"lock adapter {name}", nonempty=True)
        expected_fragment = (
            f"formal_rl-qa_Qwen2.5-7B-Instruct_rank_continuous_split0_s{seed}_"
        )
        if (
            expected_fragment not in path
            or re.search(r"/best-step-[1-9][0-9]*$", path) is None
        ):
            _error(
                f"lock adapter path for {name} violates the formal checkpoint contract"
            )
        output[name] = path
    return output


def _validate_config(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: Protocol,
    adapters: Mapping[str, str],
) -> None:
    _equal(payload.get("schema_version"), 1, f"{contract.source}.schema_version")
    _equal(payload.get("condition_order"), list(protocol.conditions), "condition_order")
    config = _mapping(payload.get("config"), f"{contract.source}.config")
    expected = {
        "model": protocol.model,
        "adapters": dict(adapters),
        "rating_json": {},
        "embedding_model": None,
        "budgets": [protocol.budget],
        "workspace_top_k": protocol.workspace_top_k,
        "dtype": protocol.dtype,
        "max_length": protocol.max_length,
        "max_new_tokens": protocol.max_new_tokens,
        "admission_batch_size": protocol.admission_batch_size,
        "qa_batch_size": protocol.qa_batch_size,
        "no_harm_batch_size": protocol.no_harm_batch_size,
        "bootstrap_samples": protocol.bootstrap_samples,
        "bootstrap_seed": protocol.bootstrap_seed,
        "skip_qa": False,
        "skip_no_harm": False,
        "original_verbal_source": "precomputed_v_ref",
        "policy_input_fields": ["context", "candidate.concept"],
        "probe_visible_to_policy": False,
        "recall_model": "adapter-disabled base checkpoint",
    }
    for key, value in expected.items():
        _equal(config.get(key), value, f"{contract.source}.config.{key}")
    specs = _array(config.get("specs"), f"{contract.source}.config.specs")
    _equal(len(specs), 1, f"{contract.source} spec count")
    _equal(
        dict(_mapping(specs[0], "spec")),
        {
            "name": contract.source,
            "source": contract.source,
            "results_path": contract.results_path,
            "battery_path": contract.battery_path,
        },
        f"{contract.source} spec",
    )
    for name in ("refs", "mcnemar", "no_harm"):
        scope = _mapping(payload.get(name), f"{contract.source}.{name}")
        _equal(scope.get("skipped"), False, f"{contract.source}.{name}.skipped")


def _validate_items(
    payload: Mapping[str, Any], contract: SourceContract, protocol: Protocol
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    rows = _array(payload.get("per_item"), f"{contract.source}.per_item")
    _equal(len(rows), contract.n_items, f"{contract.source}.per_item count")
    by_episode: list[list[dict[str, Any]]] = [[] for _ in range(contract.n_episodes)]
    seen: set[str] = set()
    normalized = []
    for row_index, raw in enumerate(rows):
        where = f"{contract.source}.per_item[{row_index}]"
        row = dict(_mapping(raw, where))
        uid = _text(row.get("uid"), f"{where}.uid", nonempty=True)
        if uid in seen:
            _error(f"duplicate candidate UID {uid!r}")
        seen.add(uid)
        _equal(row.get("source"), contract.source, f"{where}.source")
        episode_index = row.get("source_episode")
        candidate_index = row.get("candidate_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            _error(f"{where}.source_episode must be integer")
        if not 0 <= episode_index < contract.n_episodes:
            _error(f"{where}.source_episode out of range")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            _error(f"{where}.candidate_index must be integer")
        if candidate_index != len(by_episode[episode_index]):
            _error(f"{where}.candidate_index is not contiguous within episode")
        episode_uid = f"{contract.source}:episode:{episode_index:06d}"
        expected_uid = f"{episode_uid}:candidate:{candidate_index:03d}"
        _equal(row.get("episode_uid"), episode_uid, f"{where}.episode_uid")
        _equal(uid, expected_uid, f"{where}.uid")
        _text(row.get("concept"), f"{where}.concept", nonempty=True)
        label = row.get("label")
        if label not in {"load_bearing", "distractor", "filler"}:
            _error(f"{where}.label invalid")
        scores = _mapping(row.get("scores"), f"{where}.scores")
        _equal(set(scores), set(protocol.conditions), f"{where}.score keys")
        for condition in protocol.conditions:
            _number(scores.get(condition), f"{where}.scores.{condition}")
        by_episode[episode_index].append(row)
        normalized.append(row)
    if any(not rows_for_episode for rows_for_episode in by_episode):
        _error(f"{contract.source} contains an empty episode")
    return normalized, by_episode


def _validate_episodes(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: Protocol,
    items_by_episode: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, list[bool]]], dict[str, list[bool]]
]:
    rows = _array(payload.get("per_episode"), f"{contract.source}.per_episode")
    _equal(len(rows), contract.n_episodes, f"{contract.source}.per_episode count")
    vectors = {name: {"qa": [], "containment": []} for name in protocol.conditions}
    no_harm = {name: [] for name in MODEL_CONDITIONS}
    episodes = []
    for index, raw in enumerate(rows):
        where = f"{contract.source}.per_episode[{index}]"
        row = dict(_mapping(raw, where))
        uid = f"{contract.source}:episode:{index:06d}"
        _equal(row.get("uid"), uid, f"{where}.uid")
        _equal(row.get("source"), contract.source, f"{where}.source")
        _equal(row.get("source_episode"), index, f"{where}.source_episode")
        candidates = items_by_episode[index]
        policies = _mapping(row.get("policies"), f"{where}.policies")
        _equal(list(policies), list(protocol.conditions), f"{where}.policy order")
        for condition in protocol.conditions:
            policy = _mapping(policies.get(condition), f"{where}.{condition}")
            selections = _mapping(
                policy.get("selections"), f"{where}.{condition}.selections"
            )
            selected = _mapping(
                selections.get(str(protocol.budget)), f"{where}.{condition}@2"
            )
            indices = _array(
                selected.get("selected_indices"), f"{where}.{condition}.indices"
            )
            if (
                len(indices) != protocol.budget
                or len(set(indices)) != protocol.budget
                or any(
                    isinstance(i, bool)
                    or not isinstance(i, int)
                    or not 0 <= i < len(candidates)
                    for i in indices
                )
            ):
                _error(f"{where}.{condition} selection violates budget")
            expected_uids = [candidates[i]["uid"] for i in indices]
            expected_concepts = [candidates[i]["concept"] for i in indices]
            _equal(
                selected.get("selected_candidate_uids"),
                expected_uids,
                f"{where}.{condition} selected UIDs",
            )
            _equal(
                selected.get("selected_concepts"),
                expected_concepts,
                f"{where}.{condition} selected concepts",
            )
            containment = any(candidates[i]["label"] == "load_bearing" for i in indices)
            _equal(
                selected.get("contains_load_bearing"),
                containment,
                f"{where}.{condition} containment",
            )
            qa = _mapping(selected.get("qa"), f"{where}.{condition}.qa")
            vectors[condition]["qa"].append(
                _boolean(qa.get("correct"), f"{where}.{condition}.correct")
            )
            vectors[condition]["containment"].append(containment)
        refs = _mapping(row.get("refs"), f"{where}.refs")
        oracle = _mapping(refs.get("oracle@2"), f"{where}.refs.oracle@2")
        _equal(
            oracle.get("correct"), vectors["oracle"]["qa"][-1], f"{where} oracle ref"
        )
        full = _mapping(
            row.get("no_harm_full_context"), f"{where}.no_harm_full_context"
        )
        _equal(list(full), list(MODEL_CONDITIONS), f"{where} no-harm order")
        for condition in MODEL_CONDITIONS:
            detail = _mapping(full.get(condition), f"{where}.noharm.{condition}")
            _text(detail.get("answer"), f"{where}.noharm.{condition}.answer")
            no_harm[condition].append(
                _boolean(detail.get("correct"), f"{where}.noharm.{condition}.correct")
            )
        episodes.append(row)
    return episodes, vectors, no_harm


def _validate_metrics(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: Protocol,
    items: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, list[bool]]],
) -> tuple[Mapping[str, Any], dict[str, float]]:
    metrics = _mapping(payload.get("metrics"), f"{contract.source}.metrics")
    by_spec = _mapping(metrics.get("by_spec"), f"{contract.source}.metrics.by_spec")
    _equal(list(by_spec), [contract.source], f"{contract.source}.metrics source")
    scope = _mapping(by_spec.get(contract.source), f"{contract.source}.metric scope")
    _equal(scope.get("n_episodes"), contract.n_episodes, "metric episode count")
    _equal(scope.get("n_items"), contract.n_items, "metric item count")
    bootstrap = _mapping(scope.get("bootstrap"), "metric bootstrap")
    _equal(
        dict(bootstrap),
        {
            "method": "episode_cluster_percentile",
            "confidence": 0.95,
            "samples_requested": protocol.bootstrap_samples,
            "seed": protocol.bootstrap_seed,
            "skipped": False,
        },
        "metric bootstrap",
    )
    conditions = _mapping(scope.get("conditions"), "metric conditions")
    _equal(list(conditions), list(protocol.conditions), "metric condition order")
    labels = [row["label"] == "load_bearing" for row in items]
    aucs = {}
    for condition in protocol.conditions:
        summary = _mapping(conditions.get(condition), f"metrics.{condition}")
        classification = _mapping(
            summary.get("classification"), f"metrics.{condition}.classification"
        )
        auc = _auc(
            labels,
            [_number(row["scores"][condition], "candidate score") for row in items],
        )
        _close(
            _number(classification.get("pooled_auc"), "recorded AUC"),
            auc,
            f"{condition} AUC",
        )
        _equal(classification.get("n_items"), contract.n_items, f"{condition} n_items")
        _equal(
            classification.get("n_episodes"),
            contract.n_episodes,
            f"{condition} n_episodes",
        )
        auc_boot = _mapping(
            classification.get("pooled_auc_bootstrap"), f"{condition} AUC bootstrap"
        )
        _close(
            _number(auc_boot.get("estimate"), "AUC estimate"),
            auc,
            f"{condition} AUC bootstrap estimate",
        )
        _equal(
            auc_boot.get("bootstrap_samples_effective"),
            protocol.bootstrap_samples,
            f"{condition} bootstrap B",
        )
        ci = _array(auc_boot.get("ci_95"), f"{condition} AUC CI")
        _equal(len(ci), 2, f"{condition} AUC CI length")
        _number(ci[0], "AUC CI lower")
        _number(ci[1], "AUC CI upper")
        selection = _mapping(summary.get("selection"), f"{condition}.selection")
        selection2 = _mapping(selection.get("2"), f"{condition}.selection.2")
        containment = sum(vectors[condition]["containment"]) / contract.n_episodes
        _close(
            _number(selection2.get("top_k_containment"), "containment"),
            containment,
            f"{condition} containment",
        )
        _equal(
            selection2.get("n_episodes"),
            contract.n_episodes,
            f"{condition} containment n",
        )
        qa = _mapping(summary.get("qa"), f"{condition}.qa")
        qa2 = _mapping(qa.get("2"), f"{condition}.qa.2")
        qa_value = sum(vectors[condition]["qa"]) / contract.n_episodes
        _close(_number(qa2.get("accuracy"), "QA accuracy"), qa_value, f"{condition} QA")
        _equal(qa2.get("n_episodes"), contract.n_episodes, f"{condition} QA n")
        aucs[condition] = auc
    paired = _array(scope.get("paired_auc_differences"), "paired AUC differences")
    expected_pairs = list(itertools.combinations(protocol.conditions, 2))
    actual_pairs = [
        (row.get("a"), row.get("b"))
        for row in map(lambda x: _mapping(x, "paired row"), paired)
    ]
    _equal(actual_pairs, expected_pairs, "paired AUC pair order")
    return scope, aucs


def _validate_no_harm_summary(
    payload: Mapping[str, Any],
    contract: SourceContract,
    vectors: Mapping[str, Sequence[bool]],
) -> None:
    root = _mapping(payload.get("no_harm"), f"{contract.source}.no_harm")
    _equal(root.get("mode"), "adapter_enabled_full_context_qa", "no-harm mode")
    _equal(
        root.get("separate_from"),
        "adapter_disabled_frozen_base_selection_recall",
        "no-harm separation",
    )
    summary = _mapping(root.get("summary"), "no-harm summary")
    by_spec = _mapping(summary.get("by_spec"), "no-harm by_spec")
    _equal(list(by_spec), [contract.source], "no-harm source keys")
    scopes = [
        _mapping(by_spec.get(contract.source), "no-harm source scope"),
        _mapping(summary.get("aggregate"), "no-harm aggregate"),
    ]
    for scope in scopes:
        conditions = _mapping(scope.get("conditions"), "no-harm conditions")
        _equal(list(conditions), list(MODEL_CONDITIONS), "no-harm condition order")
        for condition in MODEL_CONDITIONS:
            row = _mapping(conditions.get(condition), f"no-harm {condition}")
            expected = sum(vectors[condition]) / contract.n_episodes
            _close(
                _number(row.get("accuracy"), "no-harm accuracy"),
                expected,
                f"no-harm {condition}",
            )
            _equal(row.get("n_episodes"), contract.n_episodes, f"no-harm {condition} n")
        comparisons = _array(scope.get("comparisons"), "no-harm comparisons")
        normalized_comparisons = [
            _mapping(row, f"no-harm comparisons[{index}]")
            for index, row in enumerate(comparisons)
        ]
        _equal(
            [row.get("adapter") for row in normalized_comparisons],
            list(MODEL_CONDITIONS[1:]),
            "no-harm comparison order",
        )
        for row, adapter in zip(normalized_comparisons, MODEL_CONDITIONS[1:]):
            _equal(row.get("base"), "original", "no-harm base")
            exact = _mcnemar(vectors[adapter], vectors["original"])
            delta = (
                sum(vectors[adapter]) / contract.n_episodes
                - sum(vectors["original"]) / contract.n_episodes
            )
            _close(
                _number(row.get("adapter_minus_base_accuracy"), "no-harm delta"),
                delta,
                "no-harm delta",
            )
            _equal(
                row.get("base_only_correct"),
                exact["right_only_correct"],
                "no-harm base-only",
            )
            _equal(
                row.get("adapter_only_correct"),
                exact["left_only_correct"],
                "no-harm adapter-only",
            )
            _equal(row.get("discordant"), exact["discordant"], "no-harm discordant")
            _close(
                _number(row.get("exact_mcnemar_p_value"), "no-harm p"),
                exact["exact_two_sided_p_value"],
                "no-harm p",
            )
            _equal(row.get("n"), contract.n_episodes, "no-harm n")


def _validate_paired_auc_intervals(
    scope: Mapping[str, Any], region: Mapping[str, Any], protocol: Protocol
) -> None:
    """Verify evaluator AUC pairs against independently recomputed intervals."""
    rows = _array(scope.get("paired_auc_differences"), "paired AUC differences")
    indexed = {}
    for raw in rows:
        row = _mapping(raw, "paired AUC row")
        indexed[(row.get("a"), row.get("b"))] = row
    for seed, condition in zip(protocol.seeds, RLQA):
        for comparator in COMPARATORS:
            pair = (comparator, condition)
            if pair not in indexed:
                _error(f"missing paired AUC row for {condition} and {comparator}")
            row = indexed[pair]
            _equal(row.get("direction"), "a_minus_b", "paired AUC direction")
            stored = _mapping(row.get("pooled_auc_difference"), "paired pooled AUC")
            expected = region["seeds"][str(seed)]["paired"][
                f"{condition}_minus_{comparator}"
            ]["auc"]["paired_episode_bootstrap"]
            _close(
                -_number(stored.get("estimate"), "paired AUC estimate"),
                expected["estimate"],
                f"{condition} minus {comparator} AUC estimate",
            )
            ci = _array(stored.get("ci_95"), "paired AUC CI")
            _equal(len(ci), 2, "paired AUC CI length")
            oriented = [
                -_number(ci[1], "paired AUC upper"),
                -_number(ci[0], "paired AUC lower"),
            ]
            _close(oriented[0], expected["ci_95"][0], "paired AUC CI lower")
            _close(oriented[1], expected["ci_95"][1], "paired AUC CI upper")
            _equal(
                stored.get("bootstrap_samples_effective"),
                protocol.bootstrap_samples,
                "paired AUC bootstrap B",
            )


def _episode_auc(
    items_by_episode: Sequence[Sequence[Mapping[str, Any]]],
    condition: str,
    draw: Sequence[int],
) -> float:
    labels, scores = [], []
    for index in draw:
        for row in items_by_episode[index]:
            labels.append(row["label"] == "load_bearing")
            scores.append(float(row["scores"][condition]))
    return _auc(labels, scores)


def _metric_value(
    metric: str,
    condition: str,
    indices: Sequence[int],
    items_by_episode: Sequence[Sequence[Mapping[str, Any]]],
    vectors: Mapping[str, Mapping[str, Sequence[bool]]],
    no_harm: Mapping[str, Sequence[bool]],
) -> float:
    if metric == "auc":
        return _episode_auc(items_by_episode, condition, indices)
    source = no_harm if metric == "full_context_accuracy" else vectors
    key = condition if metric == "full_context_accuracy" else condition
    values = source[key] if metric == "full_context_accuracy" else source[key][metric]
    return sum(values[index] for index in indices) / len(indices)


def _region_summary(
    indices: Sequence[int],
    protocol: Protocol,
    items_by_episode: Sequence[Sequence[Mapping[str, Any]]],
    vectors: Mapping[str, Mapping[str, Sequence[bool]]],
    no_harm: Mapping[str, Sequence[bool]],
) -> dict[str, Any]:
    local_draws = _draws(len(indices), protocol)
    draws = [[indices[position] for position in draw] for draw in local_draws]
    metrics = ("auc", "qa", "containment", "full_context_accuracy")
    compared_conditions = ("original", SFT, *RLQA)
    draw_values = {
        metric: {
            condition: [
                _metric_value(
                    metric, condition, draw, items_by_episode, vectors, no_harm
                )
                for draw in draws
            ]
            for condition in compared_conditions
        }
        for metric in metrics
    }
    raw: dict[str, dict[str, float]] = {}
    shared_baselines: dict[str, dict[str, float]] = {}
    for condition in ("original", SFT, "workspace", "oracle"):
        available = metrics if condition in MODEL_CONDITIONS else metrics[:3]
        shared_baselines[condition] = {
            metric: _metric_value(
                metric, condition, indices, items_by_episode, vectors, no_harm
            )
            for metric in available
        }
    per_seed: dict[str, Any] = {}
    for seed, condition in zip(protocol.seeds, RLQA):
        raw_metrics = {
            metric: _metric_value(
                metric, condition, indices, items_by_episode, vectors, no_harm
            )
            for metric in metrics
        }
        raw[str(seed)] = raw_metrics
        comparisons = {}
        for comparator in COMPARATORS:
            metric_outputs = {}
            for metric in metrics:
                estimate = raw_metrics[metric] - shared_baselines[comparator][metric]
                distribution = [
                    left - right
                    for left, right in zip(
                        draw_values[metric][condition],
                        draw_values[metric][comparator],
                    )
                ]
                output = {
                    "direction": f"{condition}_minus_{comparator}",
                    "paired_episode_bootstrap": _interval(estimate, distribution),
                }
                if metric != "auc":
                    source = no_harm if metric == "full_context_accuracy" else vectors
                    left = (
                        source[condition]
                        if metric == "full_context_accuracy"
                        else source[condition][metric]
                    )
                    right = (
                        source[comparator]
                        if metric == "full_context_accuracy"
                        else source[comparator][metric]
                    )
                    output["exact_mcnemar"] = _mcnemar(
                        [left[index] for index in indices],
                        [right[index] for index in indices],
                    )
                metric_outputs[metric] = output
            comparisons[f"{condition}_minus_{comparator}"] = metric_outputs
        per_seed[str(seed)] = {
            "condition": condition,
            "raw": raw_metrics,
            "paired": comparisons,
        }

    aggregate_raw = {
        metric: _mean_std(
            {str(seed): raw[str(seed)][metric] for seed in protocol.seeds}
        )
        for metric in metrics
    }
    aggregate_paired = {}
    for comparator in COMPARATORS:
        comparator_output = {}
        for metric in metrics:
            individual = {
                str(seed): raw[str(seed)][metric] - shared_baselines[comparator][metric]
                for seed in protocol.seeds
            }
            summary = _mean_std(individual)
            summary["direction"] = f"mean_rl_qa_minus_{comparator}"
            distribution = []
            for draw_index in range(len(draws)):
                effects = [
                    draw_values[metric][condition][draw_index]
                    - draw_values[metric][comparator][draw_index]
                    for condition in RLQA
                ]
                distribution.append(statistics.fmean(effects))
            summary["shared_episode_bootstrap"] = _interval(
                summary["mean"], distribution
            )
            if metric != "auc":
                summary["pooled_mcnemar"] = {
                    "status": "not-applicable",
                    "reason": "the same episodes are repeated across training seeds",
                }
            comparator_output[metric] = summary
        aggregate_paired[f"mean_rl_qa_minus_{comparator}"] = comparator_output
    return {
        "n_episodes": len(indices),
        "shared_baselines": shared_baselines,
        "seeds": per_seed,
        "aggregate": {"rl_qa": aggregate_raw, "paired": aggregate_paired},
    }


def _analyze_source(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: Protocol,
    adapters: Mapping[str, str],
) -> dict[str, Any]:
    _validate_config(payload, contract, protocol, adapters)
    items, items_by_episode = _validate_items(payload, contract, protocol)
    episodes, vectors, no_harm = _validate_episodes(
        payload, contract, protocol, items_by_episode
    )
    scope, aucs = _validate_metrics(payload, contract, protocol, items, vectors)
    _validate_no_harm_summary(payload, contract, no_harm)
    all_indices = list(range(contract.n_episodes))
    result = _region_summary(all_indices, protocol, items_by_episode, vectors, no_harm)
    for condition, auc in aucs.items():
        if condition in RLQA:
            seed = str(protocol.seeds[RLQA.index(condition)])
            observed = result["seeds"][seed]["raw"]["auc"]
        else:
            observed = result["shared_baselines"][condition]["auc"]
        _close(observed, auc, f"{condition} raw AUC")
    _validate_paired_auc_intervals(scope, result, protocol)
    result.update({"source": contract.source, "n_items": contract.n_items})
    if contract.exploitable_episodes is not None:
        exploitable = [
            index
            for index, episode in enumerate(episodes)
            if _boolean(
                _mapping(episode["refs"]["oracle@2"], "oracle ref").get("correct"),
                "oracle correct",
            )
        ]
        _equal(
            len(exploitable),
            contract.exploitable_episodes,
            "Decoupled exploitable count",
        )
        subset = _region_summary(
            exploitable, protocol, items_by_episode, vectors, no_harm
        )
        subset["definition"] = "refs['oracle@2'].correct == true"
        subset["episode_uids"] = [episodes[index]["uid"] for index in exploitable]
        result["exploitable_subset"] = subset
    return result


def build_analysis(
    decoupled: Mapping[str, Any],
    compositional: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    protocol: Protocol = FORMAL_PROTOCOL,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapters = _validate_lock(lock, protocol)
    payloads = {"decoupled": decoupled, "compositional": compositional}
    sources = {
        contract.source: _analyze_source(
            payloads[contract.source], contract, protocol, adapters
        )
        for contract in protocol.sources
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "stage_b_rlqa_three_seed_ood_k2",
        "status": "complete",
        "lock": {
            "status": "locked",
            "scope": "pre-OOD ID-only",
            "adapters": dict(adapters),
        },
        "protocol": {
            "condition_order": list(protocol.conditions),
            "seeds": list(protocol.seeds),
            "budget": protocol.budget,
            "workspace_top_k": protocol.workspace_top_k,
            "dtype": protocol.dtype,
            "admission_batch_size": protocol.admission_batch_size,
            "qa_batch_size": protocol.qa_batch_size,
            "no_harm_batch_size": protocol.no_harm_batch_size,
            "max_length": protocol.max_length,
            "max_new_tokens": protocol.max_new_tokens,
            "bootstrap": {
                "method": "shared paired episode-cluster percentile",
                "samples": protocol.bootstrap_samples,
                "seed": protocol.bootstrap_seed,
                "confidence": 0.95,
            },
            "seed_dispersion": "sample standard deviation, ddof=1",
            "pooled_seed_episode_tests": False,
        },
        "inputs": dict(provenance or {}),
        "sources": sources,
    }


def analyze_files(
    decoupled: Path,
    compositional: Path,
    lock: Path,
    *,
    protocol: Protocol = FORMAL_PROTOCOL,
) -> dict[str, Any]:
    provenance = {
        "decoupled": {"path": str(decoupled), "sha256": _sha256(decoupled)},
        "compositional": {"path": str(compositional), "sha256": _sha256(compositional)},
        "pre_ood_lock": {"path": str(lock), "sha256": _sha256(lock)},
    }
    return build_analysis(
        _strict_load(decoupled),
        _strict_load(compositional),
        _strict_load(lock),
        protocol=protocol,
        provenance=provenance,
    )


def _difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type differs"
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return f"{path}: keys differ"
        for key in sorted(left):
            result = _difference(left[key], right[key], f"{path}.{key}")
            if result:
                return result
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length differs"
        for index, (a, b) in enumerate(zip(left, right)):
            result = _difference(a, b, f"{path}[{index}]")
            if result:
                return result
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def validate_files(
    decoupled: Path,
    compositional: Path,
    lock: Path,
    analysis: Path,
    *,
    protocol: Protocol = FORMAL_PROTOCOL,
) -> dict[str, Any]:
    try:
        expected = analyze_files(decoupled, compositional, lock, protocol=protocol)
        actual = _strict_load(analysis)
        difference = _difference(actual, expected)
        return {
            "schema_version": 1,
            "validator": "stage_b_rlqa_multiseed_ood",
            "status": "pass" if difference is None else "fail",
            "errors": [] if difference is None else [difference],
            "analysis_path": str(analysis),
            "analysis_sha256": _sha256(analysis),
        }
    except StageBRLQAMultiseedError as exc:
        return {
            "schema_version": 1,
            "validator": "stage_b_rlqa_multiseed_ood",
            "status": "fail",
            "errors": [str(exc)],
            "analysis_path": str(analysis),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--decoupled", type=Path, required=True)
        command.add_argument("--compositional", type=Path, required=True)
        command.add_argument("--lock", type=Path, required=True)
        if name == "validate":
            command.add_argument("--analysis", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            output = analyze_files(args.decoupled, args.compositional, args.lock)
        else:
            output = validate_files(
                args.decoupled, args.compositional, args.lock, args.analysis
            )
        _write_once(args.out, output)
    except StageBRLQAMultiseedError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
