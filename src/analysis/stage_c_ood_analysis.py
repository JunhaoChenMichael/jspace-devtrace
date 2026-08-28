"""Fail-closed Stage-C OOD analysis for the locked seed-0 Hybrid policy.

The two inputs must be source-separated outputs from
``experiments/evaluate_memory_rl.py`` and must each contain the complete seven
condition comparison.  This module deliberately performs no model inference.
It validates the sealed evaluation protocol, recomputes episode-level QA and
containment from ``per_episode``, and audits pooled AUC against ``per_item``.

Examples::

    python src/analysis/stage_c_ood_analysis.py analyze \
      --decoupled data/results/.../decoupled.json \
      --compositional data/results/.../compositional.json \
      --lock data/results/.../stage_c_pre_ood_tiebreak_lock.json \
      --out data/results/.../stage_c_ood_analysis.json

    python src/analysis/stage_c_ood_analysis.py validate \
      --decoupled data/results/.../decoupled.json \
      --compositional data/results/.../compositional.json \
      --lock data/results/.../stage_c_pre_ood_tiebreak_lock.json \
      --analysis data/results/.../stage_c_ood_analysis.json \
      --out data/results/.../stage_c_ood_analysis_validation.json

Both commands create their output with exclusive-create semantics.  Existing
artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CONDITION_ORDER = (
    "original",
    "sft-w-s0-k2",
    "rl-w-s0-k2",
    "rl-qa-s0-k2",
    "rl-hybrid-s0-k2-lw0p25",
    "workspace",
    "oracle",
)
HYBRID = "rl-hybrid-s0-k2-lw0p25"
COMPARATORS = ("rl-qa-s0-k2", "sft-w-s0-k2", "original")
NO_HARM_CONDITIONS = CONDITION_ORDER[:5]

SFT_ADAPTER = (
    "data/results/memory_rl_campaign_20260826/train/"
    "formal_sft-w_Qwen2.5-7B-Instruct_rank_continuous_split0_s0_"
    "beta0_k2_lq1p0_lw0p5/best-step-300"
)
RL_W_ADAPTER = (
    "data/results/memory_rl_campaign_20260826/train/"
    "formal_rl-w_Qwen2.5-7B-Instruct_rank_continuous_split0_s0_"
    "beta0p03_k2_lq1p0_lw0p5/best-step-200"
)
RL_QA_ADAPTER = (
    "data/results/memory_rl_campaign_20260826/train/"
    "formal_rl-qa_Qwen2.5-7B-Instruct_rank_continuous_split0_s0_"
    "beta0p03_k2_lq1_lw0/best-step-200"
)
HYBRID_ADAPTER = (
    "data/results/memory_rl_campaign_20260826/train/"
    "formal_rl-hybrid_Qwen2.5-7B-Instruct_rank_continuous_split0_s0_"
    "beta0p03_k2_lq1_lw0p25/best-step-200"
)


class StageCOODAnalysisError(ValueError):
    """The sealed input contract was violated."""


@dataclass(frozen=True)
class SourceContract:
    source: str
    results_path: str
    battery_path: str
    n_episodes: int
    n_items: int
    exploitable_episodes: int | None = None


@dataclass(frozen=True)
class StageCProtocol:
    model: str
    conditions: tuple[str, ...]
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
    baseline_adapters: tuple[tuple[str, str], ...]
    hybrid_adapter: str

    def source_contract(self, source: str) -> SourceContract:
        for contract in self.sources:
            if contract.source == source:
                return contract
        raise StageCOODAnalysisError(f"unexpected source {source!r}")


FORMAL_PROTOCOL = StageCProtocol(
    model="Qwen/Qwen2.5-7B-Instruct",
    conditions=CONDITION_ORDER,
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
            source="decoupled",
            results_path="data/results/results_v4f_7B-Instruct.json",
            battery_path="data/benchmarks/battery_v4_final.json",
            n_episodes=68,
            n_items=335,
            exploitable_episodes=25,
        ),
        SourceContract(
            source="compositional",
            results_path="data/results/results_v3f_7B-Instruct.json",
            battery_path="data/benchmarks/battery_v3d.json",
            n_episodes=52,
            n_items=261,
        ),
    ),
    baseline_adapters=(
        ("sft-w-s0-k2", SFT_ADAPTER),
        ("rl-w-s0-k2", RL_W_ADAPTER),
        ("rl-qa-s0-k2", RL_QA_ADAPTER),
    ),
    hybrid_adapter=HYBRID_ADAPTER,
)


def _error(message: str) -> None:
    raise StageCOODAnalysisError(message)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _error(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _error(f"{where} must be a non-empty string")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        _error(f"{where} must be a string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _error(f"{where} must be boolean")
    return value


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _error(f"{where} must be an integer")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _error(f"{where} must be finite")
    return result


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _expect_equal(actual: Any, expected: Any, where: str) -> None:
    if actual != expected:
        _error(f"{where} mismatch: expected {expected!r}, got {actual!r}")


def _strict_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StageCOODAnalysisError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _error(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StageCOODAnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise StageCOODAnalysisError(f"refusing to overwrite existing file: {path}") from exc


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    position = quantile * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        _error("AUC vectors must be non-empty and have equal length")
    positives = sum(bool(value) for value in labels)
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


def _exact_mcnemar(hybrid: Sequence[bool], comparator: Sequence[bool]) -> dict[str, Any]:
    if len(hybrid) != len(comparator) or not hybrid:
        _error("McNemar vectors must be non-empty and have equal length")
    hybrid_only = sum(a and not b for a, b in zip(hybrid, comparator))
    comparator_only = sum(b and not a for a, b in zip(hybrid, comparator))
    discordant = hybrid_only + comparator_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(hybrid_only, comparator_only)
        numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
        p_value = min(1.0, 2.0 * numerator / (2**discordant))
    return {
        "hybrid_only_correct": hybrid_only,
        "comparator_only_correct": comparator_only,
        "discordant": discordant,
        "exact_two_sided_p_value": p_value,
        "n_episodes": len(hybrid),
    }


def _bootstrap_draws(n_episodes: int, samples: int, seed: int) -> list[list[int]]:
    if n_episodes <= 0 or samples <= 0:
        _error("bootstrap needs positive episode and sample counts")
    rng = random.Random(seed)
    return [
        [rng.randrange(n_episodes) for _ in range(n_episodes)]
        for _ in range(samples)
    ]


def _paired_delta(
    hybrid: Sequence[bool],
    comparator: Sequence[bool],
    draws: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(hybrid) != len(comparator) or not hybrid:
        _error("paired bootstrap vectors must be non-empty and have equal length")
    point = sum(int(a) - int(b) for a, b in zip(hybrid, comparator)) / len(hybrid)
    distribution = [
        sum(int(hybrid[index]) - int(comparator[index]) for index in draw) / len(draw)
        for draw in draws
    ]
    return {
        "estimate": point,
        "ci_95": [_percentile(distribution, 0.025), _percentile(distribution, 0.975)],
        "bootstrap_samples_effective": len(distribution),
        "probability_gt_zero": (
            sum(value > 0.0 for value in distribution) / len(distribution)
        ),
    }


def _validate_lock(lock: Mapping[str, Any], protocol: StageCProtocol) -> str:
    _expect_equal(lock.get("schema_version"), 1, "lock.schema_version")
    _expect_equal(lock.get("lock_status"), "locked", "lock.lock_status")
    authorization = _mapping(lock.get("authorization"), "lock.authorization")
    _expect_equal(authorization.get("pre_ood"), True, "lock.authorization.pre_ood")
    state = _mapping(lock.get("ood_state_at_lock"), "lock.ood_state_at_lock")
    _expect_equal(state.get("results_inspected"), False, "lock.ood_state_at_lock.results_inspected")
    _expect_equal(
        state.get("completed_artifact_exists"),
        False,
        "lock.ood_state_at_lock.completed_artifact_exists",
    )
    locked = _mapping(lock.get("locked_configuration"), "lock.locked_configuration")
    expected = {
        "method": "rl-hybrid",
        "model": protocol.model,
        "seed": 0,
        "split_seed": 0,
        "budget": protocol.budget,
        "lambda_qa": 1,
        "lambda_w": 0.25,
        "checkpoint_step": 200,
        "checkpoint_path": protocol.hybrid_adapter,
        "teacher_mismatch_override": False,
        "validator_status": "pass",
    }
    for key, value in expected.items():
        _expect_equal(locked.get(key), value, f"lock.locked_configuration.{key}")
    _expect_equal(
        lock.get("reporter_correlations_used_for_selection"),
        False,
        "lock.reporter_correlations_used_for_selection",
    )
    rule = _list(lock.get("selection_rule"), "lock.selection_rule")
    if not rule or "smallest lambda_w" not in _string(rule[-1], "lock.selection_rule[-1]"):
        _error("lock.selection_rule must contain the approved smallest-lambda tie-break")
    next_experiment = _mapping(
        lock.get("authorized_next_experiment"), "lock.authorized_next_experiment"
    )
    _expect_equal(
        next_experiment.get("scope"), "one-shot Stage C OOD", "lock authorized scope"
    )
    _expect_equal(
        set(_list(next_experiment.get("batteries"), "lock authorized batteries")),
        {contract.source for contract in protocol.sources},
        "lock authorized batteries",
    )
    _expect_equal(
        next_experiment.get("primary_qa_batch_size"),
        protocol.qa_batch_size,
        "lock authorized QA batch",
    )
    _expect_equal(
        next_experiment.get("answer_tokens"),
        protocol.max_new_tokens,
        "lock authorized answer tokens",
    )
    _expect_equal(
        next_experiment.get("bootstrap_samples"),
        protocol.bootstrap_samples,
        "lock authorized bootstrap samples",
    )
    return _string(locked.get("checkpoint_path"), "lock checkpoint path")


def _expected_adapters(protocol: StageCProtocol, hybrid_path: str) -> dict[str, str]:
    return {**dict(protocol.baseline_adapters), HYBRID: hybrid_path}


def _comparison_label(comparator: str) -> str:
    return f"{HYBRID}_minus_{comparator}"


def _validate_config(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: StageCProtocol,
    hybrid_path: str,
) -> None:
    _expect_equal(payload.get("schema_version"), 1, f"{contract.source}.schema_version")
    _expect_equal(
        payload.get("condition_order"),
        list(protocol.conditions),
        f"{contract.source}.condition_order",
    )
    config = _mapping(payload.get("config"), f"{contract.source}.config")
    expected_scalars = {
        "model": protocol.model,
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
    for key, expected in expected_scalars.items():
        _expect_equal(config.get(key), expected, f"{contract.source}.config.{key}")
    _expect_equal(
        config.get("adapters"),
        _expected_adapters(protocol, hybrid_path),
        f"{contract.source}.config.adapters",
    )
    _expect_equal(config.get("rating_json"), {}, f"{contract.source}.config.rating_json")
    _expect_equal(config.get("embedding_model"), None, f"{contract.source}.config.embedding_model")
    specs = _list(config.get("specs"), f"{contract.source}.config.specs")
    if len(specs) != 1:
        _error(f"{contract.source} must contain exactly one source spec")
    spec = _mapping(specs[0], f"{contract.source}.config.specs[0]")
    expected_spec = {
        "name": contract.source,
        "source": contract.source,
        "results_path": contract.results_path,
        "battery_path": contract.battery_path,
    }
    _expect_equal(dict(spec), expected_spec, f"{contract.source}.config.specs[0]")
    refs = _mapping(payload.get("refs"), f"{contract.source}.refs")
    _expect_equal(refs.get("skipped"), False, f"{contract.source}.refs.skipped")
    mcnemar = _mapping(payload.get("mcnemar"), f"{contract.source}.mcnemar")
    _expect_equal(mcnemar.get("skipped"), False, f"{contract.source}.mcnemar.skipped")
    no_harm = _mapping(payload.get("no_harm"), f"{contract.source}.no_harm")
    _expect_equal(no_harm.get("skipped"), False, f"{contract.source}.no_harm.skipped")


def _validate_items(
    payload: Mapping[str, Any], contract: SourceContract, protocol: StageCProtocol
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = _list(payload.get("per_item"), f"{contract.source}.per_item")
    if len(rows) != contract.n_items:
        _error(
            f"{contract.source}.per_item expected {contract.n_items}, got {len(rows)}"
        )
    seen: set[str] = set()
    by_episode: dict[str, list[dict[str, Any]]] = {}
    normalized = []
    for row_index, raw in enumerate(rows):
        where = f"{contract.source}.per_item[{row_index}]"
        row = dict(_mapping(raw, where))
        uid = _string(row.get("uid"), f"{where}.uid")
        if uid in seen:
            _error(f"duplicate candidate UID: {uid}")
        seen.add(uid)
        episode_uid = _string(row.get("episode_uid"), f"{where}.episode_uid")
        source_episode = _integer(row.get("source_episode"), f"{where}.source_episode")
        candidate_index = _integer(row.get("candidate_index"), f"{where}.candidate_index")
        _expect_equal(row.get("source"), contract.source, f"{where}.source")
        expected_episode = f"{contract.source}:episode:{source_episode:06d}"
        expected_uid = f"{expected_episode}:candidate:{candidate_index:03d}"
        _expect_equal(episode_uid, expected_episode, f"{where}.episode_uid")
        _expect_equal(uid, expected_uid, f"{where}.uid")
        if not 0 <= source_episode < contract.n_episodes:
            _error(f"{where}.source_episode out of range")
        label = _string(row.get("label"), f"{where}.label")
        if label not in {"load_bearing", "distractor", "filler"}:
            _error(f"{where}.label is not a recognized utility label")
        _string(row.get("concept"), f"{where}.concept")
        scores = _mapping(row.get("scores"), f"{where}.scores")
        _expect_equal(set(scores), set(protocol.conditions), f"{where}.scores keys")
        for condition in protocol.conditions:
            _number(scores.get(condition), f"{where}.scores.{condition}")
        normalized.append(row)
        by_episode.setdefault(episode_uid, []).append(row)

    expected_episode_uids = {
        f"{contract.source}:episode:{index:06d}" for index in range(contract.n_episodes)
    }
    _expect_equal(set(by_episode), expected_episode_uids, f"{contract.source} item episode UIDs")
    for episode_uid, episode_rows in by_episode.items():
        ordered = sorted(episode_rows, key=lambda row: row["candidate_index"])
        _expect_equal(
            [row["candidate_index"] for row in ordered],
            list(range(len(ordered))),
            f"{episode_uid} candidate indexes",
        )
        if sum(row["label"] == "load_bearing" for row in ordered) != 1:
            _error(f"{episode_uid} must contain exactly one load-bearing candidate")
        by_episode[episode_uid] = ordered
    return normalized, by_episode


def _validate_episodes(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: StageCProtocol,
    items_by_episode: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[bool]]]]:
    episodes = _list(payload.get("per_episode"), f"{contract.source}.per_episode")
    if len(episodes) != contract.n_episodes:
        _error(
            f"{contract.source}.per_episode expected {contract.n_episodes}, got {len(episodes)}"
        )
    expected_uids = [
        f"{contract.source}:episode:{index:06d}" for index in range(contract.n_episodes)
    ]
    actual_uids = [
        _string(_mapping(row, "per_episode row").get("uid"), "per_episode.uid")
        for row in episodes
    ]
    _expect_equal(actual_uids, expected_uids, f"{contract.source}.per_episode UID order")
    vectors = {
        condition: {"qa": [], "containment": []} for condition in protocol.conditions
    }
    normalized = []
    for episode_index, raw in enumerate(episodes):
        where = f"{contract.source}.per_episode[{episode_index}]"
        episode = dict(_mapping(raw, where))
        uid = expected_uids[episode_index]
        _expect_equal(episode.get("source"), contract.source, f"{where}.source")
        _expect_equal(episode.get("source_episode"), episode_index, f"{where}.source_episode")
        policies = _mapping(episode.get("policies"), f"{where}.policies")
        _expect_equal(list(policies), list(protocol.conditions), f"{where}.policy order")
        candidate_rows = items_by_episode[uid]
        candidate_uids = [row["uid"] for row in candidate_rows]
        candidate_concepts = [row["concept"] for row in candidate_rows]
        labels = [row["label"] for row in candidate_rows]
        for condition in protocol.conditions:
            policy = _mapping(policies.get(condition), f"{where}.policies.{condition}")
            selections = _mapping(
                policy.get("selections"), f"{where}.policies.{condition}.selections"
            )
            selection = _mapping(
                selections.get(str(protocol.budget)),
                f"{where}.policies.{condition}.selections.{protocol.budget}",
            )
            indices = _list(selection.get("selected_indices"), f"{where} selected_indices")
            if (
                len(indices) != protocol.budget
                or len(set(indices)) != protocol.budget
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(candidate_rows)
                    for index in indices
                )
            ):
                _error(f"{where}.{condition} must select exactly {protocol.budget} unique candidates")
            expected_selected_uids = [candidate_uids[index] for index in indices]
            expected_selected_concepts = [candidate_concepts[index] for index in indices]
            _expect_equal(
                selection.get("selected_candidate_uids"),
                expected_selected_uids,
                f"{where}.{condition}.selected_candidate_uids",
            )
            _expect_equal(
                selection.get("selected_concepts"),
                expected_selected_concepts,
                f"{where}.{condition}.selected_concepts",
            )
            expected_containment = any(labels[index] == "load_bearing" for index in indices)
            containment = _boolean(
                selection.get("contains_load_bearing"),
                f"{where}.{condition}.contains_load_bearing",
            )
            _expect_equal(
                containment, expected_containment, f"{where}.{condition}.containment"
            )
            qa = _mapping(selection.get("qa"), f"{where}.{condition}.qa")
            correct = _boolean(qa.get("correct"), f"{where}.{condition}.qa.correct")
            vectors[condition]["qa"].append(correct)
            vectors[condition]["containment"].append(containment)
        refs = _mapping(episode.get("refs"), f"{where}.refs")
        oracle_ref = _mapping(
            refs.get(f"oracle@{protocol.budget}"), f"{where}.refs.oracle@{protocol.budget}"
        )
        oracle_correct = _boolean(oracle_ref.get("correct"), f"{where}.oracle correct")
        if oracle_correct != vectors["oracle"]["qa"][-1]:
            _error(f"{where} oracle ref and oracle policy QA disagree")
        normalized.append(episode)
    return normalized, vectors


def _validate_auc_and_metrics(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: StageCProtocol,
    items: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, list[bool]]],
) -> tuple[dict[str, float], Mapping[str, Any]]:
    metrics = _mapping(payload.get("metrics"), f"{contract.source}.metrics")
    by_spec = _mapping(metrics.get("by_spec"), f"{contract.source}.metrics.by_spec")
    _expect_equal(list(by_spec), [contract.source], f"{contract.source}.metrics.by_spec keys")
    scope = _mapping(by_spec.get(contract.source), f"{contract.source} metric scope")
    _expect_equal(scope.get("n_episodes"), contract.n_episodes, f"{contract.source}.n_episodes")
    _expect_equal(scope.get("n_items"), contract.n_items, f"{contract.source}.n_items")
    bootstrap = _mapping(scope.get("bootstrap"), f"{contract.source}.bootstrap")
    expected_bootstrap = {
        "method": "episode_cluster_percentile",
        "confidence": 0.95,
        "samples_requested": protocol.bootstrap_samples,
        "seed": protocol.bootstrap_seed,
        "skipped": False,
    }
    _expect_equal(dict(bootstrap), expected_bootstrap, f"{contract.source}.bootstrap")
    conditions = _mapping(scope.get("conditions"), f"{contract.source}.conditions")
    _expect_equal(list(conditions), list(protocol.conditions), f"{contract.source}.condition metric order")
    labels = [row["label"] == "load_bearing" for row in items]
    aucs: dict[str, float] = {}
    for condition in protocol.conditions:
        summary = _mapping(conditions.get(condition), f"{contract.source}.{condition}")
        classification = _mapping(summary.get("classification"), f"{condition}.classification")
        recorded_auc = _number(classification.get("pooled_auc"), f"{condition}.pooled_auc")
        computed_auc = _binary_auc(
            labels,
            [_number(row["scores"][condition], f"score {condition}") for row in items],
        )
        if not _close(recorded_auc, computed_auc):
            _error(
                f"{contract.source}.{condition} pooled AUC mismatch: "
                f"metrics={recorded_auc}, per_item={computed_auc}"
            )
        _expect_equal(classification.get("n_items"), contract.n_items, f"{condition}.n_items")
        _expect_equal(
            classification.get("n_episodes"), contract.n_episodes, f"{condition}.n_episodes"
        )
        auc_bootstrap = _mapping(
            classification.get("pooled_auc_bootstrap"), f"{condition}.AUC bootstrap"
        )
        if not _close(
            _number(auc_bootstrap.get("estimate"), f"{condition}.AUC estimate"),
            recorded_auc,
        ):
            _error(f"{contract.source}.{condition} AUC bootstrap estimate mismatch")
        _expect_equal(
            auc_bootstrap.get("bootstrap_samples_effective"),
            protocol.bootstrap_samples,
            f"{condition}.AUC bootstrap effective samples",
        )
        ci = _list(auc_bootstrap.get("ci_95"), f"{condition}.AUC ci_95")
        if len(ci) != 2:
            _error(f"{condition}.AUC ci_95 must contain two endpoints")
        _number(ci[0], f"{condition}.AUC ci lower")
        _number(ci[1], f"{condition}.AUC ci upper")
        selection = _mapping(summary.get("selection"), f"{condition}.selection")
        at_budget = _mapping(selection.get(str(protocol.budget)), f"{condition}.selection@2")
        recorded_containment = _number(
            at_budget.get("top_k_containment"), f"{condition}.top_k_containment"
        )
        computed_containment = sum(vectors[condition]["containment"]) / contract.n_episodes
        if not _close(recorded_containment, computed_containment):
            _error(f"{contract.source}.{condition} containment mismatch")
        _expect_equal(at_budget.get("n_episodes"), contract.n_episodes, f"{condition}.selection n")
        qa = _mapping(summary.get("qa"), f"{condition}.qa")
        qa_at_budget = _mapping(qa.get(str(protocol.budget)), f"{condition}.qa@2")
        recorded_qa = _number(qa_at_budget.get("accuracy"), f"{condition}.qa accuracy")
        computed_qa = sum(vectors[condition]["qa"]) / contract.n_episodes
        if not _close(recorded_qa, computed_qa):
            _error(f"{contract.source}.{condition} QA mismatch")
        _expect_equal(qa_at_budget.get("n_episodes"), contract.n_episodes, f"{condition}.qa n")
        aucs[condition] = recorded_auc

    paired = _list(scope.get("paired_auc_differences"), f"{contract.source}.paired AUC")
    expected_pairs = list(itertools.combinations(protocol.conditions, 2))
    actual_pairs = []
    for index, raw in enumerate(paired):
        row = _mapping(raw, f"{contract.source}.paired_auc[{index}]")
        left = _string(row.get("a"), f"paired_auc[{index}].a")
        right = _string(row.get("b"), f"paired_auc[{index}].b")
        actual_pairs.append((left, right))
        _expect_equal(row.get("direction"), "a_minus_b", f"paired_auc[{index}].direction")
        pooled = _mapping(row.get("pooled_auc_difference"), f"paired_auc[{index}].pooled")
        estimate = _number(pooled.get("estimate"), f"paired_auc[{index}].estimate")
        if not _close(estimate, aucs[left] - aucs[right]):
            _error(f"{contract.source} paired AUC estimate mismatch for {left}, {right}")
        ci = _list(pooled.get("ci_95"), f"paired_auc[{index}].ci_95")
        if len(ci) != 2:
            _error("paired AUC CI must contain two endpoints")
        lower = _number(ci[0], f"paired_auc[{index}].ci lower")
        upper = _number(ci[1], f"paired_auc[{index}].ci upper")
        if lower > upper:
            _error("paired AUC CI endpoints are reversed")
        _expect_equal(
            pooled.get("bootstrap_samples_effective"),
            protocol.bootstrap_samples,
            f"paired_auc[{index}] effective samples",
        )
        probability = _number(
            pooled.get("probability_gt_zero"), f"paired_auc[{index}].probability_gt_zero"
        )
        if not 0.0 <= probability <= 1.0:
            _error("paired AUC probability_gt_zero must be in [0,1]")
    _expect_equal(actual_pairs, expected_pairs, f"{contract.source}.paired AUC pair order")
    return aucs, scope


def _oriented_auc_difference(
    scope: Mapping[str, Any], hybrid_auc: float, comparator_auc: float, comparator: str
) -> dict[str, Any]:
    rows = _list(scope.get("paired_auc_differences"), "paired_auc_differences")
    for raw in rows:
        row = _mapping(raw, "paired AUC row")
        left, right = row.get("a"), row.get("b")
        if {left, right} != {HYBRID, comparator}:
            continue
        pooled = _mapping(row.get("pooled_auc_difference"), "paired AUC pooled")
        stored_estimate = _number(pooled.get("estimate"), "paired AUC estimate")
        ci = _list(pooled.get("ci_95"), "paired AUC ci")
        lower, upper = _number(ci[0], "AUC ci lower"), _number(ci[1], "AUC ci upper")
        if left == HYBRID:
            estimate, oriented_ci = stored_estimate, [lower, upper]
        else:
            estimate, oriented_ci = -stored_estimate, [-upper, -lower]
        expected = hybrid_auc - comparator_auc
        if not _close(estimate, expected):
            _error(f"oriented AUC mismatch for Hybrid minus {comparator}")
        return {
            "direction": _comparison_label(comparator),
            "estimate": estimate,
            "ci_95": oriented_ci,
            "bootstrap_samples_effective": pooled.get("bootstrap_samples_effective"),
            "extracted_from": {"a": left, "b": right, "direction": "a_minus_b"},
        }
    _error(f"missing paired AUC comparison for Hybrid and {comparator}")


def _condition_summary(
    conditions: Sequence[str],
    aucs: Mapping[str, float],
    vectors: Mapping[str, Mapping[str, list[bool]]],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    output = {}
    metric_conditions = _mapping(scope.get("conditions"), "metric conditions")
    for condition in conditions:
        classification = _mapping(
            _mapping(metric_conditions.get(condition), condition).get("classification"),
            f"{condition}.classification",
        )
        auc_bootstrap = _mapping(
            classification.get("pooled_auc_bootstrap"), f"{condition}.AUC bootstrap"
        )
        n = len(vectors[condition]["qa"])
        output[condition] = {
            "pooled_auc": aucs[condition],
            "pooled_auc_ci_95": list(auc_bootstrap["ci_95"]),
            "top2_containment": sum(vectors[condition]["containment"]) / n,
            "qa_accuracy": sum(vectors[condition]["qa"]) / n,
            "n_episodes": n,
        }
    return output


def _paired_comparisons(
    vectors: Mapping[str, Mapping[str, list[bool]]],
    aucs: Mapping[str, float],
    scope: Mapping[str, Any],
    draws: Sequence[Sequence[int]],
) -> dict[str, Any]:
    output = {}
    for comparator in COMPARATORS:
        hybrid_qa = vectors[HYBRID]["qa"]
        comparator_qa = vectors[comparator]["qa"]
        output[_comparison_label(comparator)] = {
            "hybrid": HYBRID,
            "comparator": comparator,
            "qa": {
                "paired_episode_bootstrap": _paired_delta(
                    hybrid_qa, comparator_qa, draws
                ),
                "exact_mcnemar": _exact_mcnemar(hybrid_qa, comparator_qa),
            },
            "containment": {
                "paired_episode_bootstrap": _paired_delta(
                    vectors[HYBRID]["containment"],
                    vectors[comparator]["containment"],
                    draws,
                ),
                "exact_mcnemar": _exact_mcnemar(
                    vectors[HYBRID]["containment"],
                    vectors[comparator]["containment"],
                ),
            },
            "pooled_auc": _oriented_auc_difference(
                scope, aucs[HYBRID], aucs[comparator], comparator
            ),
        }
    return output


def _exploitable_summary(
    episodes: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, list[bool]]],
    contract: SourceContract,
    protocol: StageCProtocol,
) -> dict[str, Any]:
    indices = []
    for index, episode in enumerate(episodes):
        refs = _mapping(episode.get("refs"), f"episode[{index}].refs")
        oracle = _mapping(refs.get(f"oracle@{protocol.budget}"), "oracle ref")
        if _boolean(oracle.get("correct"), "oracle correct"):
            indices.append(index)
    _expect_equal(
        len(indices),
        contract.exploitable_episodes,
        f"{contract.source} exploitable episode count",
    )
    subset_vectors = {
        condition: {
            metric: [values[index] for index in indices]
            for metric, values in condition_vectors.items()
        }
        for condition, condition_vectors in vectors.items()
    }
    draws = _bootstrap_draws(
        len(indices), protocol.bootstrap_samples, protocol.bootstrap_seed
    )
    conditions = {}
    for condition in protocol.conditions:
        n = len(indices)
        conditions[condition] = {
            "qa_accuracy": sum(subset_vectors[condition]["qa"]) / n,
            "top2_containment": sum(subset_vectors[condition]["containment"]) / n,
            "n_episodes": n,
        }
    paired = {}
    for comparator in COMPARATORS:
        hqa = subset_vectors[HYBRID]["qa"]
        cqa = subset_vectors[comparator]["qa"]
        paired[_comparison_label(comparator)] = {
            "hybrid": HYBRID,
            "comparator": comparator,
            "qa": {
                "paired_episode_bootstrap": _paired_delta(hqa, cqa, draws),
                "exact_mcnemar": _exact_mcnemar(hqa, cqa),
            },
            "containment": {
                "paired_episode_bootstrap": _paired_delta(
                    subset_vectors[HYBRID]["containment"],
                    subset_vectors[comparator]["containment"],
                    draws,
                ),
                "exact_mcnemar": _exact_mcnemar(
                    subset_vectors[HYBRID]["containment"],
                    subset_vectors[comparator]["containment"],
                ),
            },
        }
    return {
        "definition": f"refs['oracle@{protocol.budget}'].correct == true",
        "n_episodes": len(indices),
        "total_source_episodes": contract.n_episodes,
        "episode_uids": [episodes[index]["uid"] for index in indices],
        "conditions": conditions,
        "paired_comparisons": paired,
    }


def _validate_no_harm(
    payload: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    contract: SourceContract,
) -> dict[str, Any]:
    """Recompute adapter-enabled full-context accuracy and paired tests.

    Workspace and Oracle are selection rules, not adapter-enabled language
    models.  Consequently the evaluator's no-harm scope contains the five
    model-bearing conditions only: Original and the four adapters.
    """
    no_harm = _mapping(payload.get("no_harm"), f"{contract.source}.no_harm")
    _expect_equal(
        list(no_harm),
        ["skipped", "mode", "separate_from", "summary"],
        f"{contract.source}.no_harm schema/order",
    )
    _expect_equal(no_harm.get("skipped"), False, f"{contract.source}.no_harm.skipped")
    _expect_equal(
        no_harm.get("mode"),
        "adapter_enabled_full_context_qa",
        f"{contract.source}.no_harm.mode",
    )
    _expect_equal(
        no_harm.get("separate_from"),
        "adapter_disabled_frozen_base_selection_recall",
        f"{contract.source}.no_harm.separate_from",
    )
    vectors = {condition: [] for condition in NO_HARM_CONDITIONS}
    for episode_index, episode in enumerate(episodes):
        where = f"{contract.source}.per_episode[{episode_index}].no_harm_full_context"
        details = _mapping(episode.get("no_harm_full_context"), where)
        _expect_equal(list(details), list(NO_HARM_CONDITIONS), f"{where} condition order")
        for condition in NO_HARM_CONDITIONS:
            detail = _mapping(details.get(condition), f"{where}.{condition}")
            _expect_equal(
                list(detail), ["answer", "correct"], f"{where}.{condition} schema/order"
            )
            # An empty decode is a valid model output, so completeness means the
            # answer key exists and is text, not that the decoded text is non-empty.
            _text(detail.get("answer"), f"{where}.{condition}.answer")
            vectors[condition].append(
                _boolean(detail.get("correct"), f"{where}.{condition}.correct")
            )

    summary = _mapping(no_harm.get("summary"), f"{contract.source}.no_harm.summary")
    _expect_equal(
        list(summary),
        ["by_spec", "aggregate"],
        f"{contract.source}.no_harm.summary schema/order",
    )
    by_spec = _mapping(summary.get("by_spec"), f"{contract.source}.no_harm.by_spec")
    _expect_equal(list(by_spec), [contract.source], f"{contract.source}.no_harm.by_spec keys")
    aggregate = _mapping(
        summary.get("aggregate"), f"{contract.source}.no_harm.summary.aggregate"
    )

    def validate_scope(scope_value: Mapping[str, Any], where: str) -> dict[str, Any]:
        scope = _mapping(scope_value, where)
        _expect_equal(list(scope), ["conditions", "comparisons"], f"{where} schema/order")
        recorded_conditions = _mapping(scope.get("conditions"), f"{where}.conditions")
        _expect_equal(
            list(recorded_conditions),
            list(NO_HARM_CONDITIONS),
            f"{where} condition order",
        )
        output_conditions = {}
        for condition in NO_HARM_CONDITIONS:
            recorded_where = f"{where}.conditions.{condition}"
            recorded = _mapping(recorded_conditions.get(condition), recorded_where)
            _expect_equal(
                list(recorded), ["accuracy", "n_episodes"], f"{recorded_where} schema/order"
            )
            accuracy = sum(vectors[condition]) / contract.n_episodes
            if not _close(
                _number(recorded.get("accuracy"), f"{recorded_where}.accuracy"),
                accuracy,
            ):
                _error(f"{contract.source} no-harm accuracy mismatch for {condition}")
            _expect_equal(
                recorded.get("n_episodes"),
                contract.n_episodes,
                f"{recorded_where}.n_episodes",
            )
            output_conditions[condition] = {
                "accuracy": accuracy,
                "n_episodes": contract.n_episodes,
            }

        recorded_comparisons = _list(scope.get("comparisons"), f"{where}.comparisons")
        adapters = NO_HARM_CONDITIONS[1:]
        if len(recorded_comparisons) != len(adapters):
            _error(f"{where} must compare every adapter with Original")
        output_comparisons = {}
        for index, adapter in enumerate(adapters):
            comparison_where = f"{where}.comparisons[{index}]"
            recorded = _mapping(recorded_comparisons[index], comparison_where)
            _expect_equal(
                list(recorded),
                [
                    "base",
                    "adapter",
                    "adapter_minus_base_accuracy",
                    "base_only_correct",
                    "adapter_only_correct",
                    "discordant",
                    "exact_mcnemar_p_value",
                    "n",
                ],
                f"{comparison_where} schema/order",
            )
            _expect_equal(recorded.get("base"), "original", f"{comparison_where}.base")
            _expect_equal(recorded.get("adapter"), adapter, f"{comparison_where}.adapter")
            exact = _exact_mcnemar(vectors[adapter], vectors["original"])
            delta = (
                output_conditions[adapter]["accuracy"]
                - output_conditions["original"]["accuracy"]
            )
            expected_fields = {
                "adapter_minus_base_accuracy": delta,
                "base_only_correct": exact["comparator_only_correct"],
                "adapter_only_correct": exact["hybrid_only_correct"],
                "discordant": exact["discordant"],
                "exact_mcnemar_p_value": exact["exact_two_sided_p_value"],
                "n": contract.n_episodes,
            }
            for key, expected in expected_fields.items():
                actual = recorded.get(key)
                if isinstance(expected, float):
                    if not _close(_number(actual, f"{comparison_where}.{key}"), expected):
                        _error(
                            f"{contract.source} no-harm comparison mismatch: {adapter}.{key}"
                        )
                else:
                    _expect_equal(actual, expected, f"{comparison_where}.{key}")
            output_comparisons[f"{adapter}_minus_original"] = {
                "adapter": adapter,
                "base": "original",
                "adapter_minus_original_accuracy": delta,
                "exact_mcnemar": exact,
            }
        return {
            "conditions": output_conditions,
            "comparisons": output_comparisons,
        }

    by_spec_output = validate_scope(
        _mapping(by_spec.get(contract.source), f"{contract.source}.no_harm scope"),
        f"{contract.source}.no_harm.summary.by_spec.{contract.source}",
    )
    aggregate_output = validate_scope(
        aggregate, f"{contract.source}.no_harm.summary.aggregate"
    )
    if aggregate_output != by_spec_output:
        _error(f"{contract.source} source-separated no-harm aggregate differs from by_spec")
    return {
        "mode": "adapter_enabled_full_context_qa",
        "condition_order": list(NO_HARM_CONDITIONS),
        "conditions": by_spec_output["conditions"],
        "comparisons": by_spec_output["comparisons"],
        "primary_hybrid_vs_original": by_spec_output["comparisons"][
            f"{HYBRID}_minus_original"
        ],
        "summary_verified": [f"by_spec.{contract.source}", "aggregate"],
    }


def _analyze_source(
    payload: Mapping[str, Any],
    contract: SourceContract,
    protocol: StageCProtocol,
    hybrid_path: str,
) -> dict[str, Any]:
    _validate_config(payload, contract, protocol, hybrid_path)
    items, items_by_episode = _validate_items(payload, contract, protocol)
    episodes, vectors = _validate_episodes(
        payload, contract, protocol, items_by_episode
    )
    aucs, scope = _validate_auc_and_metrics(
        payload, contract, protocol, items, vectors
    )
    draws = _bootstrap_draws(
        contract.n_episodes, protocol.bootstrap_samples, protocol.bootstrap_seed
    )
    result = {
        "source": contract.source,
        "n_episodes": contract.n_episodes,
        "n_items": contract.n_items,
        "conditions": _condition_summary(
            protocol.conditions, aucs, vectors, scope
        ),
        "paired_comparisons": _paired_comparisons(
            vectors, aucs, scope, draws
        ),
        "no_harm_full_context": _validate_no_harm(
            payload, episodes, contract
        ),
    }
    if contract.exploitable_episodes is not None:
        result["exploitable_subset"] = _exploitable_summary(
            episodes, vectors, contract, protocol
        )
    return result


def build_analysis(
    decoupled: Mapping[str, Any],
    compositional: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    protocol: StageCProtocol = FORMAL_PROTOCOL,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate raw evaluator payloads and return deterministic analysis JSON."""
    hybrid_path = _validate_lock(lock, protocol)
    payloads = {"decoupled": decoupled, "compositional": compositional}
    expected_sources = {contract.source for contract in protocol.sources}
    _expect_equal(set(payloads), expected_sources, "analysis source set")
    sources = {
        contract.source: _analyze_source(
            payloads[contract.source], contract, protocol, hybrid_path
        )
        for contract in protocol.sources
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "stage_c_ood_seed0_k2",
        "lock": {
            "status": "locked",
            "method": "rl-hybrid",
            "lambda_w": 0.25,
            "checkpoint_path": hybrid_path,
            "selection_scope": "pre-OOD ID-only lock",
        },
        "protocol": {
            "condition_order": list(protocol.conditions),
            "budget": protocol.budget,
            "workspace_top_k": protocol.workspace_top_k,
            "dtype": protocol.dtype,
            "admission_batch_size": protocol.admission_batch_size,
            "qa_batch_size": protocol.qa_batch_size,
            "no_harm_batch_size": protocol.no_harm_batch_size,
            "no_harm_condition_order": list(NO_HARM_CONDITIONS),
            "max_length": protocol.max_length,
            "max_new_tokens": protocol.max_new_tokens,
            "bootstrap": {
                "method": "paired_episode_cluster_percentile",
                "samples": protocol.bootstrap_samples,
                "seed": protocol.bootstrap_seed,
                "confidence": 0.95,
                "rng": "python.random.Random",
            },
            "mcnemar": "exact two-sided binomial",
        },
        "inputs": dict(provenance or {}),
        "sources": sources,
    }


def _analysis_provenance(
    decoupled_path: Path, compositional_path: Path, lock_path: Path
) -> dict[str, Any]:
    return {
        "decoupled": {
            "path": str(decoupled_path),
            "sha256": _file_sha256(decoupled_path),
        },
        "compositional": {
            "path": str(compositional_path),
            "sha256": _file_sha256(compositional_path),
        },
        "pre_ood_lock": {
            "path": str(lock_path),
            "sha256": _file_sha256(lock_path),
        },
    }


def analyze_files(
    decoupled_path: Path,
    compositional_path: Path,
    lock_path: Path,
    *,
    protocol: StageCProtocol = FORMAL_PROTOCOL,
) -> dict[str, Any]:
    provenance = _analysis_provenance(
        decoupled_path, compositional_path, lock_path
    )
    return build_analysis(
        _strict_load(decoupled_path),
        _strict_load(compositional_path),
        _strict_load(lock_path),
        protocol=protocol,
        provenance=provenance,
    )


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return f"{path}: object keys differ"
        for key in sorted(left):
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
    decoupled_path: Path,
    compositional_path: Path,
    lock_path: Path,
    analysis_path: Path,
    *,
    protocol: StageCProtocol = FORMAL_PROTOCOL,
) -> dict[str, Any]:
    """Independently rebuild the analysis and compare every serialized field."""
    try:
        expected = analyze_files(
            decoupled_path, compositional_path, lock_path, protocol=protocol
        )
        observed = _strict_load(analysis_path)
        difference = _first_difference(expected, observed)
        errors = [] if difference is None else [difference]
    except (StageCOODAnalysisError, OSError, ValueError) as exc:
        errors = [str(exc)]
    return {
        "schema_version": SCHEMA_VERSION,
        "validator": "stage_c_ood_analysis_raw_recomputation",
        "status": "pass" if not errors else "fail",
        "analysis": {
            "path": str(analysis_path),
            "sha256": _file_sha256(analysis_path) if analysis_path.is_file() else None,
        },
        "raw_inputs": {
            "decoupled": str(decoupled_path),
            "compositional": str(compositional_path),
            "pre_ood_lock": str(lock_path),
        },
        "errors": errors,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decoupled", required=True, type=Path)
    parser.add_argument("--compositional", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="build the analysis artifact")
    _add_common_arguments(analyze_parser)
    validate_parser = subparsers.add_parser(
        "validate", help="recompute and validate an existing analysis artifact"
    )
    _add_common_arguments(validate_parser)
    validate_parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze_files(args.decoupled, args.compositional, args.lock)
            _write_json_exclusive(args.out, result)
            print(f"saved Stage-C OOD analysis -> {args.out}")
            return 0
        result = validate_analysis_files(
            args.decoupled,
            args.compositional,
            args.lock,
            args.analysis,
        )
        _write_json_exclusive(args.out, result)
        print(f"saved Stage-C OOD analysis validation -> {args.out}")
        return 0 if result["status"] == "pass" else 1
    except StageCOODAnalysisError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
