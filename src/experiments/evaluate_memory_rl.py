"""Unified held-out evaluation for workspace-guided memory-admission policies.

The evaluator compares the frozen checkpoint's precomputed verbal report,
one or more LoRA admission policies, the workspace gate, and an oracle under
the same candidate sets and recall model.  Admission prompts are constructed
only from ``(context, candidate concept)``; the probe question is used solely
after a memory set has been selected.

Example::

    python src/experiments/evaluate_memory_rl.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --spec decoupled=data/results/results_v4f_7B-Instruct.json::data/benchmarks/battery_v4_final.json \
      --spec compositional=data/results/results_v3f_7B-Instruct.json::data/benchmarks/battery_v3d.json \
      --adapter rl_w=data/results/memory_rl/rl_w/adapter \
      --adapter hybrid=data/results/memory_rl/hybrid/adapter \
      --budgets 2,3 --out data/results/memory_rl/eval.json

For a quick selection-only pass, add both ``--skip-qa --skip-no-harm``.  The
original verbal scores and workspace scores come from each spec's result JSON,
so such a run with no adapters does not load the causal LM at all.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory_rl.data import (
    CandidateRecord,
    EpisodeRecord,
    canonicalize_source,
    within_episode_percentiles,
)
from memory_rl.modeling import (
    binary_action_logits,
    load_policy_for_eval,
    render_admission_prompt,
    selection_logits,
)
from memory_rl.recall import (
    FrozenRecall,
    full_context_prompt,
    grade_answer,
    render_chat,
)


RESERVED_CONDITIONS = {
    "original",
    "workspace",
    "oracle",
    "rating",
    "embedding",
    "no_memory",
    "full_context",
}


@dataclass(frozen=True)
class EvalSpec:
    """Read-only held-out battery/results pair.

    This intentionally does not reuse ``TrainingSpec``: the training loader
    rejects Decoupled and Compositional before touching disk, which is exactly
    the guardrail training needs and exactly the wrong contract for evaluation.
    Loaded rows are nevertheless represented by memory_rl.data's shared
    ``EpisodeRecord`` and ``CandidateRecord`` types.
    """

    name: str
    results_path: Path
    battery_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_path", Path(self.results_path))
        object.__setattr__(self, "battery_path", Path(self.battery_path))

    @property
    def canonical_source(self) -> str:
        return canonicalize_source(self.name, for_training=False)


def _finite_float(value, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite, got {result!r}")
    return result


def _parse_named_value(text: str, flag: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"{flag} must use NAME=VALUE, got {text!r}")
    name, value = text.split("=", 1)
    name, value = name.strip(), value.strip()
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{flag} must use non-empty NAME=VALUE")
    return name, value


def parse_spec(text: str) -> EvalSpec:
    """Parse the trainer-compatible NAME=RESULTS::BATTERY syntax."""
    name, paths = _parse_named_value(text, "--spec")
    if "::" not in paths:
        raise argparse.ArgumentTypeError(
            "--spec must use NAME=RESULTS_JSON::BATTERY_JSON"
        )
    results_path, battery_path = (part.strip() for part in paths.split("::", 1))
    if not results_path or not battery_path:
        raise argparse.ArgumentTypeError(
            "--spec must contain both RESULTS_JSON and BATTERY_JSON"
        )
    return EvalSpec(
        name=name,
        results_path=results_path,
        battery_path=battery_path,
    )


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_json_list(path: Path, description: str) -> list:
    if not path.is_file():
        raise ValueError(f"{description} file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description} file {path}: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{description} file must contain a JSON list: {path}")
    return value


def _required_string(value: Mapping[str, Any], key: str, where: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return result


def load_eval_spec(spec: EvalSpec, workspace_top_k: int) -> tuple[EpisodeRecord, ...]:
    """Strict occurrence-aware join for an evaluation source.

    The join mirrors memory_rl.data's validated training join but calls
    ``canonicalize_source(..., for_training=False)`` and performs no split.
    """
    source = spec.canonical_source
    battery = _load_json_list(spec.battery_path, "battery")
    raw_results = _load_json_list(spec.results_path, "results")
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_index, raw_row in enumerate(raw_results):
        where = f"{spec.results_path}[{row_index}]"
        if not isinstance(raw_row, dict):
            raise ValueError(f"{where} must be an object")
        row = dict(raw_row)
        episode_index = row.get("episode")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or not 0 <= episode_index < len(battery)
        ):
            raise ValueError(
                f"{where}.episode must be an integer in [0, {len(battery)})"
            )
        _required_string(row, "concept", where)
        _required_string(row, "label", where)
        w_ref = _finite_float(row.get("W_rr"), f"{where}.W_rr")
        if w_ref < 0:
            raise ValueError(f"{where}.W_rr must be non-negative")
        row["W_rr"] = w_ref
        if row.get("V") is not None:
            v_ref = _finite_float(row["V"], f"{where}.V")
            if not 0.0 <= v_ref <= 1.0:
                raise ValueError(f"{where}.V must be in [0, 1]")
            row["V"] = v_ref
        by_episode[episode_index].append(row)

    expected = set(range(len(battery)))
    if set(by_episode) != expected:
        raise ValueError(
            f"results/battery episode mismatch for {source}: "
            f"missing={sorted(expected - set(by_episode))}, "
            f"extra={sorted(set(by_episode) - expected)}"
        )

    episodes: list[EpisodeRecord] = []
    for episode_index, raw_episode in enumerate(battery):
        where = f"{spec.battery_path}[{episode_index}]"
        if not isinstance(raw_episode, dict):
            raise ValueError(f"{where} must be an object")
        context = _required_string(raw_episode, "context", where)
        probe_question = _required_string(raw_episode, "probe_question", where)
        answer = _required_string(raw_episode, "answer", where)
        items = raw_episode.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{where}.items must be a non-empty list")
        result_rows = by_episode[episode_index]
        if len(result_rows) != len(items):
            raise ValueError(
                f"{source} episode {episode_index} has {len(items)} candidates but "
                f"{len(result_rows)} result rows"
            )

        queues: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for row in result_rows:
            queues[(row["concept"], row["label"])].append(row)
        joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for candidate_index, raw_item in enumerate(items):
            item_where = f"{where}.items[{candidate_index}]"
            if not isinstance(raw_item, dict):
                raise ValueError(f"{item_where} must be an object")
            concept = _required_string(raw_item, "concept", item_where)
            label = _required_string(raw_item, "label", item_where)
            key = (concept, label)
            if not queues[key]:
                raise ValueError(
                    f"no result row matches {item_where}: concept={concept!r}, "
                    f"label={label!r}"
                )
            joined.append((dict(raw_item), queues[key].popleft()))
        leftovers = [
            (concept, label, len(rows))
            for (concept, label), rows in queues.items()
            if rows
        ]
        if leftovers:
            raise ValueError(
                f"unmatched result rows in {source} episode {episode_index}: {leftovers}"
            )

        episode_uid = f"{source}:episode:{episode_index:06d}"
        episode_fingerprint = _json_fingerprint(raw_episode)
        percentiles = within_episode_percentiles([row["W_rr"] for _, row in joined])
        ranked = sorted(
            range(len(joined)),
            key=lambda index: (
                -joined[index][1]["W_rr"],
                _json_fingerprint({"item": joined[index][0], "candidate_index": index}),
            ),
        )
        workspace_targets = set(ranked[: min(workspace_top_k, len(ranked))])
        candidates = []
        for candidate_index, ((item, row), percentile) in enumerate(
            zip(joined, percentiles)
        ):
            role = item.get("role")
            if role is not None and not isinstance(role, str):
                raise ValueError(
                    f"{where}.items[{candidate_index}].role must be a string or null"
                )
            candidate_uid = f"{episode_uid}:candidate:{candidate_index:03d}"
            candidate_fingerprint = _json_fingerprint(
                {
                    "episode_fingerprint": episode_fingerprint,
                    "candidate_index": candidate_index,
                    "item": item,
                }
            )
            candidates.append(
                CandidateRecord(
                    uid=candidate_uid,
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
                    episode_uid=episode_uid,
                    role=role,
                    fingerprint_sha256=candidate_fingerprint,
                    result=row,
                )
            )
        episodes.append(
            EpisodeRecord(
                uid=episode_uid,
                source=source,
                source_episode=episode_index,
                context=context,
                probe_question=probe_question,
                answer=answer,
                candidates=tuple(candidates),
                fingerprint_sha256=episode_fingerprint,
            )
        )
    return tuple(episodes)


def load_eval_specs(
    specs: Sequence[EvalSpec], workspace_top_k: int
) -> tuple[EpisodeRecord, ...]:
    canonical_sources = [spec.canonical_source for spec in specs]
    if len(set(canonical_sources)) != len(canonical_sources):
        raise ValueError(
            f"evaluation source specified more than once: {canonical_sources}"
        )
    return tuple(
        episode
        for spec in specs
        for episode in load_eval_spec(spec, workspace_top_k)
    )


def parse_named_paths(values: Sequence[str], flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, path = _parse_named_value(value, flag)
        if name in parsed:
            raise ValueError(f"duplicate {flag} name: {name!r}")
        parsed[name] = path
    return parsed


def parse_budgets(text: str) -> list[int]:
    try:
        budgets = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--budgets must be comma-separated integers") from exc
    budgets = list(dict.fromkeys(budgets))
    if not budgets or any(budget <= 0 for budget in budgets):
        raise argparse.ArgumentTypeError("--budgets must contain positive integers")
    return budgets


def ensure_output_paths_absent(paths: Sequence[Path]) -> None:
    """Fail closed before expensive evaluation if any output target exists."""
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {rendered}")


def all_candidates(episodes: Sequence[object]):
    for episode in episodes:
        for candidate in episode.candidates:
            yield episode, candidate


def validate_records(episodes: Sequence[object]) -> None:
    episode_uids: set[str] = set()
    candidate_uids: set[str] = set()
    for episode in episodes:
        if episode.uid in episode_uids:
            raise ValueError(f"duplicate episode uid: {episode.uid!r}")
        episode_uids.add(episode.uid)
        if not episode.candidates:
            raise ValueError(f"episode {episode.uid!r} has no candidates")
        for candidate in episode.candidates:
            if candidate.uid in candidate_uids:
                raise ValueError(f"duplicate candidate uid: {candidate.uid!r}")
            candidate_uids.add(candidate.uid)
            if candidate.episode_uid != episode.uid:
                raise ValueError(
                    f"candidate {candidate.uid!r} points to {candidate.episode_uid!r}, "
                    f"expected {episode.uid!r}"
                )


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Tie-aware Mann-Whitney AUC without an sklearn dependency."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores have different lengths")
    n_pos = sum(bool(label) for label in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum_pos = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        score = scores[order[start]]
        while stop < len(order) and scores[order[stop]] == score:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        rank_sum_pos += average_rank * sum(bool(labels[order[i]]) for i in range(start, stop))
        start = stop
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def classification_summary(
    episodes: Sequence[object], score_map: dict[str, float]
) -> dict:
    labels: list[bool] = []
    scores: list[float] = []
    episode_aucs: list[float] = []
    per_episode: dict[str, float | None] = {}
    eligible = []
    for episode in episodes:
        if not all(candidate.uid in score_map for candidate in episode.candidates):
            continue
        eligible.append(episode)
        ep_labels = [candidate.label == "load_bearing" for candidate in episode.candidates]
        ep_scores = [score_map[candidate.uid] for candidate in episode.candidates]
        value = binary_auc(ep_labels, ep_scores)
        per_episode[episode.uid] = value
        if value is not None:
            episode_aucs.append(value)
        labels.extend(ep_labels)
        scores.extend(ep_scores)
    return {
        "pooled_auc": binary_auc(labels, scores),
        "within_episode_auc": (
            sum(episode_aucs) / len(episode_aucs) if episode_aucs else None
        ),
        "n_items": len(labels),
        "n_episodes": len(eligible),
        "n_within_episode_auc": len(episode_aucs),
        "per_episode_auc": per_episode,
    }


def selected_indices(episode, score_map: dict[str, float], budget: int) -> list[int]:
    if not all(candidate.uid in score_map for candidate in episode.candidates):
        raise KeyError(f"incomplete scores for episode {episode.uid!r}")
    order = sorted(
        range(len(episode.candidates)),
        key=lambda index: (-score_map[episode.candidates[index].uid], index),
    )
    return order[: min(budget, len(order))]


def selection_record(episode, indices: Sequence[int]) -> dict:
    chosen = set(indices)
    positives = [
        index
        for index, candidate in enumerate(episode.candidates)
        if candidate.label == "load_bearing"
    ]
    selected_positive = [index for index in positives if index in chosen]
    return {
        "selected_indices": list(indices),
        "selected_candidate_uids": [episode.candidates[index].uid for index in indices],
        "selected_concepts": [episode.candidates[index].concept for index in indices],
        "contains_load_bearing": bool(selected_positive),
        "contains_all_load_bearing": bool(positives) and len(selected_positive) == len(positives),
        "selected_load_bearing": len(selected_positive),
        "total_load_bearing": len(positives),
        "load_bearing_recall": (
            len(selected_positive) / len(positives) if positives else None
        ),
    }


def condition_summary(
    episodes: Sequence[object],
    score_map: dict[str, float],
    budgets: Sequence[int],
) -> dict:
    classification = classification_summary(episodes, score_map)
    selection: dict[str, dict] = {}
    for budget in budgets:
        records = []
        for episode in episodes:
            if not all(candidate.uid in score_map for candidate in episode.candidates):
                continue
            records.append(selection_record(episode, selected_indices(episode, score_map, budget)))
        recalls = [
            record["load_bearing_recall"]
            for record in records
            if record["load_bearing_recall"] is not None
        ]
        selection[str(budget)] = {
            "top_k_containment": (
                sum(record["contains_load_bearing"] for record in records) / len(records)
                if records
                else None
            ),
            "all_load_bearing_containment": (
                sum(record["contains_all_load_bearing"] for record in records) / len(records)
                if records
                else None
            ),
            "mean_load_bearing_recall": sum(recalls) / len(recalls) if recalls else None,
            "n_episodes": len(records),
        }
    classification.pop("per_episode_auc")
    return {"classification": classification, "selection": selection}


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linearly interpolated percentile for a finite scalar sample."""
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if len(finite) == 1:
        return finite[0]
    position = quantile * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def interval_summary(estimate: float | None, values: Sequence[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "estimate": estimate,
        "ci_95": [percentile(finite, 0.025), percentile(finite, 0.975)],
        "bootstrap_samples_effective": len(finite),
        "probability_gt_zero": (
            sum(value > 0.0 for value in finite) / len(finite) if finite else None
        ),
    }


def episode_auc_payload(
    episode, score_map: dict[str, float]
) -> tuple[list[bool], list[float], float | None]:
    labels = [candidate.label == "load_bearing" for candidate in episode.candidates]
    values = [score_map[candidate.uid] for candidate in episode.candidates]
    return labels, values, binary_auc(labels, values)


def attach_episode_cluster_bootstrap(
    metric_scope: dict,
    episodes: Sequence[object],
    condition_order: Sequence[str],
    scores: dict[str, dict[str, float]],
    samples: int,
    seed: int,
) -> None:
    """Attach episode-cluster AUC intervals and paired-difference intervals.

    Every draw resamples whole episodes with replacement.  All conditions use
    the same draws, so subtracting their bootstrap values gives a paired AUC
    difference rather than two independent confidence intervals.
    """
    metric_scope["bootstrap"] = {
        "method": "episode_cluster_percentile",
        "confidence": 0.95,
        "samples_requested": samples,
        "seed": seed,
        "skipped": samples == 0,
    }
    if samples == 0:
        metric_scope["paired_auc_differences"] = []
        return
    if samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    if not episodes:
        metric_scope["paired_auc_differences"] = []
        return

    conditions = [
        condition
        for condition in condition_order
        if condition in metric_scope["conditions"]
        and all(
            candidate.uid in scores[condition]
            for episode in episodes
            for candidate in episode.candidates
        )
    ]
    payload = {
        condition: [episode_auc_payload(episode, scores[condition]) for episode in episodes]
        for condition in conditions
    }
    distributions: dict[str, dict[str, list[float]]] = {
        condition: {"pooled": [], "within": []} for condition in conditions
    }
    rng = random.Random(seed)
    for _ in range(samples):
        draw = [rng.randrange(len(episodes)) for _ in episodes]
        for condition in conditions:
            pooled_labels: list[bool] = []
            pooled_scores: list[float] = []
            within_values: list[float] = []
            for episode_index in draw:
                labels, values, within_auc = payload[condition][episode_index]
                pooled_labels.extend(labels)
                pooled_scores.extend(values)
                if within_auc is not None:
                    within_values.append(within_auc)
            pooled_auc = binary_auc(pooled_labels, pooled_scores)
            if pooled_auc is not None:
                distributions[condition]["pooled"].append(pooled_auc)
            if within_values:
                distributions[condition]["within"].append(
                    sum(within_values) / len(within_values)
                )

    for condition in conditions:
        classification = metric_scope["conditions"][condition]["classification"]
        pooled_interval = interval_summary(
            classification["pooled_auc"], distributions[condition]["pooled"]
        )
        within_interval = interval_summary(
            classification["within_episode_auc"], distributions[condition]["within"]
        )
        # P(>0) is useful for differences but not informative for AUC itself.
        pooled_interval.pop("probability_gt_zero")
        within_interval.pop("probability_gt_zero")
        classification["pooled_auc_bootstrap"] = pooled_interval
        classification["within_episode_auc_bootstrap"] = within_interval

    paired = []
    for left, right in itertools.combinations(conditions, 2):
        left_classification = metric_scope["conditions"][left]["classification"]
        right_classification = metric_scope["conditions"][right]["classification"]
        pooled_values = [
            left_value - right_value
            for left_value, right_value in zip(
                distributions[left]["pooled"], distributions[right]["pooled"]
            )
        ]
        within_values = [
            left_value - right_value
            for left_value, right_value in zip(
                distributions[left]["within"], distributions[right]["within"]
            )
        ]
        paired.append(
            {
                "a": left,
                "b": right,
                "direction": "a_minus_b",
                "pooled_auc_difference": interval_summary(
                    left_classification["pooled_auc"]
                    - right_classification["pooled_auc"],
                    pooled_values,
                ),
                "within_episode_auc_difference": interval_summary(
                    left_classification["within_episode_auc"]
                    - right_classification["within_episode_auc"],
                    within_values,
                ),
            }
        )
    metric_scope["paired_auc_differences"] = paired


@torch.no_grad()
def score_model_policy(
    bundle,
    episodes: Sequence[object],
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score candidates without ever rendering or passing the probe question."""
    keys: list[str] = []
    prompts: list[str] = []
    for episode, candidate in all_candidates(episodes):
        keys.append(candidate.uid)
        prompts.append(
            render_admission_prompt(bundle.tokenizer, episode.context, candidate.concept)
        )

    probabilities: dict[str, float] = {}
    log_odds: dict[str, float] = {}
    bundle.model.eval()
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        logits = binary_action_logits(
            bundle.model,
            bundle.tokenizer,
            prompts[start:stop],
            bundle.action_token_ids,
            bundle.device,
            max_length,
        )
        yes_probability = F.softmax(logits, dim=-1)[:, 1]
        candidate_log_odds = selection_logits(logits)
        for uid, probability, odds in zip(
            keys[start:stop], yes_probability.tolist(), candidate_log_odds.tolist()
        ):
            probabilities[uid] = _finite_float(probability, f"probability for {uid}")
            log_odds[uid] = _finite_float(odds, f"log odds for {uid}")
    return probabilities, log_odds


@torch.no_grad()
def evaluate_adapter_enabled_full_context(
    bundle,
    episodes: Sequence[object],
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, dict]:
    """Run no-harm full-context QA with the bundle's current adapter enabled.

    Unlike ``FrozenRecall``, this function deliberately never enters
    ``adapter_disabled``.  It is a capability-preservation measurement, not a
    memory-selection payoff measurement.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    was_training = model.training
    old_cache = getattr(model.config, "use_cache", None)
    old_padding = tokenizer.padding_side
    model.eval()
    if old_cache is not None:
        model.config.use_cache = True
    tokenizer.padding_side = "left"
    details: dict[str, dict] = {}
    try:
        for episode_batch in chunked(list(episodes), batch_size):
            prompts = [
                full_context_prompt(episode.context, episode.probe_question)
                for episode in episode_batch
            ]
            rendered = [render_chat(tokenizer, prompt) for prompt in prompts]
            encoded = tokenizer(rendered, padding=True, return_tensors="pt")
            encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
            pad_token_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
            prompt_length = encoded["input_ids"].shape[1]
            answers = tokenizer.batch_decode(
                output[:, prompt_length:], skip_special_tokens=True
            )
            for episode, answer in zip(episode_batch, answers):
                answer = answer.strip()
                details[episode.uid] = {
                    "answer": answer,
                    "correct": bool(grade_answer(answer, episode.answer)),
                }
    finally:
        tokenizer.padding_side = old_padding
        if old_cache is not None:
            model.config.use_cache = old_cache
        model.train(was_training)
    return details


def release_bundle(bundle):
    if bundle is None:
        return None
    try:
        del bundle.model
    except AttributeError:
        pass
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def load_rating_scores(path: str, episodes: Sequence[object]) -> dict[str, float]:
    try:
        payload = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"rating JSON does not exist: {path}") from exc
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("per_item"))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"rating JSON {path!r} must contain a list of item rows")

    by_uid: dict[str, float] = {}
    by_episode_concept: dict[tuple[int, str], float] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rating row {row_index} in {path!r} is not an object")
        value = row.get("V_rating", row.get("rating"))
        if value is None:
            raise ValueError(
                f"rating row {row_index} in {path!r} lacks V_rating/rating"
            )
        value = _finite_float(value, f"rating row {row_index} in {path}")
        candidate_uid = row.get("candidate_uid", row.get("uid"))
        if candidate_uid:
            if candidate_uid in by_uid:
                raise ValueError(f"duplicate rating candidate uid {candidate_uid!r}")
            by_uid[str(candidate_uid)] = value
        if "episode" in row and "concept" in row:
            key = (int(row["episode"]), str(row["concept"]))
            if key in by_episode_concept:
                raise ValueError(f"duplicate rating key {key!r} in {path!r}")
            by_episode_concept[key] = value

    scores: dict[str, float] = {}
    missing: list[str] = []
    for episode, candidate in all_candidates(episodes):
        value = by_uid.get(candidate.uid)
        if value is None:
            value = by_episode_concept.get((int(episode.source_episode), candidate.concept))
        if value is None:
            missing.append(candidate.uid)
        else:
            scores[candidate.uid] = value
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"rating JSON {path!r} is missing {len(missing)} evaluated candidates: {preview}"
        )
    return scores


@torch.no_grad()
def compute_embedding_scores(
    episodes: Sequence[object],
    model_name: str,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Reproduce downstream.py's normalized CLS cosine baseline."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "--embedding-model requires transformers; install the RL requirements"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"could not load embedding model {model_name!r}; ensure its dependencies "
            "and weights are installed/cached"
        ) from exc

    def embed(texts: Sequence[str]) -> torch.Tensor:
        batches = []
        for start in range(0, len(texts), batch_size):
            batch_text = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            if not hasattr(output, "last_hidden_state"):
                raise RuntimeError(
                    f"embedding model {model_name!r} has no last_hidden_state"
                )
            batches.append(F.normalize(output.last_hidden_state[:, 0].float(), dim=-1).cpu())
        return torch.cat(batches, dim=0)

    contexts = [episode.context for episode in episodes]
    context_vectors = embed(contexts)
    candidate_rows = list(all_candidates(episodes))
    candidate_vectors = embed([candidate.concept for _, candidate in candidate_rows])
    episode_position = {episode.uid: index for index, episode in enumerate(episodes)}
    scores = {}
    for row_index, (episode, candidate) in enumerate(candidate_rows):
        value = torch.dot(
            candidate_vectors[row_index], context_vectors[episode_position[episode.uid]]
        ).item()
        scores[candidate.uid] = _finite_float(value, f"embedding score for {candidate.uid}")
    del model, tokenizer, context_vectors, candidate_vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def exact_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    if len(a) != len(b):
        raise ValueError("McNemar vectors have different lengths")
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    b_only = sum(bool(y) and not bool(x) for x, y in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(a_only, b_only)
        numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
        p_value = min(1.0, 2.0 * numerator / (2**discordant))
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": p_value,
        "n": len(a),
    }


def no_harm_scope_summary(
    episodes: Sequence[object],
    condition_order: Sequence[str],
    details: dict[str, dict[str, dict]],
) -> dict:
    """Summarize adapter-enabled full-context accuracy vs the original base."""
    episode_uids = [episode.uid for episode in episodes]
    conditions = {}
    for condition in condition_order:
        rows = details.get(condition, {})
        shared = [uid for uid in episode_uids if uid in rows]
        value, count = accuracy(rows[uid]["correct"] for uid in shared)
        conditions[condition] = {"accuracy": value, "n_episodes": count}

    comparisons = []
    base_rows = details.get("original", {})
    for adapter in [name for name in condition_order if name != "original"]:
        adapter_rows = details.get(adapter, {})
        shared = [
            uid for uid in episode_uids if uid in base_rows and uid in adapter_rows
        ]
        if not shared:
            continue
        base_accuracy, _ = accuracy(base_rows[uid]["correct"] for uid in shared)
        adapter_accuracy, _ = accuracy(adapter_rows[uid]["correct"] for uid in shared)
        comparison = exact_mcnemar(
            [base_rows[uid]["correct"] for uid in shared],
            [adapter_rows[uid]["correct"] for uid in shared],
        )
        comparisons.append(
            {
                "base": "original",
                "adapter": adapter,
                "adapter_minus_base_accuracy": (
                    adapter_accuracy - base_accuracy
                    if adapter_accuracy is not None and base_accuracy is not None
                    else None
                ),
                "base_only_correct": comparison["a_only"],
                "adapter_only_correct": comparison["b_only"],
                "discordant": comparison["discordant"],
                "exact_mcnemar_p_value": comparison["p_value"],
                "n": comparison["n"],
            }
        )
    return {"conditions": conditions, "comparisons": comparisons}


def mcnemar_scope(
    episodes: Sequence[object],
    conditions: Sequence[str],
    budgets: Sequence[int],
    qa_correct: dict[tuple[str, int], dict[str, bool]],
    no_memory_correct: dict[str, bool],
) -> dict[str, list[dict]]:
    episode_uids = {episode.uid for episode in episodes}
    output: dict[str, list[dict]] = {}
    for budget in budgets:
        vectors: dict[str, dict[str, bool]] = {
            condition: {
                uid: value
                for uid, value in qa_correct.get((condition, budget), {}).items()
                if uid in episode_uids
            }
            for condition in conditions
        }
        vectors["no_memory"] = {
            uid: value for uid, value in no_memory_correct.items() if uid in episode_uids
        }
        comparisons = []
        for left, right in itertools.combinations(vectors, 2):
            shared = sorted(set(vectors[left]) & set(vectors[right]))
            if not shared:
                continue
            result = exact_mcnemar(
                [vectors[left][uid] for uid in shared],
                [vectors[right][uid] for uid in shared],
            )
            comparisons.append({"a": left, "b": right, **result})
        output[str(budget)] = comparisons
    return output


def chunked(values: Sequence, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def evaluate_recall(
    bundle,
    episodes: Sequence[object],
    conditions: Sequence[str],
    scores: dict[str, dict[str, float]],
    budgets: Sequence[int],
    max_new_tokens: int,
    qa_batch_size: int,
) -> tuple[dict, dict, dict, dict]:
    """Evaluate every selected set with one adapter-disabled base recall model."""
    recall = FrozenRecall(
        bundle.model,
        bundle.tokenizer,
        bundle.device,
        max_new_tokens=max_new_tokens,
    )
    qa_details: dict[tuple[str, int, str], dict] = {}
    qa_correct: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    refs_by_episode: dict[str, dict] = {}
    no_memory_correct: dict[str, bool] = {}
    full_context_correct: dict[str, bool] = {}

    for episode_index, episode in enumerate(episodes):
        references: dict[str, dict] = {}
        for budget_index, budget in enumerate(budgets):
            ref = recall.references(episode, budget)
            references[f"oracle@{budget}"] = {
                "selected_indices": ref["oracle_set"],
                "selected_concepts": [
                    episode.candidates[index].concept for index in ref["oracle_set"]
                ],
                "answer": ref["oracle_answer"],
                "correct": bool(ref["oracle_correct"]),
            }
            qa_details[("oracle", budget, episode.uid)] = {
                "selected_concepts": references[f"oracle@{budget}"]["selected_concepts"],
                "answer": ref["oracle_answer"],
                "correct": bool(ref["oracle_correct"]),
            }
            qa_correct[("oracle", budget)][episode.uid] = bool(ref["oracle_correct"])
            if budget_index == 0:
                references["full_context"] = {
                    "answer": ref["full_context_answer"],
                    "correct": bool(ref["full_context_correct"]),
                }
                references["no_memory"] = {
                    "answer": ref["no_memory_answer"],
                    "correct": bool(ref["no_memory_correct"]),
                }
                full_context_correct[episode.uid] = bool(ref["full_context_correct"])
                no_memory_correct[episode.uid] = bool(ref["no_memory_correct"])
        refs_by_episode[episode.uid] = references

        # Identical selected sets share one greedy recall result.  The oracle is
        # taken from FrozenRecall.references above so refs and policy QA agree.
        consumers: dict[tuple[int, ...], list[tuple[str, int]]] = defaultdict(list)
        for condition in conditions:
            if condition == "oracle":
                continue
            score_map = scores[condition]
            if not all(candidate.uid in score_map for candidate in episode.candidates):
                continue
            for budget in budgets:
                indices = tuple(selected_indices(episode, score_map, budget))
                consumers[indices].append((condition, budget))

        unique_sets = list(consumers)
        set_results: dict[tuple[int, ...], dict] = {}
        for set_chunk in chunked(unique_sets, qa_batch_size):
            results = recall.evaluate_sets(episode, [list(indices) for indices in set_chunk])
            set_results.update(zip(set_chunk, results))
        for indices, targets in consumers.items():
            result = set_results[indices]
            for condition, budget in targets:
                detail = {
                    "selected_concepts": result["selected_concepts"],
                    "answer": result["answer"],
                    "correct": bool(result["correct"]),
                }
                qa_details[(condition, budget, episode.uid)] = detail
                qa_correct[(condition, budget)][episode.uid] = bool(result["correct"])
        print(
            f"  recall {episode_index + 1}/{len(episodes)}: {episode.uid}",
            flush=True,
        )

    return qa_details, dict(qa_correct), refs_by_episode, {
        "no_memory": no_memory_correct,
        "full_context": full_context_correct,
    }


def accuracy(values: Iterable[bool]) -> tuple[float | None, int]:
    values = [bool(value) for value in values]
    return (sum(values) / len(values) if values else None, len(values))


def evaluation_batch_provenance(args) -> dict[str, int]:
    """Record batching knobs that can affect finite-precision greedy decoding."""

    return {
        "admission_batch_size": int(args.batch_size),
        "qa_batch_size": int(args.qa_batch_size),
        "no_harm_batch_size": int(args.no_harm_batch_size),
    }


def add_qa_metrics(
    metric_scope: dict,
    episodes: Sequence[object],
    conditions: Sequence[str],
    budgets: Sequence[int],
    qa_correct: dict[tuple[str, int], dict[str, bool]],
) -> None:
    episode_uids = {episode.uid for episode in episodes}
    for condition in conditions:
        if condition not in metric_scope["conditions"]:
            continue
        qa = {}
        for budget in budgets:
            vector = qa_correct.get((condition, budget), {})
            score, count = accuracy(
                value for uid, value in vector.items() if uid in episode_uids
            )
            qa[str(budget)] = {"accuracy": score, "n_episodes": count}
        metric_scope["conditions"][condition]["qa"] = qa


def build_metric_scope(
    episodes: Sequence[object],
    condition_order: Sequence[str],
    scores: dict[str, dict[str, float]],
    budgets: Sequence[int],
) -> dict:
    conditions = {}
    for condition in condition_order:
        summary = condition_summary(episodes, scores[condition], budgets)
        if summary["classification"]["n_episodes"]:
            conditions[condition] = summary
    return {
        "n_episodes": len(episodes),
        "n_items": sum(len(episode.candidates) for episode in episodes),
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="frozen base/instruct checkpoint")
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        type=parse_spec,
        help="repeatable NAME=RESULTS_JSON::BATTERY_JSON held-out specification",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeatable named LoRA adapter",
    )
    parser.add_argument(
        "--rating-json",
        action="append",
        default=[],
        metavar="SPEC=PATH",
        help="optional precomputed V_rating JSON; repeat once per evaluated spec",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="optional embedding baseline (same normalized-CLS cosine as downstream.py)",
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--budgets", type=parse_budgets, default=parse_budgets("2,3"))
    parser.add_argument("--workspace-top-k", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="admission scoring batch")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--qa-batch-size", type=int, default=8)
    parser.add_argument("--no-harm-batch-size", type=int, default=8)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=4000,
        help="episode-cluster bootstrap draws; 0 disables confidence intervals",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--recompute-original-verbal",
        action="store_true",
        help="ignore cached v_ref and score the original model again",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="skip frozen selection-recall generation, refs, and selection McNemar tests",
    )
    parser.add_argument(
        "--skip-no-harm",
        action="store_true",
        help="skip adapter-enabled full-context capability checks",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    if args.workspace_top_k <= 0:
        parser.error("--workspace-top-k must be positive")
    for name in (
        "batch_size",
        "embedding_batch_size",
        "max_length",
        "max_new_tokens",
        "qa_batch_size",
        "no_harm_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    spec_names = [spec.name for spec in args.spec]
    if len(set(spec_names)) != len(spec_names):
        parser.error("--spec names must be unique")
    canonical_sources = [spec.canonical_source for spec in args.spec]
    if len(set(canonical_sources)) != len(canonical_sources):
        parser.error(
            "--spec entries must resolve to distinct evaluation sources; got "
            + repr(canonical_sources)
        )
    source_for_name = {spec.name: spec.canonical_source for spec in args.spec}

    try:
        adapter_paths = parse_named_paths(args.adapter, "--adapter")
        rating_paths = parse_named_paths(args.rating_json, "--rating-json")
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    invalid_adapters = set(adapter_paths) & RESERVED_CONDITIONS
    if invalid_adapters:
        parser.error(f"reserved adapter name(s): {', '.join(sorted(invalid_adapters))}")
    unknown_rating_specs = set(rating_paths) - set(spec_names)
    if unknown_rating_specs:
        parser.error(
            "--rating-json refers to unknown spec(s): "
            + ", ".join(sorted(unknown_rating_specs))
        )

    output_path = Path(args.out)
    try:
        ensure_output_paths_absent([output_path])
    except FileExistsError as exc:
        parser.error(str(exc))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device.startswith("cuda") else "float32")
    embedding_device = args.embedding_device or device

    episodes = list(load_eval_specs(args.spec, args.workspace_top_k))
    if not episodes:
        raise ValueError("held-out specs produced no episodes")
    validate_records(episodes)
    by_source: dict[str, list] = defaultdict(list)
    for episode in episodes:
        by_source[episode.source].append(episode)
    missing_sources = set(canonical_sources) - set(by_source)
    if missing_sources:
        raise ValueError(f"spec(s) produced no episodes: {sorted(missing_sources)}")

    scores: dict[str, dict[str, float]] = {}
    log_odds: dict[str, dict[str, float]] = {}
    no_harm_details: dict[str, dict[str, dict]] = {}
    scores["workspace"] = {
        candidate.uid: _finite_float(candidate.w_ref, f"w_ref for {candidate.uid}")
        for _, candidate in all_candidates(episodes)
    }
    scores["oracle"] = {
        candidate.uid: float(candidate.label == "load_bearing")
        for _, candidate in all_candidates(episodes)
    }

    cached_original_available = all(
        getattr(candidate, "v_ref", None) is not None
        for _, candidate in all_candidates(episodes)
    )
    use_cached_original = (
        cached_original_available and not args.recompute_original_verbal
    )
    if use_cached_original:
        scores["original"] = {
            candidate.uid: _finite_float(candidate.v_ref, f"v_ref for {candidate.uid}")
            for _, candidate in all_candidates(episodes)
        }
        original_source = "precomputed_v_ref"
    if not use_cached_original or not args.skip_no_harm:
        print("loading original model for evaluation", flush=True)
        original_bundle = load_policy_for_eval(args.model, None, device, dtype)
        if not use_cached_original:
            print("scoring original verbal policy", flush=True)
            scores["original"], log_odds["original"] = score_model_policy(
                original_bundle, episodes, args.batch_size, args.max_length
            )
            original_source = "model_forward"
        if not args.skip_no_harm:
            print("running original adapter-enabled full-context no-harm QA", flush=True)
            no_harm_details["original"] = evaluate_adapter_enabled_full_context(
                original_bundle,
                episodes,
                args.no_harm_batch_size,
                args.max_new_tokens,
            )
        original_bundle = release_bundle(original_bundle)

    for adapter_name, adapter_path in adapter_paths.items():
        print(f"scoring adapter {adapter_name}: {adapter_path}", flush=True)
        try:
            bundle = load_policy_for_eval(args.model, adapter_path, device, dtype)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load adapter {adapter_name!r} from {adapter_path!r}"
            ) from exc
        scores[adapter_name], log_odds[adapter_name] = score_model_policy(
            bundle, episodes, args.batch_size, args.max_length
        )
        if not args.skip_no_harm:
            print(
                f"running adapter-enabled full-context no-harm QA: {adapter_name}",
                flush=True,
            )
            no_harm_details[adapter_name] = evaluate_adapter_enabled_full_context(
                bundle,
                episodes,
                args.no_harm_batch_size,
                args.max_new_tokens,
            )
        bundle = release_bundle(bundle)

    if rating_paths:
        rating_scores = {}
        for spec_name, path in rating_paths.items():
            source = source_for_name[spec_name]
            rating_scores.update(load_rating_scores(path, by_source[source]))
        scores["rating"] = rating_scores

    if args.embedding_model:
        print(f"computing embedding baseline: {args.embedding_model}", flush=True)
        scores["embedding"] = compute_embedding_scores(
            episodes,
            args.embedding_model,
            embedding_device,
            args.embedding_batch_size,
        )

    condition_order = ["original", *adapter_paths.keys(), "workspace"]
    if "rating" in scores:
        condition_order.append("rating")
    if "embedding" in scores:
        condition_order.append("embedding")
    condition_order.append("oracle")

    metrics = {
        "by_spec": {
            source: build_metric_scope(
                source_episodes, condition_order, scores, args.budgets
            )
            for source, source_episodes in by_source.items()
        },
        "aggregate": build_metric_scope(episodes, condition_order, scores, args.budgets),
    }
    for source_index, (source, source_episodes) in enumerate(by_source.items()):
        attach_episode_cluster_bootstrap(
            metrics["by_spec"][source],
            source_episodes,
            condition_order,
            scores,
            args.bootstrap_samples,
            args.bootstrap_seed + source_index,
        )

    classification_by_episode = {
        condition: classification_summary(episodes, score_map)["per_episode_auc"]
        for condition, score_map in scores.items()
    }
    per_episode: dict[str, dict] = {}
    for episode in episodes:
        policies = {}
        for condition in condition_order:
            score_map = scores[condition]
            if not all(candidate.uid in score_map for candidate in episode.candidates):
                continue
            selections = {
                str(budget): selection_record(
                    episode, selected_indices(episode, score_map, budget)
                )
                for budget in args.budgets
            }
            policies[condition] = {
                "within_episode_auc": classification_by_episode[condition].get(episode.uid),
                "selections": selections,
            }
        per_episode[episode.uid] = {
            "uid": episode.uid,
            "source": episode.source,
            "source_episode": episode.source_episode,
            "probe_question": episode.probe_question,
            "gold_answer": episode.answer,
            "policies": policies,
        }

    no_harm_conditions = ["original", *adapter_paths.keys()]
    if args.skip_no_harm:
        no_harm = {
            "skipped": True,
            "mode": "adapter_enabled_full_context_qa",
            "summary": {},
        }
    else:
        for episode in episodes:
            per_episode[episode.uid]["no_harm_full_context"] = {
                condition: no_harm_details[condition][episode.uid]
                for condition in no_harm_conditions
            }
        no_harm = {
            "skipped": False,
            "mode": "adapter_enabled_full_context_qa",
            "separate_from": "adapter_disabled_frozen_base_selection_recall",
            "summary": {
                "by_spec": {
                    source: no_harm_scope_summary(
                        source_episodes, no_harm_conditions, no_harm_details
                    )
                    for source, source_episodes in by_source.items()
                },
                "aggregate": no_harm_scope_summary(
                    episodes, no_harm_conditions, no_harm_details
                ),
            },
        }

    qa_details: dict = {}
    qa_correct: dict = {}
    refs_by_episode: dict = {}
    ref_vectors = {"no_memory": {}, "full_context": {}}
    refs = {"skipped": bool(args.skip_qa), "summary": {}}
    mcnemar = {"skipped": bool(args.skip_qa)}
    if not args.skip_qa:
        print("loading frozen base recall model", flush=True)
        recall_bundle = load_policy_for_eval(args.model, None, device, dtype)
        qa_details, qa_correct, refs_by_episode, ref_vectors = evaluate_recall(
            recall_bundle,
            episodes,
            condition_order,
            scores,
            args.budgets,
            args.max_new_tokens,
            args.qa_batch_size,
        )
        recall_bundle = release_bundle(recall_bundle)

        for episode in episodes:
            per_episode[episode.uid]["refs"] = refs_by_episode[episode.uid]
            for condition in condition_order:
                policy = per_episode[episode.uid]["policies"].get(condition)
                if policy is None:
                    continue
                for budget in args.budgets:
                    detail = qa_details.get((condition, budget, episode.uid))
                    if detail is not None:
                        policy["selections"][str(budget)]["qa"] = detail

        for source, source_episodes in by_source.items():
            add_qa_metrics(
                metrics["by_spec"][source],
                source_episodes,
                condition_order,
                args.budgets,
                qa_correct,
            )
        add_qa_metrics(
            metrics["aggregate"], episodes, condition_order, args.budgets, qa_correct
        )

        refs_summary = {}
        for scope, scope_episodes in [*by_source.items(), ("aggregate", episodes)]:
            uids = {episode.uid for episode in scope_episodes}
            full_accuracy, full_n = accuracy(
                value for uid, value in ref_vectors["full_context"].items() if uid in uids
            )
            no_accuracy, no_n = accuracy(
                value for uid, value in ref_vectors["no_memory"].items() if uid in uids
            )
            oracle = {}
            for budget in args.budgets:
                oracle_accuracy, oracle_n = accuracy(
                    value
                    for uid, value in qa_correct.get(("oracle", budget), {}).items()
                    if uid in uids
                )
                oracle[str(budget)] = {
                    "accuracy": oracle_accuracy,
                    "n_episodes": oracle_n,
                }
            refs_summary[scope] = {
                "full_context": {"accuracy": full_accuracy, "n_episodes": full_n},
                "no_memory": {"accuracy": no_accuracy, "n_episodes": no_n},
                "oracle": oracle,
            }
        refs = {"skipped": False, "summary": refs_summary}

        mcnemar_by_spec = {
            source: mcnemar_scope(
                source_episodes,
                condition_order,
                args.budgets,
                qa_correct,
                ref_vectors["no_memory"],
            )
            for source, source_episodes in by_source.items()
        }
        mcnemar = {
            "skipped": False,
            "by_spec": mcnemar_by_spec,
            "aggregate": mcnemar_scope(
                episodes,
                condition_order,
                args.budgets,
                qa_correct,
                ref_vectors["no_memory"],
            ),
        }

    per_item = []
    for episode in episodes:
        for index, candidate in enumerate(episode.candidates):
            item_scores = {
                condition: score_map[candidate.uid]
                for condition, score_map in scores.items()
                if candidate.uid in score_map
            }
            item_log_odds = {
                condition: score_map[candidate.uid]
                for condition, score_map in log_odds.items()
                if candidate.uid in score_map
            }
            row = {
                "uid": candidate.uid,
                "episode_uid": episode.uid,
                "source": episode.source,
                "source_episode": episode.source_episode,
                "candidate_index": getattr(candidate, "candidate_index", index),
                "concept": candidate.concept,
                "label": candidate.label,
                "scores": item_scores,
            }
            if item_log_odds:
                row["model_log_odds"] = item_log_odds
            for optional_field in ("role", "provenance"):
                value = getattr(candidate, optional_field, None)
                if value is not None:
                    row[optional_field] = value
            per_item.append(row)

    output = {
        "schema_version": 1,
        "config": {
            "model": args.model,
            "specs": [
                {
                    "name": spec.name,
                    "source": spec.canonical_source,
                    "results_path": str(spec.results_path),
                    "battery_path": str(spec.battery_path),
                }
                for spec in args.spec
            ],
            "adapters": adapter_paths,
            "rating_json": rating_paths,
            "embedding_model": args.embedding_model,
            "budgets": args.budgets,
            "workspace_top_k": args.workspace_top_k,
            "device": device,
            "dtype": dtype,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            **evaluation_batch_provenance(args),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "skip_qa": bool(args.skip_qa),
            "skip_no_harm": bool(args.skip_no_harm),
            "original_verbal_source": original_source,
            "policy_input_fields": ["context", "candidate.concept"],
            "probe_visible_to_policy": False,
            "recall_model": "adapter-disabled base checkpoint",
        },
        "condition_order": condition_order,
        "metrics": metrics,
        "no_harm": no_harm,
        "refs": refs,
        "mcnemar": mcnemar,
        "per_item": per_item,
        "per_episode": list(per_episode.values()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(output, handle, indent=args.indent, allow_nan=False)
    print(f"saved unified evaluation -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
