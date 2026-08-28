"""Stage B0: measure exact-budget QA reward diversity without training.

The admission policy sees only ``(context, candidate concept)``.  The probe and
gold answer enter only after all exact-budget sets have been sampled, inside the
frozen adapter-disabled recall environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from memory_rl.data import (  # noqa: E402
    TrainingSpec,
    build_training_bundle,
    default_training_specs,
    file_sha256,
    write_split_manifest,
)
from memory_rl.modeling import (  # noqa: E402
    binary_action_logits,
    disable_dropout,
    load_lora_policy,
    render_admission_prompt,
    selection_logits,
)
from memory_rl.objectives import sample_gumbel_topk, set_logprob  # noqa: E402
from memory_rl.qa_preflight import (  # noqa: E402
    classify_gate_b0,
    select_temperature,
    summarize_group,
    summarize_preflight,
)
from memory_rl.recall import FrozenRecall  # noqa: E402


LOCKED_QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
KNOWN_TEACHER_MODELS = {"7B-Instruct": "Qwen/Qwen2.5-7B-Instruct"}


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, default=json_default, allow_nan=False) + "\n"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def software_versions() -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0], "cuda": torch.version.cuda}
    for package in ("torch", "transformers", "peft", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cached_snapshot_revision(model_name: str, filename: str) -> tuple[str | None, str | None]:
    """Resolve a cached Hub file and extract its immutable snapshot revision."""

    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_name, filename)
    if not isinstance(cached, str):
        return None, None
    # Keep the snapshot symlink path: resolving it points into ``blobs/`` and
    # discards the immutable commit component we need to audit.
    snapshot_path = Path(cached).absolute()
    parts = snapshot_path.parts
    try:
        revision = parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        revision = None
    return revision, str(snapshot_path)


def parse_spec(text: str) -> TrainingSpec:
    if "=" not in text or "::" not in text:
        raise argparse.ArgumentTypeError(
            "train spec must be NAME=RESULTS_JSON::BATTERY_JSON"
        )
    name, paths = text.split("=", 1)
    results, battery = paths.split("::", 1)
    return TrainingSpec(source=name, results_path=results, battery_path=battery)


def parse_temperatures(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "temperature candidates must be comma-separated numbers"
        ) from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("temperature candidates must be finite and > 0")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("temperature candidates must be unique")
    return tuple(sorted(values))


def chunks(values, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def policy_prompts(tokenizer, episode) -> tuple[list[str], list[str]]:
    """Construct and audit prompts before the reward-side probe is touched."""

    prompts = [
        render_admission_prompt(tokenizer, episode.context, candidate.concept)
        for candidate in episode.candidates
    ]
    # The locked ID corpus has no probe verbatim inside its context.  Fail
    # closed if a future data/code change puts q into the admission prompt.
    leaked = [index for index, prompt in enumerate(prompts) if episode.probe_question in prompt]
    if leaked:
        raise RuntimeError(
            f"probe leaked into admission prompt for {episode.uid}, candidates {leaked}"
        )
    hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    return prompts, hashes


@torch.no_grad()
def score_initial_policy(bundle, episodes, max_length: int, batch_size: int):
    """Score every ID-train candidate once with the fresh zero-init LoRA policy."""

    all_candidates = [
        candidate for episode in episodes for candidate in episode.candidates
    ]
    all_prompts: list[str] = []
    prompt_hashes: dict[str, str] = {}
    for episode in episodes:
        prompts, hashes = policy_prompts(bundle.tokenizer, episode)
        all_prompts.extend(prompts)
        prompt_hashes.update(
            (candidate.uid, digest)
            for candidate, digest in zip(episode.candidates, hashes)
        )

    was_training = bundle.model.training
    bundle.model.eval()
    action_by_uid: dict[str, list[float]] = {}
    try:
        for candidate_chunk, prompt_chunk in zip(
            chunks(all_candidates, batch_size), chunks(all_prompts, batch_size)
        ):
            action_logits = binary_action_logits(
                bundle.model,
                bundle.tokenizer,
                list(prompt_chunk),
                bundle.action_token_ids,
                bundle.device,
                max_length,
            )
            for candidate, row in zip(candidate_chunk, action_logits.cpu().tolist()):
                if len(row) != 2 or not all(math.isfinite(float(value)) for value in row):
                    raise RuntimeError(f"non-finite action logits for {candidate.uid}")
                action_by_uid[candidate.uid] = [float(row[0]), float(row[1])]
    finally:
        bundle.model.train(was_training)

    episode_scores: dict[str, dict] = {}
    for episode in episodes:
        action = torch.tensor(
            [action_by_uid[candidate.uid] for candidate in episode.candidates],
            dtype=torch.float32,
        )
        episode_scores[episode.uid] = {
            "action_logits_no_yes": action.tolist(),
            "action_probabilities_no_yes": F.softmax(action, dim=-1).tolist(),
            "selection_logits": selection_logits(action).tolist(),
            "yes_probabilities": F.softmax(action, dim=-1)[:, 1].tolist(),
            "policy_prompt_sha256": [
                prompt_hashes[candidate.uid] for candidate in episode.candidates
            ],
        }
    return episode_scores


def sample_episode_sets(
    episodes,
    episode_scores: dict[str, dict],
    *,
    budget: int,
    group_size: int,
    temperature: float,
    seed: int,
) -> dict[str, list[list[int]]]:
    result = {}
    for episode in episodes:
        # Episode-derived streams make the draws invariant to ordering,
        # sharding, and safe resume boundaries while preserving common random
        # numbers across the temperature candidates.
        digest = hashlib.sha256(f"{seed}:{episode.uid}".encode("utf-8")).digest()
        episode_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(episode_seed)
        scores = torch.tensor(
            episode_scores[episode.uid]["selection_logits"], dtype=torch.float32
        )
        selected = sample_gumbel_topk(
            scores,
            budget,
            num_samples=group_size,
            temperature=temperature,
            generator=generator,
        )
        result[episode.uid] = selected.tolist()
    return result


def temperature_calibration(episodes, episode_scores, args):
    rows = []
    draws = {}
    for temperature in args.temperature_candidates:
        sampled = sample_episode_sets(
            episodes,
            episode_scores,
            budget=args.budget,
            group_size=args.group_size,
            temperature=temperature,
            seed=args.seed,
        )
        draws[temperature] = sampled
        unique_counts = [
            len({tuple(indices) for indices in sampled[episode.uid]})
            for episode in episodes
        ]
        diverse = sum(value >= 2 for value in unique_counts)
        rows.append(
            {
                "temperature": temperature,
                "episodes": len(episodes),
                "group_size": args.group_size,
                "median_unique_selected_sets": float(statistics.median(unique_counts)),
                "mean_unique_selected_sets": statistics.fmean(unique_counts),
                "min_unique_selected_sets": min(unique_counts),
                "max_unique_selected_sets": max(unique_counts),
                "groups_with_at_least_2_unique_sets": diverse,
                "groups_with_at_least_2_unique_sets_fraction": diverse / len(episodes),
            }
        )
    chosen, reason = select_temperature(
        rows, min_median_unique_sets=args.min_median_unique_sets
    )
    return {
        "schema_version": 1,
        "selection_uses_QA_or_OOD": False,
        "common_random_numbers": True,
        "sampling_seed": args.seed,
        "sampling_seed_scheme": "sha256(seed:episode_id), first 63 bits",
        "selection_rule": (
            "lowest candidate with median unique selected sets >= "
            f"{args.min_median_unique_sets:g}; otherwise highest candidate"
        ),
        "candidates": rows,
        "selected_temperature": chosen,
        "selection_reason": reason,
    }, draws[chosen]


def exact_inclusion_probabilities(scores: torch.Tensor, budget: int, temperature: float):
    """Enumerate the tiny n<=7 exact-k support and return item marginals."""

    probabilities = []
    inclusion = torch.zeros(scores.numel(), dtype=torch.float64)
    for indices in itertools.combinations(range(scores.numel()), budget):
        probability = set_logprob(
            scores, indices, temperature=temperature
        ).double().exp()
        probabilities.append(probability)
        for index in indices:
            inclusion[index] += probability
    total = torch.stack(probabilities).sum()
    if not torch.isfinite(total) or abs(float(total) - 1.0) > 1e-5:
        raise RuntimeError(f"enumerated exact-set probabilities sum to {float(total)}")
    return (inclusion / total).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-training Stage-B0 exact-budget QA reward preflight"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--teacher-tag", default="7B-Instruct")
    parser.add_argument(
        "--workspace-teacher-model", default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--expected-model-revision",
        default=None,
        help="optional exact resolved model commit required for a locked campaign",
    )
    parser.add_argument(
        "--expected-split-sha256",
        default=None,
        help="optional exact split manifest hash required for a locked campaign",
    )
    parser.add_argument("--train-spec", action="append", type=parse_spec)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workspace-top-k", type=int, default=2)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--temperature-candidates", type=parse_temperatures, default=parse_temperatures("0.7,1,2,3,5")
    )
    parser.add_argument("--min-median-unique-sets", type=float, default=4.0)
    parser.add_argument("--policy-batch-size", type=int, default=16)
    parser.add_argument("--recall-batch-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if args.budget < 1 or args.group_size != 16:
        parser.error("--budget must be >= 1 and Stage B0 requires --group-size 16")
    if args.workspace_top_k < 1 or args.lora_rank < 1:
        parser.error("--workspace-top-k and --lora-rank must be >= 1")
    if args.policy_batch_size < 1 or args.recall_batch_size < 1:
        parser.error("batch sizes must be >= 1")
    if args.max_length < 1 or args.answer_tokens < 1:
        parser.error("length limits must be positive")
    if args.limit_episodes < 0:
        parser.error("--limit-episodes must be non-negative")
    if not math.isfinite(args.min_median_unique_sets) or args.min_median_unique_sets < 1:
        parser.error("--min-median-unique-sets must be finite and >= 1")

    teacher_model = args.workspace_teacher_model or KNOWN_TEACHER_MODELS.get(
        args.teacher_tag
    )
    if teacher_model is None or teacher_model.rstrip("/") != args.model.rstrip("/"):
        parser.error("B0 requires workspace teacher, policy, and recall to be the same model")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    specs = tuple(args.train_spec or default_training_specs(args.teacher_tag))
    data_bundle = build_training_bundle(
        specs,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        top_k=args.workspace_top_k,
    )
    manifest = write_split_manifest(
        out_dir / "split_manifest.json", data_bundle.split_manifest
    )
    if (
        args.expected_split_sha256
        and manifest["manifest_sha256"] != args.expected_split_sha256
    ):
        raise RuntimeError(
            "split manifest mismatch: "
            f"{manifest['manifest_sha256']} != {args.expected_split_sha256}"
        )
    episodes = list(data_bundle.train_episodes)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    if not episodes:
        raise RuntimeError("B0 training split is empty")
    too_small = [episode.uid for episode in episodes if len(episode.candidates) <= args.budget]
    if too_small:
        raise RuntimeError(f"budget must be smaller than candidate count: {too_small[0]}")

    config = vars(args).copy()
    config["temperature_candidates"] = list(args.temperature_candidates)
    config["train_spec"] = [
        {
            "source": spec.canonical_source,
            "results_path": str(spec.results_path),
            "battery_path": str(spec.battery_path),
        }
        for spec in specs
    ]
    config.update(
        {
            "schema_version": 1,
            "stage": "B0",
            "training_performed": False,
            "optimizer_created": False,
            "policy_initialization": "fresh zero-initialized LoRA over frozen base",
            "policy_input_fields": ["context", "candidate.concept"],
            "probe_visible_to_policy": False,
            "gold_answer_visible_to_policy": False,
            "sets_sampled_before_reward_prompts": True,
            "sampling": "exact-budget Gumbel top-k",
            "selection_probability_semantics": (
                "softmax(logit/temperature) is the first-draw Plackett-Luce weight; "
                "selected_set_probability is the exact unordered set probability"
            ),
            "frozen_recall_adapter_disabled": True,
            "grader": "memory_rl.recall.grade_answer",
            "train_episode_count": len(episodes),
            "validation_episode_count": len(data_bundle.validation_episodes),
            "full_train_episode_count": len(data_bundle.train_episodes),
            "split_manifest_sha256": manifest["manifest_sha256"],
            "workspace_teacher_model": teacher_model,
            "teacher_matches_policy_reference": True,
            "teacher_mismatch_override": False,
            "software_versions": software_versions(),
        }
    )
    write_json(out_dir / "run_config.json", config)
    if args.dry_run:
        write_json(
            out_dir / "summary.json",
            {
                "schema_version": 1,
                "stage": "B0",
                "status": "dry-run",
                "training_performed": False,
                "train_episode_count": len(episodes),
                "validation_episode_count": len(data_bundle.validation_episodes),
                "split_manifest_sha256": manifest["manifest_sha256"],
            },
        )
        print(
            f"B0 dry-run: train={len(episodes)} validation={len(data_bundle.validation_episodes)} "
            f"manifest={manifest['manifest_sha256']}",
            flush=True,
        )
        return

    started = time.time()
    bundle = load_lora_policy(
        args.model,
        device=args.device,
        dtype=args.dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha or None,
        lora_dropout=args.lora_dropout,
    )
    dropout_audit = disable_dropout(bundle.model)
    if not dropout_audit["postcondition_satisfied"]:
        raise RuntimeError("dropout could not be disabled for B0 sampling")
    write_json(out_dir / "dropout_audit.json", dropout_audit)

    get_base = getattr(bundle.model, "get_base_model", None)
    resolved_model = get_base() if callable(get_base) else bundle.model
    resolved_revision = getattr(resolved_model.config, "_commit_hash", None)
    tokenizer_revision = getattr(bundle.tokenizer, "init_kwargs", {}).get("_commit_hash")
    tokenizer_config_path = None
    if tokenizer_revision is None:
        tokenizer_revision, tokenizer_config_path = cached_snapshot_revision(
            args.model, "tokenizer_config.json"
        )
    if args.expected_model_revision and resolved_revision != args.expected_model_revision:
        raise RuntimeError(
            f"resolved model revision {resolved_revision!r} does not match "
            f"{args.expected_model_revision!r}"
        )
    if args.expected_model_revision and tokenizer_revision != args.expected_model_revision:
        raise RuntimeError(
            f"resolved tokenizer revision {tokenizer_revision!r} does not match "
            f"{args.expected_model_revision!r}"
        )
    config["resolved_model_name_or_path"] = getattr(
        resolved_model.config, "_name_or_path", args.model
    )
    config["resolved_model_commit"] = resolved_revision
    config["resolved_tokenizer_commit"] = tokenizer_revision
    config["resolved_tokenizer_config_path"] = tokenizer_config_path
    config["trainable_parameter_count"] = sum(
        parameter.numel()
        for parameter in bundle.model.parameters()
        if parameter.requires_grad
    )
    config["total_parameter_count"] = sum(
        parameter.numel() for parameter in bundle.model.parameters()
    )

    print(f"scoring fresh admission policy on {len(episodes)} ID-train episodes", flush=True)
    episode_scores = score_initial_policy(
        bundle, episodes, args.max_length, args.policy_batch_size
    )
    calibration, sampled_sets = temperature_calibration(episodes, episode_scores, args)
    chosen_temperature = float(calibration["selected_temperature"])
    config["selected_temperature"] = chosen_temperature
    config["temperature_selection_uses_QA_or_OOD"] = False
    write_json(out_dir / "temperature_calibration.json", calibration)
    write_json(out_dir / "run_config.json", config)
    print(
        f"selected temperature={chosen_temperature:g} using selection diversity only",
        flush=True,
    )

    recall = FrozenRecall(
        bundle.model,
        bundle.tokenizer,
        bundle.device,
        max_new_tokens=args.answer_tokens,
    )
    sample_path = out_dir / "samples.jsonl"
    group_path = out_dir / "groups.jsonl"
    reference_path = out_dir / "references.jsonl"
    sample_rows: list[dict] = []
    group_rows: list[dict] = []
    reference_rows: list[dict] = []

    for episode_index, episode in enumerate(episodes):
        # This is the first point at which the probe/gold-bearing environment is
        # invoked.  Every selected set above is already sampled and detached.
        refs = recall.references(episode, args.budget)
        selected_for_episode = sampled_sets[episode.uid]
        unique_sets = list(dict.fromkeys(tuple(indices) for indices in selected_for_episode))
        results_by_set: dict[tuple[int, ...], dict] = {}
        for set_chunk in chunks(unique_sets, args.recall_batch_size):
            evaluated = recall.evaluate_sets(
                episode, [list(indices) for indices in set_chunk]
            )
            results_by_set.update(zip(set_chunk, evaluated))

        reference_row = {
            "episode_id": episode.uid,
            "source": episode.source,
            "source_episode": episode.source_episode,
            "probe_question": episode.probe_question,
            "gold_answer": episode.answer,
            "oracle_set": [episode.candidates[index].uid for index in refs["oracle_set"]],
            "oracle_concepts": [
                episode.candidates[index].concept for index in refs["oracle_set"]
            ],
            "oracle_answer": refs["oracle_answer"],
            "oracle_QA_correct": bool(refs["oracle_correct"]),
            "full_context_answer": refs["full_context_answer"],
            "full_context_QA_correct": bool(refs["full_context_correct"]),
            "no_memory_answer": refs["no_memory_answer"],
            "no_memory_QA_correct": bool(refs["no_memory_correct"]),
        }
        reference_rows.append(reference_row)
        append_jsonl(reference_path, reference_row)

        policy = episode_scores[episode.uid]
        raw_scores = torch.tensor(policy["selection_logits"], dtype=torch.float32)
        first_draw_probabilities = F.softmax(
            raw_scores / chosen_temperature, dim=-1
        ).tolist()
        inclusion_probabilities = exact_inclusion_probabilities(
            raw_scores, args.budget, chosen_temperature
        )
        load_bearing_indices = {
            index
            for index, candidate in enumerate(episode.candidates)
            if candidate.label == "load_bearing"
        }
        occurrence_counts = {
            indices: selected_for_episode.count(list(indices))
            for indices in set(tuple(value) for value in selected_for_episode)
        }
        episode_sample_rows = []
        for sample_id, raw_indices in enumerate(selected_for_episode):
            indices = tuple(sorted(set(raw_indices)))
            if len(indices) != args.budget:
                raise RuntimeError(f"sampler violated exact budget in {episode.uid}")
            result = results_by_set[indices]
            selected_logp = float(
                set_logprob(
                    raw_scores, indices, temperature=chosen_temperature
                ).item()
            )
            contains = any(
                episode.candidates[index].label == "load_bearing" for index in indices
            )
            selected_load_bearing_count = len(load_bearing_indices & set(indices))
            row = {
                "schema_version": 1,
                "episode_id": episode.uid,
                "source": episode.source,
                "source_episode": episode.source_episode,
                "sample_id": sample_id,
                "split": "train",
                "seed": args.seed,
                "split_seed": args.split_seed,
                "group_size": args.group_size,
                "budget": args.budget,
                "temperature": chosen_temperature,
                "candidate_ids": [candidate.uid for candidate in episode.candidates],
                "candidate_text": [candidate.concept for candidate in episode.candidates],
                "candidate_labels": [candidate.label for candidate in episode.candidates],
                "policy_input_fields": ["context", "candidate.concept"],
                "probe_visible_to_policy": False,
                "policy_prompt_sha256": policy["policy_prompt_sha256"],
                "action_logits_no_yes": policy["action_logits_no_yes"],
                "action_probabilities_no_yes": policy[
                    "action_probabilities_no_yes"
                ],
                "selection_logits": policy["selection_logits"],
                "selection_probabilities": first_draw_probabilities,
                "first_draw_probabilities": first_draw_probabilities,
                "inclusion_probabilities": inclusion_probabilities,
                "yes_probabilities": policy["yes_probabilities"],
                "selected_set_log_probability": selected_logp,
                "selected_set_probability": math.exp(selected_logp),
                "selected_set": [episode.candidates[index].uid for index in indices],
                "selected_indices": list(indices),
                "selected_concepts": [
                    episode.candidates[index].concept for index in indices
                ],
                "set_occurrence_in_group": occurrence_counts[indices],
                "exact_budget": True,
                "workspace_scores": [candidate.w_ref for candidate in episode.candidates],
                "workspace_percentiles": [
                    candidate.w_percentile for candidate in episode.candidates
                ],
                "verbal_scores": [candidate.v_ref for candidate in episode.candidates],
                "contains_load_bearing": contains,
                "selected_load_bearing_count": selected_load_bearing_count,
                "contains_all_load_bearing": load_bearing_indices.issubset(indices),
                "workspace_reward_diagnostic": statistics.fmean(
                    episode.candidates[index].w_percentile for index in indices
                ),
                "probe_question": episode.probe_question,
                "gold_answer": episode.answer,
                "generated_answer": result["answer"],
                "QA_correct": bool(result["correct"]),
                "QA_reward": float(result["correct"]),
                "oracle_QA_correct": bool(refs["oracle_correct"]),
                "full_context_QA_correct": bool(refs["full_context_correct"]),
                "no_memory_QA_correct": bool(refs["no_memory_correct"]),
                "failure_type": (
                    "selection"
                    if not contains
                    else ("recall_or_composition" if not result["correct"] else "none")
                ),
            }
            sample_rows.append(row)
            episode_sample_rows.append(row)
            append_jsonl(sample_path, row)

        group = summarize_group(episode_sample_rows)
        group.update(
            {
                "source": episode.source,
                "source_episode": episode.source_episode,
                "temperature": chosen_temperature,
            }
        )
        group_rows.append(group)
        append_jsonl(group_path, group)
        if episode_index == 0 or (episode_index + 1) % 10 == 0:
            print(
                f"B0 recall {episode_index + 1}/{len(episodes)}: {episode.uid} "
                f"unique={group['number_unique_selected_sets']} "
                f"qa_std={group['QA_reward_std']:.3f}",
                flush=True,
            )

    summary = summarize_preflight(sample_rows, group_rows, reference_rows)
    summary["by_source"] = {}
    for source in sorted({row["source"] for row in group_rows}):
        summary["by_source"][source] = summarize_preflight(
            [row for row in sample_rows if row["source"] == source],
            [row for row in group_rows if row["source"] == source],
            [row for row in reference_rows if row["source"] == source],
        )

    # B1 uses G=8.  The continuation plan explicitly makes G=16 primary for
    # B0, so do not replace that gate; report two fixed, non-overlapping halves
    # as a sensitivity analysis for the later training regime.
    g8_groups = []
    for episode in episodes:
        episode_rows = sorted(
            [row for row in sample_rows if row["episode_id"] == episode.uid],
            key=lambda row: row["sample_id"],
        )
        if len(episode_rows) != 16:
            raise RuntimeError("G=8 sensitivity requires the locked G=16 B0 run")
        for half_index, half in enumerate((episode_rows[:8], episode_rows[8:])):
            group = summarize_group(half)
            group["half_index"] = half_index
            g8_groups.append(group)
    g8_mixed = sum(row["mixed_QA_reward_group"] for row in g8_groups)
    g8_unique = [row["number_unique_selected_sets"] for row in g8_groups]
    g8_mixed_fraction = g8_mixed / len(g8_groups)
    g8_median_unique = float(statistics.median(g8_unique))
    summary["g8_sensitivity"] = {
        "role": "secondary sensitivity for the planned B1 group size; does not replace the G16 B0 gate",
        "groups": len(g8_groups),
        "mixed_QA_reward_groups": g8_mixed,
        "mixed_QA_reward_groups_fraction": g8_mixed_fraction,
        "mixed_QA_reward_groups_percent": 100.0 * g8_mixed_fraction,
        "median_unique_selected_sets": g8_median_unique,
        "mean_unique_selected_sets": statistics.fmean(g8_unique),
        "mean_within_group_QA_reward_std": statistics.fmean(
            row["QA_reward_std"] for row in g8_groups
        ),
        "gate_if_G8_thresholds_were_applied": classify_gate_b0(
            g8_mixed_fraction, g8_median_unique
        ),
    }
    memory_dependent = sum(
        row["oracle_QA_correct"] and not row["no_memory_QA_correct"]
        for row in reference_rows
    )
    summary["references"]["memory_dependent_exploitable_episodes_secondary"] = (
        memory_dependent
    )
    summary["references"]["memory_dependent_exploitable_fraction_secondary"] = (
        memory_dependent / len(reference_rows)
    )
    summary.update(
        {
            "stage": "B0",
            "status": "complete",
            "training_performed": False,
            "selected_temperature": chosen_temperature,
            "split_manifest_sha256": manifest["manifest_sha256"],
            "elapsed_seconds": time.time() - started,
            "artifacts": {
                "samples.jsonl": file_sha256(sample_path),
                "groups.jsonl": file_sha256(group_path),
                "references.jsonl": file_sha256(reference_path),
                "temperature_calibration.json": file_sha256(
                    out_dir / "temperature_calibration.json"
                ),
                "split_manifest.json": file_sha256(out_dir / "split_manifest.json"),
            },
        }
    )
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
