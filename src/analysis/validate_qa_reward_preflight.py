"""Strict artifact validator for the no-training Stage-B0 QA preflight."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from memory_rl.data import file_sha256, verify_split_manifest  # noqa: E402
from memory_rl.objectives import set_logprob  # noqa: E402
from memory_rl.qa_preflight import (  # noqa: E402
    classify_gate_b0,
    select_temperature,
    summarize_group,
    summarize_preflight,
)
from memory_rl.recall import grade_answer  # noqa: E402


REQUIRED_FORMAL = (
    "run_config.json",
    "split_manifest.json",
    "dropout_audit.json",
    "temperature_calibration.json",
    "samples.jsonl",
    "groups.jsonl",
    "references.jsonl",
    "summary.json",
)


def _reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value}")


def strict_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def strict_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at line {line_number}")
        value = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


def _finite(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _close(left, right, tolerance: float = 1e-8) -> bool:
    return _finite(left) and _finite(right) and math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def _subset_equal(expected, actual, path: str, add_error) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            add_error("summary_mismatch", f"{path} is not an object")
            return
        for key, value in expected.items():
            if key not in actual:
                add_error("summary_mismatch", f"{path}.{key} is missing")
            else:
                _subset_equal(value, actual[key], f"{path}.{key}", add_error)
        return
    if isinstance(expected, list):
        if expected != actual:
            add_error("summary_mismatch", f"{path} differs from recomputed value")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not _close(expected, actual):
            add_error(
                "summary_mismatch",
                f"{path}={actual!r}, expected recomputed {expected!r}",
            )
        return
    if expected != actual:
        add_error(
            "summary_mismatch", f"{path}={actual!r}, expected {expected!r}"
        )


def validate_run(
    run_dir: str | Path,
    *,
    profile: str = "formal",
    expected_manifest_sha256: str | None = None,
    expected_model_revision: str | None = None,
) -> dict:
    root = Path(run_dir).resolve()
    errors: list[dict] = []
    warnings: list[dict] = []
    suppressed_errors = 0

    def add_error(code: str, message: str) -> None:
        nonlocal suppressed_errors
        if len(errors) < 200:
            errors.append({"code": code, "message": message})
        else:
            suppressed_errors += 1

    if profile not in {"dry-run", "smoke", "formal"}:
        raise ValueError("profile must be dry-run, smoke, or formal")
    required = (
        ("run_config.json", "split_manifest.json", "summary.json")
        if profile == "dry-run"
        else REQUIRED_FORMAL
    )
    for name in required:
        if not (root / name).is_file():
            add_error("missing_artifact", f"missing {name}")
    if errors:
        return {
            "schema_version": 1,
            "status": "fail",
            "profile": profile,
            "run_dir": str(root),
            "errors": errors,
            "warnings": warnings,
        }

    try:
        config = strict_json(root / "run_config.json")
        manifest = strict_json(root / "split_manifest.json")
        summary = strict_json(root / "summary.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_error("invalid_json", str(exc))
        return {
            "schema_version": 1,
            "status": "fail",
            "profile": profile,
            "run_dir": str(root),
            "errors": errors,
            "warnings": warnings,
        }

    if not verify_split_manifest(manifest):
        add_error("split_manifest_invalid", "split manifest self-hash is invalid")
    manifest_sha = manifest.get("manifest_sha256")
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        add_error(
            "split_manifest_mismatch",
            f"manifest {manifest_sha!r} != {expected_manifest_sha256!r}",
        )
    if config.get("split_manifest_sha256") != manifest_sha:
        add_error("split_manifest_mismatch", "run_config does not bind the manifest")

    fixed_config = {
        "stage": "B0",
        "training_performed": False,
        "optimizer_created": False,
        "group_size": 16,
        "budget": 2,
        "seed": 0,
        "split_seed": 0,
        "answer_tokens": 64,
        "max_length": 2048,
        "probe_visible_to_policy": False,
        "gold_answer_visible_to_policy": False,
        "sets_sampled_before_reward_prompts": True,
        "frozen_recall_adapter_disabled": True,
        "teacher_matches_policy_reference": True,
        "teacher_mismatch_override": False,
    }
    for key, expected in fixed_config.items():
        if config.get(key) != expected:
            add_error("config_lock_mismatch", f"{key}={config.get(key)!r}, expected {expected!r}")
    if config.get("policy_input_fields") != ["context", "candidate.concept"]:
        add_error("probe_leakage_contract", "policy_input_fields are not locked")
    if config.get("sampling") != "exact-budget Gumbel top-k":
        add_error("sampling_contract", "sampling is not exact-budget Gumbel top-k")
    if expected_model_revision:
        for key in ("resolved_model_commit", "resolved_tokenizer_commit"):
            if config.get(key) != expected_model_revision:
                add_error(
                    "model_revision_mismatch",
                    f"{key}={config.get(key)!r}, expected {expected_model_revision!r}",
                )

    if profile == "dry-run":
        if summary.get("status") != "dry-run":
            add_error("summary_status", "dry-run summary status is not dry-run")
        return {
            "schema_version": 1,
            "status": "pass" if not errors else "fail",
            "profile": profile,
            "run_dir": str(root),
            "details": {
                "manifest_sha256": manifest_sha,
                "train_episode_count": manifest.get("train_episode_count"),
                "validation_episode_count": manifest.get("validation_episode_count"),
            },
            "errors": errors,
            "warnings": warnings,
        }

    if summary.get("status") != "complete" or summary.get("training_performed") is not False:
        add_error("summary_status", "B0 summary is not a complete no-training run")
    if profile == "formal" and config.get("limit_episodes") != 0:
        add_error("episode_scope", "formal B0 cannot use --limit-episodes")
    for forbidden in ("final_adapter", "best_checkpoint.json"):
        if (root / forbidden).exists():
            add_error("training_artifact_present", f"unexpected {forbidden}")
    if list(root.glob("best-step-*")) or list(root.glob("*.safetensors")):
        add_error("training_artifact_present", "B0 contains a checkpoint/model artifact")

    try:
        dropout = strict_json(root / "dropout_audit.json")
        calibration = strict_json(root / "temperature_calibration.json")
        samples = strict_jsonl(root / "samples.jsonl")
        groups = strict_jsonl(root / "groups.jsonl")
        references = strict_jsonl(root / "references.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_error("invalid_json", str(exc))
        samples, groups, references, calibration, dropout = [], [], [], {}, {}

    if dropout.get("postcondition_satisfied") is not True or dropout.get(
        "remaining_nonzero"
    ) != []:
        add_error("dropout_postcondition", "dropout audit postcondition failed")
    if calibration.get("selection_uses_QA_or_OOD") is not False:
        add_error("temperature_leakage", "temperature selection used QA or OOD")
    try:
        expected_temperature, _ = select_temperature(
            calibration.get("candidates", []),
            min_median_unique_sets=config.get("min_median_unique_sets", 4.0),
        )
        if not _close(expected_temperature, calibration.get("selected_temperature")):
            add_error("temperature_selection", "selected temperature violates the rule")
        if not _close(expected_temperature, config.get("selected_temperature")):
            add_error("temperature_selection", "run_config temperature differs")
    except ValueError as exc:
        add_error("temperature_selection", str(exc))

    ordered_train_ids = [
        episode_id
        for source_name in sorted(manifest.get("sources", {}))
        for episode_id in manifest["sources"][source_name].get("train_episode_ids", [])
    ]
    if profile == "smoke":
        limit = config.get("limit_episodes")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            add_error("episode_scope", "smoke profile requires a positive episode limit")
            limit = 0
        ordered_train_ids = ordered_train_ids[:limit]
    expected_train_ids = set(ordered_train_ids)
    sample_by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str):
            sample_by_episode[episode_id].append(row)
        else:
            add_error("sample_identity", "sample has no valid episode_id")
    group_by_episode = {row.get("episode_id"): row for row in groups}
    reference_by_episode = {row.get("episode_id"): row for row in references}
    observed_ids = set(sample_by_episode)
    if observed_ids != expected_train_ids:
        add_error(
            "episode_scope",
            f"sample episode IDs differ from the sealed train split: {len(observed_ids)} vs {len(expected_train_ids)}",
        )
    if set(group_by_episode) != observed_ids or set(reference_by_episode) != observed_ids:
        add_error("episode_scope", "sample/group/reference episode IDs do not match")
    if len(groups) != len(group_by_episode) or len(references) != len(reference_by_episode):
        add_error("duplicate_episode_summary", "duplicate group/reference episode rows")
    if len(samples) != len(observed_ids) * 16:
        add_error("sample_count", f"found {len(samples)} samples, expected {len(observed_ids) * 16}")

    required_sample_fields = {
        "episode_id",
        "sample_id",
        "candidate_ids",
        "candidate_text",
        "selection_logits",
        "selection_probabilities",
        "selected_set",
        "workspace_scores",
        "verbal_scores",
        "contains_load_bearing",
        "generated_answer",
        "QA_correct",
        "QA_reward",
        "oracle_QA_correct",
        "full_context_QA_correct",
        "no_memory_QA_correct",
        "temperature",
        "selected_indices",
        "selected_set_log_probability",
        "selected_set_probability",
    }
    deterministic_results: dict[tuple[str, tuple[str, ...]], tuple[str, bool]] = {}
    for episode_id, episode_rows in sample_by_episode.items():
        sample_ids = {row.get("sample_id") for row in episode_rows}
        if sample_ids != set(range(16)):
            add_error("sample_identity", f"{episode_id} sample IDs are not 0..15")
        reference = reference_by_episode.get(episode_id, {})
        first = episode_rows[0]
        repeated_fields = (
            "candidate_ids",
            "candidate_text",
            "candidate_labels",
            "policy_prompt_sha256",
            "action_logits_no_yes",
            "action_probabilities_no_yes",
            "selection_logits",
            "selection_probabilities",
            "first_draw_probabilities",
            "inclusion_probabilities",
            "yes_probabilities",
            "workspace_scores",
            "workspace_percentiles",
            "verbal_scores",
            "probe_question",
            "gold_answer",
        )
        counts = Counter(tuple(row.get("selected_indices", [])) for row in episode_rows)
        for row in episode_rows:
            missing = required_sample_fields - set(row)
            if missing:
                add_error("sample_schema", f"{episode_id} missing fields {sorted(missing)}")
                continue
            if any(row.get(key) != first.get(key) for key in repeated_fields):
                add_error("sample_schema", f"{episode_id} repeats inconsistent policy data")
                break
            if row.get("policy_input_fields") != ["context", "candidate.concept"] or row.get(
                "probe_visible_to_policy"
            ) is not False:
                add_error("probe_leakage_contract", f"{episode_id} sample policy boundary invalid")
            candidates = row.get("candidate_ids")
            arrays = [
                row.get("candidate_text"),
                row.get("candidate_labels"),
                row.get("policy_prompt_sha256"),
                row.get("action_logits_no_yes"),
                row.get("action_probabilities_no_yes"),
                row.get("selection_logits"),
                row.get("selection_probabilities"),
                row.get("first_draw_probabilities"),
                row.get("inclusion_probabilities"),
                row.get("yes_probabilities"),
                row.get("workspace_scores"),
                row.get("workspace_percentiles"),
                row.get("verbal_scores"),
            ]
            if not isinstance(candidates, list) or any(
                not isinstance(value, list) or len(value) != len(candidates)
                for value in arrays
            ):
                add_error("candidate_alignment", f"{episode_id} candidate arrays are misaligned")
                continue
            indices = row.get("selected_indices")
            if (
                not isinstance(indices, list)
                or indices != sorted(indices)
                or len(indices) != 2
                or len(set(indices)) != 2
                or any(not isinstance(index, int) or not 0 <= index < len(candidates) for index in indices)
            ):
                add_error("exact_budget", f"{episode_id} has an invalid selected index set")
                continue
            if row.get("selected_set") != [candidates[index] for index in indices]:
                add_error("exact_budget", f"{episode_id} selected IDs do not match indices")
            if row.get("selected_concepts") != [row["candidate_text"][index] for index in indices]:
                add_error("candidate_alignment", f"{episode_id} selected concepts do not match")
            if row.get("exact_budget") is not True or row.get("budget") != 2:
                add_error("exact_budget", f"{episode_id} exact-budget marker invalid")
            labels = row.get("candidate_labels")
            selected_lb = sum(labels[index] == "load_bearing" for index in indices)
            all_lb = all(
                index in indices
                for index, label in enumerate(labels)
                if label == "load_bearing"
            )
            if row.get("contains_load_bearing") != (selected_lb > 0):
                add_error("containment_mismatch", f"{episode_id} any-LB containment is wrong")
            if row.get("selected_load_bearing_count") != selected_lb or row.get(
                "contains_all_load_bearing"
            ) != all_lb:
                add_error("containment_mismatch", f"{episode_id} LB sensitivity fields are wrong")
            if row.get("set_occurrence_in_group") != counts[tuple(indices)]:
                add_error("set_occurrence", f"{episode_id} set occurrence count is wrong")
            correct = row.get("QA_correct")
            reward = row.get("QA_reward")
            if not isinstance(correct, bool) or reward != float(correct):
                add_error("qa_reward_mismatch", f"{episode_id} reward != correctness")
            if isinstance(row.get("generated_answer"), str) and isinstance(
                row.get("gold_answer"), str
            ):
                if grade_answer(row["generated_answer"], row["gold_answer"]) != correct:
                    add_error("grader_mismatch", f"{episode_id} generated answer regrades differently")
            else:
                add_error("sample_schema", f"{episode_id} answer/gold is not text")
            for sample_key, reference_key in (
                ("oracle_QA_correct", "oracle_QA_correct"),
                ("full_context_QA_correct", "full_context_QA_correct"),
                ("no_memory_QA_correct", "no_memory_QA_correct"),
            ):
                if row.get(sample_key) != reference.get(reference_key):
                    add_error("reference_mismatch", f"{episode_id} {sample_key} differs")
            identity = (episode_id, tuple(row.get("selected_set", [])))
            result = (row.get("generated_answer"), correct)
            if identity in deterministic_results and deterministic_results[identity] != result:
                add_error("nondeterministic_duplicate", f"{episode_id} duplicate set differs")
            deterministic_results[identity] = result

            numeric_vectors = (
                "selection_logits",
                "selection_probabilities",
                "first_draw_probabilities",
                "inclusion_probabilities",
                "yes_probabilities",
                "workspace_scores",
                "workspace_percentiles",
            )
            if any(
                any(not _finite(value) for value in row.get(key, []))
                for key in numeric_vectors
            ):
                add_error("nonfinite", f"{episode_id} contains non-finite scores")
                continue
            if not _close(
                sum(row["selection_probabilities"]), 1.0, tolerance=1e-6
            ) or row[
                "selection_probabilities"
            ] != row["first_draw_probabilities"]:
                add_error("probability_contract", f"{episode_id} first-draw probabilities invalid")
            elif _finite(row.get("temperature")):
                expected_first = torch.softmax(
                    torch.tensor(row["selection_logits"], dtype=torch.float32)
                    / float(row["temperature"]),
                    dim=-1,
                ).tolist()
                if any(
                    not _close(observed, expected, tolerance=1e-6)
                    for observed, expected in zip(
                        row["selection_probabilities"], expected_first
                    )
                ):
                    add_error(
                        "probability_contract",
                        f"{episode_id} first-draw probabilities disagree with logits",
                    )
            if not _close(sum(row["inclusion_probabilities"]), 2.0, tolerance=1e-6):
                add_error("probability_contract", f"{episode_id} inclusion marginals do not sum to k")
            action = row.get("action_logits_no_yes")
            action_probs = row.get("action_probabilities_no_yes")
            if any(
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(_finite(value) for value in pair)
                for pair in action + action_probs
            ):
                add_error("probability_contract", f"{episode_id} action arrays invalid")
            else:
                expected_action = torch.softmax(
                    torch.tensor(action, dtype=torch.float32), dim=-1
                ).tolist()
                for index, (logit_pair, probability_pair) in enumerate(
                    zip(action, action_probs)
                ):
                    if not _close(
                        sum(probability_pair), 1.0, tolerance=1e-6
                    ) or not _close(
                        probability_pair[1],
                        row["yes_probabilities"][index],
                        tolerance=1e-6,
                    ) or not _close(
                        logit_pair[1] - logit_pair[0], row["selection_logits"][index], tolerance=1e-6
                    ):
                        add_error("probability_contract", f"{episode_id} action probabilities disagree")
                        break
                    if any(
                        not _close(observed, expected, tolerance=1e-6)
                        for observed, expected in zip(
                            probability_pair, expected_action[index]
                        )
                    ):
                        add_error(
                            "probability_contract",
                            f"{episode_id} action probabilities disagree with logits",
                        )
                        break
            if not all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in row.get("policy_prompt_sha256", [])
            ):
                add_error("prompt_provenance", f"{episode_id} prompt hashes invalid")
            if _finite(row.get("selected_set_log_probability")) and _finite(
                row.get("temperature")
            ):
                recomputed_logp = float(
                    set_logprob(
                        torch.tensor(row["selection_logits"], dtype=torch.float32),
                        indices,
                        temperature=float(row.get("temperature")),
                    ).item()
                )
                if not _close(recomputed_logp, row["selected_set_log_probability"], tolerance=1e-6):
                    add_error("set_probability", f"{episode_id} set log probability is wrong")
                if not _close(
                    math.exp(recomputed_logp), row.get("selected_set_probability"), tolerance=1e-6
                ):
                    add_error("set_probability", f"{episode_id} set probability is wrong")
            else:
                add_error("set_probability", f"{episode_id} set log probability is non-finite")

        if reference:
            if reference.get("probe_question") != first.get("probe_question") or reference.get(
                "gold_answer"
            ) != first.get("gold_answer"):
                add_error("reference_mismatch", f"{episode_id} reference question/gold differs")
            gold_answer = reference.get("gold_answer")
            if not isinstance(gold_answer, str):
                add_error("reference_mismatch", f"{episode_id} reference gold is not text")
                gold_answer = ""
            for answer_key, correct_key in (
                ("oracle_answer", "oracle_QA_correct"),
                ("full_context_answer", "full_context_QA_correct"),
                ("no_memory_answer", "no_memory_QA_correct"),
            ):
                answer = reference.get(answer_key)
                if not isinstance(answer, str) or grade_answer(
                    answer, gold_answer
                ) != reference.get(correct_key):
                    add_error("grader_mismatch", f"{episode_id} {answer_key} regrades differently")
            positives = [
                candidate
                for candidate, label in zip(first.get("candidate_ids", []), first.get("candidate_labels", []))
                if label == "load_bearing"
            ]
            negatives = [
                candidate
                for candidate, label in zip(first.get("candidate_ids", []), first.get("candidate_labels", []))
                if label != "load_bearing"
            ]
            if reference.get("oracle_set") != (positives + negatives)[:2]:
                add_error("oracle_definition", f"{episode_id} oracle set is not label-selected exact-k")

        try:
            recomputed_group = summarize_group(episode_rows)
            _subset_equal(
                recomputed_group,
                group_by_episode.get(episode_id),
                f"groups[{episode_id}]",
                add_error,
            )
        except ValueError as exc:
            add_error("group_recompute", f"{episode_id}: {exc}")

    if samples and groups and references:
        try:
            recomputed_summary = summarize_preflight(samples, groups, references)
            _subset_equal(recomputed_summary, summary, "summary", add_error)
        except ValueError as exc:
            add_error("summary_recompute", str(exc))

        by_source = summary.get("by_source", {})
        for source in sorted({row.get("source") for row in groups}):
            try:
                expected_source = summarize_preflight(
                    [row for row in samples if row.get("source") == source],
                    [row for row in groups if row.get("source") == source],
                    [row for row in references if row.get("source") == source],
                )
                _subset_equal(
                    expected_source,
                    by_source.get(source),
                    f"summary.by_source.{source}",
                    add_error,
                )
            except ValueError as exc:
                add_error("summary_recompute", f"source {source}: {exc}")

        g8_groups = []
        for episode_rows in sample_by_episode.values():
            ordered = sorted(episode_rows, key=lambda row: row.get("sample_id", -1))
            if len(ordered) == 16:
                try:
                    g8_groups.extend(
                        (summarize_group(ordered[:8]), summarize_group(ordered[8:]))
                    )
                except ValueError as exc:
                    add_error("g8_sensitivity", f"cannot recompute G8 subgroup: {exc}")
        if g8_groups:
            mixed = sum(row["mixed_QA_reward_group"] for row in g8_groups) / len(g8_groups)
            unique = [row["number_unique_selected_sets"] for row in g8_groups]
            median_unique = float(torch.tensor(unique, dtype=torch.float64).median().item())
            # torch.median selects the lower middle for even N; Python's median
            # is the declared statistic, so correct that explicitly.
            unique_sorted = sorted(unique)
            middle = len(unique_sorted) // 2
            if len(unique_sorted) % 2:
                median_unique = float(unique_sorted[middle])
            else:
                median_unique = (unique_sorted[middle - 1] + unique_sorted[middle]) / 2
            expected_gate = classify_gate_b0(mixed, median_unique)
            observed_g8 = summary.get("g8_sensitivity", {})
            if not _close(observed_g8.get("mixed_QA_reward_groups_fraction"), mixed) or not _close(
                observed_g8.get("median_unique_selected_sets"), median_unique
            ):
                add_error("g8_sensitivity", "G8 sensitivity statistics do not recompute")
            _subset_equal(
                expected_gate,
                observed_g8.get("gate_if_G8_thresholds_were_applied"),
                "summary.g8_sensitivity.gate",
                add_error,
            )

    artifacts = summary.get("artifacts", {})
    for name in (
        "samples.jsonl",
        "groups.jsonl",
        "references.jsonl",
        "temperature_calibration.json",
        "split_manifest.json",
    ):
        if artifacts.get(name) != file_sha256(root / name):
            add_error("artifact_hash", f"summary hash for {name} is wrong")

    if suppressed_errors:
        warnings.append(
            {
                "code": "suppressed_errors",
                "message": f"{suppressed_errors} additional errors were suppressed",
            }
        )
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "profile": profile,
        "run_dir": str(root),
        "details": {
            "manifest_sha256": manifest_sha,
            "resolved_model_commit": config.get("resolved_model_commit"),
            "selected_temperature": config.get("selected_temperature"),
            "episodes": len(observed_ids),
            "samples": len(samples),
            "groups": len(groups),
            "references": len(references),
            "gate_b0": summary.get("gate_b0", {}).get("status"),
        },
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument(
        "--profile", choices=("dry-run", "smoke", "formal"), default="formal"
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = validate_run(
        args.run_dir,
        profile=args.profile,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_model_revision=args.expected_model_revision,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
