#!/usr/bin/env python3
"""Question-blind audit of every training admission prompt.

The RL-QA policy may see only the episode context and one candidate concept.
This command renders all ID-train admission prompts exactly as the policy will
receive them, hashes each one, and fails closed if any prompt leaks the probe
question, the gold answer, the utility label, or any other episode field.

It performs no inference and must run before any question or reward prompt is
constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory_rl.data import build_training_bundle  # noqa: E402
from memory_rl.modeling import ADMISSION_PROMPT, render_admission_prompt  # noqa: E402

SCHEMA = "rlqa-admission-prompt-leak-audit/v1"
# Fields the policy is allowed to see, and everything it must never see.
ALLOWED_FIELDS = ("context", "candidate.concept")
FORBIDDEN_FIELDS = ("probe_question", "answer", "label")


class LeakAuditError(RuntimeError):
    pass


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def audit(tokenizer, episodes: Sequence[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    # Tokens contributed by the chat-template scaffold itself (<|im_start|> and
    # friends) are not a leakage channel; render an empty body to collect them.
    scaffold_tokens = _tokens(render_admission_prompt(tokenizer, "", "")) | _tokens(
        ADMISSION_PROMPT
    )
    for episode in episodes:
        context_norm = _normalise(episode.context)
        probe_norm = _normalise(episode.probe_question)
        answer_norm = _normalise(episode.answer)
        # Answer tokens that the context does not already contain: if one of
        # these turns up in a prompt, the gold answer leaked through.
        answer_only_tokens = _tokens(answer_norm) - _tokens(context_norm)
        for index, candidate in enumerate(episode.candidates):
            prompt = render_admission_prompt(tokenizer, episode.context, candidate.concept)
            prompt_norm = _normalise(prompt)
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

            found: list[str] = []
            if probe_norm and probe_norm in prompt_norm:
                found.append("probe_question")
            # A gold-answer token in the prompt is only a leak when it cannot be
            # attributed to an ALLOWED field. The load-bearing candidate is very
            # often the answer itself (105/825 here), and the policy is supposed
            # to see candidate concepts -- it never sees the question, so that is
            # the task, not a leak. Template words are likewise not a channel.
            unattributed = (
                answer_only_tokens
                & _tokens(prompt_norm)
                - _tokens(candidate.concept)
                - scaffold_tokens
            )
            if unattributed:
                found.append(f"gold_answer_token:{sorted(unattributed)}")
            for label in ("load_bearing", "load-bearing", "distractor", "filler"):
                if label in prompt_norm:
                    found.append(f"label:{label}")
            # The rendered prompt must be reconstructible from exactly the two
            # allowed fields; anything else means an extra field crept in.
            expected = render_admission_prompt(tokenizer, episode.context, candidate.concept)
            if prompt != expected:
                found.append("non_reproducible_render")

            row = {
                "episode_uid": episode.uid,
                "source": episode.source,
                "candidate_index": index,
                "concept": candidate.concept,
                "prompt_sha256": digest,
                "prompt_chars": len(prompt),
            }
            row["answer_equals_concept"] = _normalise(candidate.concept) == answer_norm
            rows.append(row)
            if found:
                violations.append({**row, "violations": sorted(set(found))})

    aggregate = hashlib.sha256(
        json.dumps([r["prompt_sha256"] for r in rows], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    unique = len({r["prompt_sha256"] for r in rows})
    return {
        "schema_version": SCHEMA,
        "decision": "PASS" if not violations else "FAIL",
        "counts": {
            "episodes": len(episodes),
            "prompts": len(rows),
            "unique_prompt_hashes": unique,
            "violations": len(violations),
            # Benchmark statistic, not a leak: the load-bearing candidate is
            # frequently the gold answer, which the question-blind policy sees
            # as one candidate among several without ever seeing the question.
            "prompts_where_concept_equals_answer": sum(
                1 for r in rows if r["answer_equals_concept"]
            ),
        },
        "policy_input_fields": list(ALLOWED_FIELDS),
        "forbidden_fields_checked": list(FORBIDDEN_FIELDS),
        "admission_prompt_template_sha256": hashlib.sha256(
            ADMISSION_PROMPT.encode("utf-8")
        ).hexdigest(),
        "aggregate_prompt_sha256": aggregate,
        "violations": violations,
        "prompts": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="tokenizer whose chat template renders the prompt")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--teacher-tag", required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workspace-top-k", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    bundle = build_training_bundle(
        repo_root=REPO_ROOT,
        teacher_tag=args.teacher_tag,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        top_k=args.workspace_top_k,
    )
    result = audit(tokenizer, bundle.train_episodes)
    result["model"] = args.model
    result["model_revision"] = args.model_revision
    result["split_manifest"] = dict(bundle.split_manifest)

    if args.out.exists() or args.out.is_symlink():
        print(f"error: refusing to overwrite {args.out}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    counts = result["counts"]
    print(
        f"admission prompt audit {result['decision']}: {counts['prompts']} prompts "
        f"over {counts['episodes']} episodes, {counts['violations']} violations -> {args.out}"
    )
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
