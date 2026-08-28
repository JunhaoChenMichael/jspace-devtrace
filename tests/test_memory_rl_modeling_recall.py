from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TRAIN_SCRIPT = SRC_ROOT / "experiments" / "train_memory_rl.py"
sys.path.insert(0, str(SRC_ROOT))

from memory_rl.modeling import (  # noqa: E402
    binary_action_logits,
    render_admission_prompt,
    selection_logits,
    yes_no_token_ids,
)
from memory_rl.recall import FrozenRecall, grade_answer  # noqa: E402
import experiments.train_memory_rl as train_memory_rl  # noqa: E402
from experiments.train_memory_rl import (  # noqa: E402
    evaluate_selector,
    selector_candidate_prompts,
)


class FakeTokenizer:
    """Small tokenizer surface sufficient for the admission-policy helpers."""

    chat_template = None
    padding_side = "right"
    truncation_side = "right"

    _ACTION_IDS = {
        "no": 1,
        " no": 2,
        "No": 1,
        " No": 2,
        "NO": 1,
        " NO": 2,
        "yes": 3,
        " yes": 4,
        "Yes": 3,
        " Yes": 4,
        "YES": 3,
        " YES": 4,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._ACTION_IDS[text]] if text in self._ACTION_IDS else [9]

    def __call__(
        self,
        prompts,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding and truncation and max_length > 0 and return_tensors == "pt"
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        batch = len(prompts)
        return {
            "input_ids": torch.arange(1, 4).repeat(batch, 1),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }


class FakeModel:
    def __init__(self, final_logits: torch.Tensor):
        self.final_logits = final_logits

    def __call__(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        batch, sequence = input_ids.shape
        assert batch == self.final_logits.shape[0]
        logits = torch.zeros(batch, sequence, self.final_logits.shape[-1])
        logits[:, -1] = self.final_logits
        return SimpleNamespace(logits=logits)


def test_yes_no_ids_and_binary_logits_aggregate_all_token_variants():
    tokenizer = FakeTokenizer()
    token_ids = yes_no_token_ids(tokenizer)
    assert token_ids == ([1, 2], [3, 4])  # action order is [No, Yes]

    # Each action logit must be logsumexp over both surface variants, not max or
    # one arbitrarily chosen spelling.
    final_logits = torch.zeros(2, 6)
    final_logits[0, [1, 2]] = torch.tensor([math.log(2.0), math.log(3.0)])
    final_logits[0, [3, 4]] = torch.tensor([math.log(5.0), math.log(7.0)])
    final_logits[1, [1, 2]] = torch.tensor([math.log(11.0), math.log(13.0)])
    final_logits[1, [3, 4]] = torch.tensor([math.log(17.0), math.log(19.0)])

    actual = binary_action_logits(
        FakeModel(final_logits),
        tokenizer,
        ["first prompt", "second prompt"],
        token_ids,
        "cpu",
        max_length=32,
    )
    expected = torch.tensor(
        [[math.log(5.0), math.log(12.0)], [math.log(24.0), math.log(36.0)]]
    )
    torch.testing.assert_close(actual, expected)
    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"


def test_selection_logit_is_yes_minus_no_and_shift_invariant():
    actions = torch.tensor([[1.0, 3.0], [4.0, -2.0]])
    expected = torch.tensor([2.0, -6.0])
    torch.testing.assert_close(selection_logits(actions), expected)
    torch.testing.assert_close(selection_logits(actions + 123.0), expected)


def test_admission_prompt_contains_context_and_candidate_but_not_future_probe():
    probe = "SECRET_FUTURE_PROBE_DO_NOT_LEAK"
    prompt = render_admission_prompt(
        FakeTokenizer(),
        "Mina trained near the river before sunrise.",
        "running",
    )
    assert "Mina trained near the river" in prompt
    assert '"running"' in prompt
    assert probe not in prompt
    assert prompt.endswith("Answer:")


def test_selector_training_fails_closed_if_probe_enters_real_policy_prompt():
    episode = SimpleNamespace(
        uid="sentinel-episode",
        context="ordinary context",
        probe_question="SECRET_FUTURE_PROBE_DO_NOT_LEAK",
        candidates=[SimpleNamespace(concept="running")],
    )
    prompts = selector_candidate_prompts(FakeTokenizer(), episode)
    assert episode.probe_question not in prompts[0]

    episode.context = f"ordinary context {episode.probe_question}"
    with pytest.raises(RuntimeError, match="probe leaked"):
        selector_candidate_prompts(FakeTokenizer(), episode)


def test_selector_validation_records_adapter_enabled_yes_probability(monkeypatch):
    class Policy:
        training = True

        def eval(self):
            self.training = False

        def train(self, value=True):
            self.training = value

    candidates = [
        SimpleNamespace(
            uid=f"explicit:episode:000000:candidate:{index:03d}",
            episode_uid="explicit:episode:000000",
            concept=f"concept-{index}",
            label="load_bearing" if index == 1 else "distractor",
            w_ref=[0.1, 0.9, 0.2][index],
            w_percentile=[0.0, 1.0, 0.5][index],
        )
        for index in range(3)
    ]
    episode = SimpleNamespace(
        uid="explicit:episode:000000",
        source="explicit",
        context="probe-blind context",
        probe_question="Which concept matters?",
        candidates=candidates,
    )
    expected_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]])

    def fake_logits(model, tokenizer, prompts, token_ids, device, max_length):
        del model, tokenizer, token_ids, device, max_length
        assert len(prompts) == 3
        assert all(episode.probe_question not in prompt for prompt in prompts)
        return expected_logits

    monkeypatch.setattr(train_memory_rl, "binary_action_logits", fake_logits)
    recall = SimpleNamespace(
        evaluate_sets=lambda observed_episode, sets: [
            {
                "correct": True,
                "selected_concepts": [
                    observed_episode.candidates[index].concept for index in sets[0]
                ],
            }
        ]
    )
    bundle = SimpleNamespace(
        model=Policy(),
        tokenizer=FakeTokenizer(),
        action_token_ids=([1, 2], [3, 4]),
        device="cpu",
    )

    metrics, rows = evaluate_selector(
        bundle,
        [episode],
        recall,
        budget=2,
        max_length=32,
        reporter_bootstrap_samples=0,
    )

    expected_yes = torch.softmax(expected_logits, dim=-1)[:, 1].tolist()
    assert [row["v_rl"] for row in rows] == pytest.approx(expected_yes)
    assert [row["y_utility"] for row in rows] == [0, 1, 0]
    assert [row["w_ref"] for row in rows] == [0.1, 0.9, 0.2]
    assert metrics["verbal_auc"] == 1.0
    assert metrics["reporter_correlations"]["definitions"]["v_rl"].startswith(
        "adapter-enabled P(Yes)"
    )
    assert bundle.model.training is True


@pytest.mark.parametrize(
    ("output", "gold", "expected"),
    [
        ("The answer is: BRUSSELS!", "Brussels", True),
        ("She probably spoke French.", "French language", True),
        ("The answer is Madrid.", "Brussels", False),
        ("Perhaps.", "to be", False),
    ],
)
def test_grade_answer_matches_existing_downstream_grader(output, gold, expected):
    assert grade_answer(output, gold) is expected


def test_frozen_recall_canonicalizes_selected_concepts_before_prompting():
    candidates = [
        SimpleNamespace(concept="alpha", label="distractor"),
        SimpleNamespace(concept="beta", label="distractor"),
        SimpleNamespace(concept="gamma", label="load_bearing"),
    ]
    episode = SimpleNamespace(
        candidates=candidates,
        probe_question="Which concept is load-bearing?",
        answer="gamma",
    )
    recall = FrozenRecall(model=object(), tokenizer=object(), device="cpu")
    seen_prompts: list[str] = []

    def fake_generate(prompts: list[str]) -> list[str]:
        seen_prompts.extend(prompts)
        return ["gamma", "not known"]

    recall.generate = fake_generate
    records = recall.evaluate_sets(episode, [[2, 0, 2], [1]])

    assert records[0]["selected_concepts"] == ["alpha", "gamma"]
    assert records[1]["selected_concepts"] == ["beta"]
    assert records[0]["correct"] is True
    assert records[1]["correct"] is False
    assert "alpha, gamma" in seen_prompts[0]
    assert "gamma, alpha" not in seen_prompts[0]
    assert "Which concept is load-bearing?" in seen_prompts[0]


def _write_synthetic_training_source(directory: Path) -> tuple[Path, Path]:
    battery = []
    results = []
    for episode_index in range(5):
        items = [
            {"concept": f"bridge-{episode_index}", "label": "load_bearing"},
            {"concept": f"noise-{episode_index}", "label": "distractor"},
            {"concept": f"filler-{episode_index}", "label": "filler"},
        ]
        battery.append(
            {
                "context": f"synthetic context {episode_index}",
                "probe_question": f"synthetic question {episode_index}?",
                "answer": f"answer-{episode_index}",
                "items": items,
            }
        )
        for candidate_index, item in enumerate(items):
            results.append(
                {
                    "episode": episode_index,
                    "concept": item["concept"],
                    "label": item["label"],
                    "W_rr": [0.9, 0.3, 0.1][candidate_index],
                    "V": [0.8, 0.4, 0.2][candidate_index],
                }
            )

    battery_path = directory / "battery.json"
    results_path = directory / "results.json"
    battery_path.write_text(json.dumps(battery), encoding="utf-8")
    results_path.write_text(json.dumps(results), encoding="utf-8")
    return results_path, battery_path


def _run_training_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    # If a dry-run regresses and reaches a model loader, fail locally instead of
    # downloading a checkpoint during a unit test.
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_train_cli_dry_run_validates_data_without_loading_a_model(tmp_path: Path):
    results, battery = _write_synthetic_training_source(tmp_path)
    output = tmp_path / "successful-run"
    process = _run_training_cli(
        "--mode",
        "rl-w",
        "--model",
        "local/fake-policy",
        "--teacher-tag",
        "synthetic",
        "--workspace-teacher-model",
        "local/fake-policy",
        "--train-spec",
        f"explicit={results}::{battery}",
        "--out-dir",
        str(output),
        "--dry-run",
    )
    assert process.returncode == 0, process.stderr
    assert "model not loaded" in process.stdout
    assert (output / "split_manifest.json").is_file()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert config["workspace_teacher_model"] == "local/fake-policy"
    assert config["teacher_matches_policy_reference"] is True
    assert config["effective_advantage_normalization"] == "center"


def test_train_cli_rejects_workspace_teacher_mismatch_before_model_loading(tmp_path: Path):
    process = _run_training_cli(
        "--mode",
        "rl-w",
        "--model",
        "local/policy",
        "--workspace-teacher-model",
        "local/different-teacher",
        "--out-dir",
        str(tmp_path / "mismatch-run"),
        "--dry-run",
    )
    assert process.returncode != 0
    assert "does not match policy/reference checkpoint" in process.stderr
    assert "model not loaded" not in process.stdout


def test_train_cli_rejects_ood_train_spec_before_reading_it(tmp_path: Path):
    # Deliberately nonexistent paths demonstrate that source allow-listing runs
    # before file IO and, in particular, before any model loader.
    process = _run_training_cli(
        "--mode",
        "rl-w",
        "--model",
        "local/fake-policy",
        "--workspace-teacher-model",
        "local/fake-policy",
        "--train-spec",
        f"decoupled={tmp_path / 'missing-results.json'}::{tmp_path / 'missing-battery.json'}",
        "--out-dir",
        str(tmp_path / "ood-run"),
        "--dry-run",
    )
    assert process.returncode != 0
    assert "held-out 'decoupled'" in process.stderr
    assert "model not loaded" not in process.stdout
