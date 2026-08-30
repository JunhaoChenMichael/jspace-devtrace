from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import measure


class _FakeTokenizer:
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    name_or_path = "fake-tokenizer"
    init_kwargs = {"_commit_hash": "tokenizer-commit", "chat_template": chat_template}
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [sum(text.encode("utf-8")) % 101 + 1]

    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"

    def decode(self, token_ids):
        return f"decoded-{token_ids[0]}"


class _FakeLens:
    def __init__(self, model, **kwargs):
        self.constructor = {"model": model, **kwargs}
        self.tok = _FakeTokenizer()
        self.device = "cpu"
        self.n_layers = 2
        self.final_norm = torch.nn.LayerNorm(2)
        self.unembed = torch.nn.Linear(2, 8)
        self.model_revision_resolved = "model-commit"
        self.tokenizer_revision_resolved = "tokenizer-commit"
        self.tokenizer_revision_effective = kwargs.get("tokenizer_revision") or kwargs.get(
            "model_revision"
        )


def test_measure_pair_writer_cleans_new_partial_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    metadata = Path(f"{raw}.metadata")

    with pytest.raises(TypeError):
        measure.write_output_pair_exclusive(
            [(raw, "complete raw\n"), (metadata, object())]
        )

    assert not raw.exists()
    assert not metadata.exists()


def test_measure_revision_pins_and_complete_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    battery = tmp_path / "battery.json"
    battery.write_text(
        json.dumps(
            [
                {
                    "context": "context",
                    "items": [
                        {"concept": "alpha", "label": "load_bearing"},
                        {"concept": "beta", "label": "distractor"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "raw.json"
    constructed = []

    def fake_lens(*args, **kwargs):
        lens = _FakeLens(*args, **kwargs)
        constructed.append(lens)
        return lens

    def fake_workspace(_lens, _context, concept_ids, end_only=False):
        del end_only
        return (
            {token_id: 0.1 for token_id in concept_ids},
            {token_id: 0.2 for token_id in concept_ids},
            {token_id: 0.25 for token_id in concept_ids},
        )

    monkeypatch.setattr(measure, "WorkspaceLens", fake_lens)
    monkeypatch.setattr(measure, "workspace_salience", fake_workspace)
    monkeypatch.setattr(measure, "verbal_salience", lambda *_args: 0.75)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure.py",
            "--model",
            "fake/model",
            "--model-revision",
            "model-pin",
            "--tokenizer-revision",
            "tokenizer-pin",
            "--battery",
            str(battery),
            "--out",
            str(output),
            "--device",
            "cpu",
            "--no-verbal-raw",
        ],
    )

    measure.main()

    assert constructed[0].constructor["model_revision"] == "model-pin"
    assert constructed[0].constructor["tokenizer_revision"] == "tokenizer-pin"
    raw_rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["candidate_index"] for row in raw_rows] == [0, 1]
    metadata = json.loads(Path(f"{output}.metadata").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "workspace_measurement_metadata.v3"
    assert metadata["model_revision"] == "model-commit"
    assert metadata["tokenizer_revision"] == "tokenizer-commit"
    assert metadata["chat_template_sha256"] == hashlib.sha256(
        _FakeTokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    assert metadata["yes_token_ids"]
    assert metadata["no_token_ids"]
    assert metadata["counts"]["episodes_evaluated"] == 1
    assert metadata["counts"]["candidates_evaluated"] == 2
    assert metadata["counts"]["candidate_rows_written"] == 2
    assert metadata["runtime"]["gpu"] == {
        "available": False,
        "count": 0,
        "devices": [],
    }
    assert metadata["hashes"]["battery_file_sha256"] == hashlib.sha256(
        battery.read_bytes()
    ).hexdigest()
    assert metadata["hashes"]["raw_output_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_verbal_score_is_the_ratio_not_the_absolute_mass():
    """Regression: the guard epsilon must not dominate a tiny yes/no mass.

    Before the fix, verbal_salience returned py/(py+pn+1e-9) computed on
    full-vocabulary softmax probabilities. Real probes put ~1e-13 of mass on
    the yes/no tokens, so the epsilon swamped the denominator and the function
    returned py*1e9 -- a monotone function of the ABSOLUTE yes probability
    rather than the documented yes-versus-no ratio. On the logits below the old
    form returns about 5.9e-4 while the true ratio is about 0.92.
    """
    import math
    import torch

    from experiments.measure import _yes_vs_no

    logits = torch.full((256,), -50.0)
    logits[0] = 30.0          # some other token holds essentially all the mass
    yes_ids, no_ids = [1, 2], [3, 4]
    logits[yes_ids[0]] = 1.3125
    logits[yes_ids[1]] = -0.7539
    logits[no_ids[0]] = -0.8423
    logits[no_ids[1]] = -4.625

    value = _yes_vs_no(logits, yes_ids, no_ids)

    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    assert value == pytest.approx(float(torch.sigmoid(yes - no)), abs=1e-9)
    assert value > 0.9

    probabilities = torch.softmax(logits, dim=-1)
    py = float(sum(probabilities[i] for i in yes_ids))
    pn = float(sum(probabilities[i] for i in no_ids))
    assert py + pn < 1e-9, "fixture must sit in the regime where the epsilon bit"
    old_form = py / (py + pn + 1e-9)
    assert old_form < 0.01
    assert not math.isclose(old_form, value, abs_tol=0.5)


def test_verbal_score_matches_the_rl_admission_policy_exactly():
    """The metacognitive probe and the RL policy must agree on identical logits.

    They are two implementations of the same quantity; before the fix they
    disagreed by up to 0.98 on real prompts.
    """
    import torch
    import torch.nn.functional as F

    from experiments.measure import _yes_vs_no

    torch.manual_seed(0)
    for _ in range(20):
        logits = torch.randn(512) * 6.0
        yes_ids, no_ids = [7, 11, 13], [17, 19]
        probe = _yes_vs_no(logits, yes_ids, no_ids)
        # the RL path: logsumexp per action, then softmax over [no, yes]
        actions = torch.stack(
            (torch.logsumexp(logits[no_ids], 0), torch.logsumexp(logits[yes_ids], 0))
        )
        policy = float(F.softmax(actions, dim=-1)[1])
        assert probe == pytest.approx(policy, abs=1e-6)


def test_verbal_score_is_invariant_to_the_absolute_probability_mass():
    """Epsilon-independence: shifting all other logits must not move the score.

    Adding a constant to an unrelated token changes how much absolute mass
    lands on yes/no while leaving their relative preference untouched. The
    corrected score must be flat across that shift; the old form was not.
    """
    import torch

    from experiments.measure import _yes_vs_no

    yes_ids, no_ids = [1, 2], [3, 4]
    scores, old_scores = [], []
    for other in (0.0, 10.0, 20.0, 30.0, 40.0):
        logits = torch.full((256,), -50.0)
        logits[0] = other
        logits[yes_ids[0]], logits[yes_ids[1]] = 1.3125, -0.7539
        logits[no_ids[0]], logits[no_ids[1]] = -0.8423, -4.625
        scores.append(_yes_vs_no(logits, yes_ids, no_ids))
        probabilities = torch.softmax(logits, dim=-1)
        py = float(sum(probabilities[i] for i in yes_ids))
        pn = float(sum(probabilities[i] for i in no_ids))
        old_scores.append(py / (py + pn + 1e-9))

    assert max(scores) - min(scores) < 1e-6, "corrected score must not track total mass"
    assert max(old_scores) - min(old_scores) > 0.5, "the old form did track total mass"


def test_verbal_score_token_sets_agree_between_probe_and_policy():
    """The probe's 5 variants and the policy's 6 must not change the ranking.

    The policy adds ' YES' and ' NO'. Measured on real prompts the two sets
    differ by under 1e-5, so a disagreement between the tracks can never be
    blamed on the token set.
    """
    import torch

    from experiments.measure import _yes_vs_no

    torch.manual_seed(3)
    probe_yes, probe_no = [11, 12, 13, 14, 15], [21, 22, 23, 24, 25]
    policy_yes, policy_no = probe_yes + [16], probe_no + [26]
    for _ in range(10):
        logits = torch.randn(64) * 4.0
        # the extra variants are rare continuations, far below the common ones
        logits[16] = logits[probe_yes].min() - 6.0
        logits[26] = logits[probe_no].min() - 6.0
        assert _yes_vs_no(logits, probe_yes, probe_no) == pytest.approx(
            _yes_vs_no(logits, policy_yes, policy_no), abs=1e-3
        )


def test_verbal_score_ranking_is_deterministic_and_orders_by_preference():
    """Same logits give the same score, and a stronger yes ranks higher."""
    import torch

    from experiments.measure import _yes_vs_no

    yes_ids, no_ids = [1], [2]
    previous = -1.0
    for yes_logit in (-4.0, -2.0, 0.0, 2.0, 4.0):
        logits = torch.full((32,), -50.0)
        logits[1], logits[2] = yes_logit, 0.0
        value = _yes_vs_no(logits, yes_ids, no_ids)
        assert value == _yes_vs_no(logits.clone(), yes_ids, no_ids)
        assert value > previous
        previous = value
    assert 0.0 < previous < 1.0
