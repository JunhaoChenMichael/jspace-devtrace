from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory_rl.modeling import disable_dropout  # noqa: E402


class DropoutFixture(torch.nn.Module):
    def __init__(self, active_probability: float) -> None:
        super().__init__()
        self.active = torch.nn.Dropout(active_probability)
        self.already_zero = torch.nn.Dropout(0.0)
        self.nested = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2))
        self.config = SimpleNamespace(
            attention_dropout=0.15,
            hidden_dropout=0.0,
            unrelated_probability=0.9,
        )


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {row["name"]: row for row in rows}


def test_disable_dropout_zeroes_values_and_records_independent_post_scan() -> None:
    model = DropoutFixture(active_probability=0.4)

    audit = disable_dropout(model)

    assert model.active.p == 0.0
    assert model.already_zero.p == 0.0
    assert model.nested[1].p == 0.0
    assert model.config.attention_dropout == 0.0
    assert model.config.hidden_dropout == 0.0
    assert model.config.unrelated_probability == 0.9

    # Keep the original summary fields stable for existing artifact readers.
    assert audit["dropout_modules_zeroed"] == 2
    assert audit["config_fields_zeroed"] == ["attention_dropout"]

    modules = _by_name(audit["dropout_modules_found"])
    assert modules["active"] == {
        "name": "active",
        "type": "Dropout",
        "before": 0.4,
        "after": 0.0,
        "modified": True,
    }
    assert modules["already_zero"]["before"] == 0.0
    assert modules["already_zero"]["modified"] is False
    assert audit["dropout_modules_modified"] == ["active", "nested.1"]

    config = _by_name(audit["config_fields_found"])
    assert config["attention_dropout"]["modified"] is True
    assert config["hidden_dropout"] == {
        "name": "hidden_dropout",
        "before": 0.0,
        "after": 0.0,
        "modified": False,
    }
    assert audit["config_fields_modified"] == ["attention_dropout"]
    assert audit["remaining_nonzero"] == []
    assert audit["postcondition_satisfied"] is True

    # dropout_audit.json must never need the trainer's custom JSON encoder.
    json.dumps(audit, allow_nan=False)


def test_disable_dropout_proves_postcondition_when_everything_was_already_zero() -> None:
    model = DropoutFixture(active_probability=0.0)
    model.nested[1].p = 0.0
    model.config.attention_dropout = 0.0

    audit = disable_dropout(model)

    assert audit["dropout_modules_zeroed"] == 0
    assert audit["config_fields_zeroed"] == []
    assert len(audit["dropout_modules_found"]) == 3
    assert all(not row["modified"] for row in audit["dropout_modules_found"])
    assert all(not row["modified"] for row in audit["config_fields_found"])
    assert audit["remaining_nonzero"] == []
    assert audit["postcondition_satisfied"] is True
    json.dumps(audit, allow_nan=False)
