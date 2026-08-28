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
    assert metadata["schema_version"] == "workspace_measurement_metadata.v2"
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
