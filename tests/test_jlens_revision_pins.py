from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jlens import WorkspaceLens


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    init_kwargs = {"_commit_hash": "resolved-tokenizer"}


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.output = torch.nn.Linear(4, 16, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=4,
            vocab_size=16,
            _commit_hash="resolved-model",
        )

    def get_output_embeddings(self):
        return self.output


def test_workspace_lens_forwards_independent_revision_pins(monkeypatch) -> None:
    calls = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["tokenizer"] = (name, kwargs)
            return _Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["model"] = (name, kwargs)
            return _Model()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    lens = WorkspaceLens(
        "fake/model",
        device="cpu",
        model_revision="model-pin",
        tokenizer_revision="tokenizer-pin",
    )

    assert calls["model"][1]["revision"] == "model-pin"
    assert calls["tokenizer"][1]["revision"] == "tokenizer-pin"
    assert lens.model_revision_resolved == "resolved-model"
    assert lens.tokenizer_revision_resolved == "resolved-tokenizer"


def test_workspace_lens_defaults_tokenizer_to_model_revision(monkeypatch) -> None:
    calls = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            calls["tokenizer"] = kwargs
            return _Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            return _Model()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    WorkspaceLens("fake/model", device="cpu", model_revision="shared-pin")

    assert calls["tokenizer"]["revision"] == "shared-pin"
