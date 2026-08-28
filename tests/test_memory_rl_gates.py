from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.memory_rl_gates import analyze_gates


SCRIPT = REPO_ROOT / "src" / "analysis" / "memory_rl_gates.py"


def _condition(auc: float | None, qa: float | None) -> dict:
    return {
        "classification": {"pooled_auc": auc, "n_episodes": 20},
        "qa": {"2": {"accuracy": qa, "n_episodes": 20}},
    }


def _payload() -> dict:
    conditions = {
        "original": _condition(0.45, 0.50),
        "sft_seed0": _condition(0.60, 0.51),
        "sft_seed1": _condition(0.62, 0.52),
        "sft_seed2": _condition(0.64, 0.53),
        "rl_w_seed0": _condition(0.64, 0.51),
        "rl_w_seed1": _condition(0.66, 0.52),
        "rl_w_seed2": _condition(0.68, 0.53),
        "rl_qa_seed0": _condition(0.55, 0.55),
        "rl_qa_seed1": _condition(0.56, 0.56),
        "rl_qa_seed2": _condition(0.57, 0.57),
        "hybrid_seed0": _condition(0.66, 0.59),
        "hybrid_seed1": _condition(0.67, 0.60),
        "hybrid_seed2": _condition(0.68, 0.61),
        "workspace": _condition(0.65, 0.54),
        "oracle": _condition(1.0, 0.80),
    }
    adapters = {
        name: f"/adapters/{name}"
        for name in conditions
        if name not in {"original", "workspace", "oracle"}
    }
    comparisons = []
    for index, name in enumerate(adapters):
        comparisons.append(
            {
                "base": "original",
                "adapter": name,
                "adapter_minus_base_accuracy": -0.01 + index * 0.001,
            }
        )
    return {
        "schema_version": 1,
        "config": {"adapters": adapters, "budgets": [2]},
        "metrics": {
            "by_spec": {
                "decoupled": {
                    "n_episodes": 20,
                    "conditions": conditions,
                }
            }
        },
        "no_harm": {
            "skipped": False,
            "summary": {
                "by_spec": {
                    "decoupled": {
                        "comparisons": comparisons,
                    }
                }
            },
        },
    }


def _analyze(payload: dict) -> dict:
    return analyze_gates(
        payload,
        sft_patterns=("sft_seed*",),
        rl_w_patterns=("rl_w_seed*",),
        rl_qa_patterns=("rl_qa_seed*",),
        hybrid_patterns=("hybrid_seed*",),
    )


def test_seed_aggregation_sample_std_no_harm_and_all_passing_gates() -> None:
    report = _analyze(_payload())

    sft_auc = report["families"]["sft"]["pooled_auc"]
    assert sft_auc["individual"] == {
        "sft_seed0": 0.60,
        "sft_seed1": 0.62,
        "sft_seed2": 0.64,
    }
    assert sft_auc["mean"] == pytest.approx(0.62)
    assert sft_auc["sample_std"] == pytest.approx(0.02)
    assert sft_auc["n"] == sft_auc["n_matched"] == 3

    no_harm = report["families"]["sft"][
        "no_harm_adapter_minus_base_accuracy"
    ]
    assert no_harm["n"] == 3
    assert no_harm["individual"]["sft_seed0"] == pytest.approx(-0.01)

    assert report["gates"]["A"]["status"] == "pass"
    assert report["gates"]["A"]["delta"] == pytest.approx(0.04)
    assert report["gates"]["B"]["status"] == "pass"
    assert report["gates"]["B"]["delta"] == pytest.approx(0.06)
    assert report["gates"]["C"]["status"] == "pass"
    assert report["gates"]["C"]["beats_rl_w"] is True
    assert report["gates"]["C"]["beats_rl_qa"] is True


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.63, 0.64, 0.65), "tie"),
        ((0.56, 0.57, 0.58), "worse"),
    ],
)
def test_gate_a_reports_tie_and_worse(values: tuple[float, ...], expected: str) -> None:
    payload = _payload()
    conditions = payload["metrics"]["by_spec"]["decoupled"]["conditions"]
    for seed, value in enumerate(values):
        conditions[f"rl_w_seed{seed}"]["classification"]["pooled_auc"] = value
    assert _analyze(payload)["gates"]["A"]["status"] == expected


def test_missing_one_seed_metric_makes_relevant_gates_insufficient() -> None:
    payload = _payload()
    condition = payload["metrics"]["by_spec"]["decoupled"]["conditions"][
        "rl_qa_seed1"
    ]
    condition["qa"]["2"]["accuracy"] = None
    report = _analyze(payload)

    summary = report["families"]["rl_qa"]["qa"]
    assert summary["n"] == 2
    assert summary["n_matched"] == 3
    assert summary["missing_conditions"] == ["rl_qa_seed1"]
    assert report["gates"]["A"]["status"] == "pass"
    assert report["gates"]["B"]["status"] == "insufficient-data"
    assert report["gates"]["C"]["status"] == "insufficient-data"


def test_missing_source_returns_insufficient_data_instead_of_guessing() -> None:
    report = analyze_gates(_payload(), source="compositional")
    assert report["source_found"] is False
    assert all(
        gate["status"] == "insufficient-data" for gate in report["gates"].values()
    )
    assert all(
        gate["source_missing"] == "compositional"
        for gate in report["gates"].values()
    )


def test_single_seed_has_null_sample_std_and_overlapping_globs_are_rejected() -> None:
    report = analyze_gates(
        _payload(),
        sft_patterns=("sft_seed0",),
        rl_w_patterns=("rl_w_seed0",),
        rl_qa_patterns=("rl_qa_seed0",),
        hybrid_patterns=("hybrid_seed0",),
    )
    assert report["families"]["sft"]["pooled_auc"]["sample_std"] is None

    with pytest.raises(ValueError, match="overlap"):
        analyze_gates(
            _payload(),
            sft_patterns=("*_seed0",),
            rl_w_patterns=("rl_w_seed0",),
            rl_qa_patterns=("rl_qa_seed*",),
            hybrid_patterns=("hybrid_seed*",),
        )


def test_cli_reads_unified_json_globs_conditions_and_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "unified.json"
    output_path = tmp_path / "gates.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_path),
            "--sft",
            "sft_seed*",
            "--rl-w",
            "rl_w_seed*",
            "--rl-qa",
            "rl_qa_seed*",
            "--hybrid",
            "hybrid_seed*",
            "--source",
            "decoupled",
            "--budget",
            "2",
            "--out",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    stdout = json.loads(process.stdout)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout == written
    assert written["input"] == str(input_path)
    assert written["gates"]["A"]["status"] == "pass"
