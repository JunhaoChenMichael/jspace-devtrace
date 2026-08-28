from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import statistics
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import stage_b_rlqa_workspace_noharm as workspace_noharm


PROTOCOL = workspace_noharm.Protocol(
    source="decoupled",
    n_episodes=4,
    n_rows=12,
    seeds=(0, 1, 2),
    bootstrap_samples=64,
    bootstrap_seed=0,
    alert_threshold=-0.03,
)


def _rows(scores: tuple[float, float, float], *, verbal: bool = False) -> list[dict]:
    rows = []
    for episode in range(PROTOCOL.n_episodes):
        for candidate, label in enumerate(
            ("load_bearing", "distractor", "filler")
        ):
            row = {
                "episode": episode,
                "concept": f"concept-{episode}-{candidate}",
                "label": label,
                "W_end": 0.01 * (candidate + 1),
                "W_max": 0.0,
                "W_rr": scores[candidate] + episode * 0.0001,
            }
            if verbal:
                row["V"] = 0.5
                row["V_raw"] = 0.4
            rows.append(row)
    return rows


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, allow_nan=True) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    paths = tuple(tmp_path / name for name in ("base.json", "s0.json", "s1.json", "s2.json"))
    _write(paths[0], _rows((0.8, 0.2, 0.1)))
    _write(paths[1], _rows((0.8, 0.2, 0.1), verbal=True))
    _write(paths[2], _rows((0.1, 0.9, 0.8)))
    _write(paths[3], _rows((0.6, 0.5, 0.4)))
    return paths


def test_analysis_strict_join_shared_bootstrap_alerts_and_seed_summary(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    result = workspace_noharm.analyze_files(*paths, protocol=PROTOCOL)

    assert result["join"]["status"] == "exact-match"
    assert result["join"]["n_rows"] == 12
    assert result["join"]["n_episodes"] == 4
    assert result["join"]["label_counts"] == {
        "distractor": 4,
        "filler": 4,
        "load_bearing": 4,
    }
    assert result["base"]["workspace_w_rr_pooled_auc"] == 1.0

    assert result["seeds"]["0"]["workspace_w_rr_pooled_auc"] == 1.0
    assert result["seeds"]["0"]["paired_vs_base"]["estimate"] == 0.0
    assert result["seeds"]["0"]["paired_vs_base"]["ci_95"] == [0.0, 0.0]
    assert result["seeds"]["0"]["no_harm"]["alert"] is False

    assert result["seeds"]["1"]["workspace_w_rr_pooled_auc"] == 0.0
    assert result["seeds"]["1"]["paired_vs_base"]["estimate"] == -1.0
    assert result["seeds"]["1"]["paired_vs_base"]["ci_95"] == [-1.0, -1.0]
    assert result["seeds"]["1"]["no_harm"]["alert"] is True

    for seed in ("0", "1", "2"):
        assert (
            result["seeds"][seed]["paired_vs_base"][
                "bootstrap_samples_effective"
            ]
            == 64
        )
    bootstrap = result["protocol"]["bootstrap"]
    assert bootstrap["seed"] == 0
    assert bootstrap["rng"] == "numpy.random.default_rng / PCG64"
    assert bootstrap["shared_draws_across_seeds"] is True

    auc_summary = result["summary"]["workspace_w_rr_pooled_auc"]
    assert auc_summary["individual"] == {"0": 1.0, "1": 0.0, "2": 1.0}
    assert auc_summary["mean"] == pytest.approx(2.0 / 3.0)
    assert auc_summary["sample_std"] == pytest.approx(
        statistics.stdev([1.0, 0.0, 1.0])
    )
    assert result["summary"]["alerts"] == {
        "individual": {"0": False, "1": True, "2": False},
        "any_alert": True,
        "n_alerts": 1,
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows.pop(), "row count mismatch"),
        (
            lambda rows: rows[1].__setitem__("concept", rows[0]["concept"]),
            "duplicate (episode, concept) UID",
        ),
        (lambda rows: rows[1].__setitem__("label", "filler"), "label mismatch"),
        (lambda rows: rows[1].__setitem__("episode", 99), "episode IDs mismatch"),
        (lambda rows: rows.reverse(), "UID/order mismatch"),
        (
            lambda rows: rows[1].__setitem__("W_max", 0.1),
            "must be exactly 0",
        ),
        (
            lambda rows: rows[1].__setitem__("unexpected", 1),
            "schema mismatch",
        ),
        (lambda rows: rows[1].__setitem__("W_rr", math.nan), "non-finite"),
    ],
)
def test_analysis_rejects_incomplete_or_misaligned_seed_raw(
    mutator, message: str, tmp_path: Path
) -> None:
    paths = _inputs(tmp_path)
    corrupted = json.loads(paths[2].read_text(encoding="utf-8"))
    mutator(corrupted)
    _write(paths[2], corrupted)

    with pytest.raises(
        workspace_noharm.WorkspaceNoHarmError, match=re.escape(message)
    ):
        workspace_noharm.analyze_files(*paths, protocol=PROTOCOL)


def test_analysis_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"key": 1, "key": 2}\n', encoding="utf-8")

    with pytest.raises(
        workspace_noharm.WorkspaceNoHarmError, match="duplicate JSON key"
    ):
        workspace_noharm._strict_load(path)


def test_formal_lock_can_reject_the_wrong_base_hash(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    protocol = workspace_noharm.Protocol(
        source="decoupled",
        n_episodes=4,
        n_rows=12,
        seeds=(0, 1, 2),
        bootstrap_samples=8,
        bootstrap_seed=0,
        alert_threshold=-0.03,
        locked_base_sha256="0" * 64,
    )

    with pytest.raises(
        workspace_noharm.WorkspaceNoHarmError, match="base SHA-256 mismatch"
    ):
        workspace_noharm.analyze_files(*paths, protocol=protocol)


def test_validate_recomputes_complete_artifact_and_detects_tampering(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    result = workspace_noharm.analyze_files(*paths, protocol=PROTOCOL)
    analysis_path = tmp_path / "analysis.json"
    _write(analysis_path, result)

    validation = workspace_noharm.validate_analysis_files(
        *paths, analysis_path, protocol=PROTOCOL
    )
    assert validation["status"] == "pass"
    assert validation["errors"] == []

    tampered = copy.deepcopy(result)
    tampered["seeds"]["2"]["paired_vs_base"]["estimate"] = 123.0
    _write(analysis_path, tampered)
    failed = workspace_noharm.validate_analysis_files(
        *paths, analysis_path, protocol=PROTOCOL
    )
    assert failed["status"] == "fail"
    assert "paired_vs_base.estimate" in failed["errors"][0]


def test_output_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    output.write_text("sealed\n", encoding="utf-8")

    with pytest.raises(
        workspace_noharm.WorkspaceNoHarmError, match="refusing to overwrite"
    ):
        workspace_noharm._write_json_exclusive(output, {"status": "pass"})

    assert output.read_text(encoding="utf-8") == "sealed\n"
