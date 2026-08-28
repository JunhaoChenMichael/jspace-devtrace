from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import gate_metacog_m0 as m0_gate


def _green_rows() -> list[dict]:
    # Three positive/negative pairs.  Pooled AUC is 3/9=.333 for V and
    # 6/9=.667 for W_rr, both inside the predeclared +/- .05 gate.
    negative_v = [0.2, 0.5, 0.8]
    positive_v = [0.1, 0.4, 0.7]
    negative_w = [0.2, 0.5, 0.8]
    positive_w = [0.3, 0.6, 0.9]
    rows = []
    for episode in range(3):
        rows.extend(
            [
                {
                    "episode": episode,
                    "candidate_index": 0,
                    "concept": f"positive-{episode}",
                    "label": "load_bearing",
                    "V": positive_v[episode],
                    "W_rr": positive_w[episode],
                },
                {
                    "episode": episode,
                    "candidate_index": 1,
                    "concept": f"negative-{episode}",
                    "label": "distractor" if episode % 2 else "filler",
                    "V": negative_v[episode],
                    "W_rr": negative_w[episode],
                },
            ]
        )
    return rows


def _metadata_for(path: Path, row_count: int) -> dict:
    return {
        "schema_version": "workspace_measurement_metadata.v2",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "model-commit",
        "tokenizer_revision": "tokenizer-commit",
        "chat_template_sha256": "a" * 64,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "adapter": None,
        "limit_episodes": 0,
        "end_only": True,
        "verbal_enabled": True,
        "policy_input_includes_probe": False,
        "workspace_readout": {
            "position": "final context token",
            "layer_aggregation": "maximum",
        },
        "yes_token_ids": [1, 2],
        "no_token_ids": [3, 4],
        "runtime": {
            "versions": {"torch": "test", "transformers": "test"},
            "gpu": {"available": True, "count": 1, "devices": ["A5000"]},
        },
        "counts": {
            "candidate_rows_written": row_count,
            "candidates_skipped_no_token": 0,
        },
        "hashes": {
            "raw_output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "measure_source_sha256": "b" * 64,
            "workspace_lens_source_sha256": "c" * 64,
        },
    }


def _write_inputs(
    tmp_path: Path,
    rows: list[dict] | None = None,
    *,
    write_metadata: bool = True,
) -> dict[str, Path]:
    paths = {}
    for condition in m0_gate.CONDITIONS:
        path = tmp_path / f"{condition}.json"
        condition_rows = rows or _green_rows()
        path.write_text(json.dumps(condition_rows), encoding="utf-8")
        if write_metadata:
            Path(f"{path}.metadata").write_text(
                json.dumps(_metadata_for(path, len(condition_rows))), encoding="utf-8"
            )
        paths[condition] = path
    return paths


def test_m0_gate_computes_metrics_and_green_decision(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    result = m0_gate.analyze_files(paths)

    assert result["decision"] == "GREEN"
    decoupled = result["conditions"]["decoupled"]
    assert decoupled["pooled_auc"]["V"] == pytest.approx(1 / 3)
    assert decoupled["pooled_auc"]["W_rr"] == pytest.approx(2 / 3)
    assert decoupled["within_episode_auc"] == {"V": 0.0, "W_rr": 1.0}
    assert decoupled["yes_rate"] == 0.5
    assert decoupled["counts"]["episodes"] == 3
    assert decoupled["counts"]["candidates"] == 6
    assert decoupled["counts"]["labels"] == {
        "distractor": 1,
        "filler": 2,
        "load_bearing": 3,
    }


def test_m0_gate_writes_both_reports_and_refuses_overwrite(tmp_path: Path) -> None:
    result = m0_gate.analyze_files(_write_inputs(tmp_path))
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    m0_gate.write_reports(result, out_json, out_md)

    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["decision"] == "GREEN"
    assert "**Decision: GREEN**" in out_md.read_text(encoding="utf-8")
    original_json = out_json.read_bytes()
    original_md = out_md.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        m0_gate.write_reports(result, out_json, out_md)
    assert out_json.read_bytes() == original_json
    assert out_md.read_bytes() == original_md


def test_m0_cli_checks_outputs_before_missing_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_json = tmp_path / "sealed.json"
    out_json.write_text("sealed\n", encoding="utf-8")

    argv = []
    for condition in m0_gate.CONDITIONS:
        argv.extend([f"--{condition}", str(tmp_path / f"missing-{condition}.json")])
    argv.extend(
        ["--out-json", str(out_json), "--out-md", str(tmp_path / "report.md")]
    )
    with pytest.raises(SystemExit) as exc_info:
        m0_gate.main(argv)

    assert exc_info.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert out_json.read_text(encoding="utf-8") == "sealed\n"


def test_m0_gate_verifies_measurement_sidecar_hash(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    raw_path = paths["explicit"]
    sidecar = Path(f"{raw_path}.metadata")
    sidecar.write_text(
        json.dumps({"hashes": {"raw_output_sha256": "0" * 64}}),
        encoding="utf-8",
    )

    with pytest.raises(m0_gate.M0GateError, match="raw output hash disagrees"):
        m0_gate.analyze_files(paths)

    actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    valid_metadata = _metadata_for(raw_path, len(_green_rows()))
    valid_metadata["hashes"]["raw_output_sha256"] = actual
    sidecar.write_text(json.dumps(valid_metadata), encoding="utf-8")
    result = m0_gate.analyze_files(paths)
    assert result["conditions"]["explicit"]["source"]["metadata"][
        "raw_output_hash_verified"
    ]


def test_m0_gate_investigates_out_of_tolerance_decoupled(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    rows = _green_rows()
    for row in rows:
        row["V"] = 1.0 if row["label"] == "load_bearing" else 0.0
    paths["decoupled"].write_text(json.dumps(rows), encoding="utf-8")
    Path(f"{paths['decoupled']}.metadata").write_text(
        json.dumps(_metadata_for(paths["decoupled"], len(rows))), encoding="utf-8"
    )

    result = m0_gate.analyze_files(paths)

    assert result["decision"] == "INVESTIGATE"
    assert result["gate"]["checks"] == {"V": False, "W_rr": True}


def test_m0_gate_allows_duplicate_concepts_with_candidate_indices(tmp_path: Path) -> None:
    rows = _green_rows()
    rows[1]["concept"] = rows[0]["concept"]
    paths = _write_inputs(tmp_path, rows)

    result = m0_gate.analyze_files(paths)

    assert result["decision"] == "GREEN"
    assert result["conditions"]["explicit"]["source"]["candidate_identity"] == (
        "episode+candidate_index"
    )


def test_m0_gate_requires_sidecars_by_default_but_has_legacy_escape_hatch(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, write_metadata=False)

    with pytest.raises(m0_gate.M0GateError, match="requires measurement metadata"):
        m0_gate.analyze_files(paths)

    result = m0_gate.analyze_files(paths, require_metadata=False)
    assert result["decision"] == "GREEN"
    assert result["provenance_audit"]["metadata_missing"] == list(
        m0_gate.CONDITIONS
    )
