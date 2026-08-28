from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import measure


@pytest.mark.parametrize("existing_target", ["raw", "metadata"])
def test_main_refuses_any_existing_output_before_loading_model(
    existing_target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "measurement.json"
    metadata = Path(f"{output}.metadata")
    target = output if existing_target == "raw" else metadata
    target.write_text("sealed\n", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        pytest.fail("WorkspaceLens was loaded before output protection")

    monkeypatch.setattr(measure, "WorkspaceLens", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure.py",
            "--model",
            "unused-model",
            "--battery",
            str(tmp_path / "missing-battery.json"),
            "--out",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        measure.main()

    assert exc_info.value.code == 2
    assert "refusing to overwrite existing output" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "sealed\n"
    other = metadata if existing_target == "raw" else output
    assert not other.exists()
