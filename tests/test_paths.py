import datetime
import json

import pytest

from epf import paths


def test_a_new_run_is_named_by_its_minute_and_listed_with_its_kind_alone(
    tmp_path, monkeypatch
):
    """Un nom equivocat no fallaria: faria continuar una altra passada."""
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(paths, "SMOKE_OUTPUT", tmp_path / "assaig_output")
    minute = datetime.datetime(2026, 10, 22, 14, 30, 59)
    for smoke in (False, True):
        paths.next_run(smoke, minute).mkdir(parents=True)
    # Una carpeta d'un altre nom no és cap passada.
    (paths.runs(False) / "baseline").mkdir()
    assert [path.name for path in paths.named_runs(False)] == ["2026-10-22_14-30"]
    assert [path.name for path in paths.named_runs(True)] == [
        f"{paths.SMOKE_PREFIX}2026-10-22_14-30"
    ]
    assert paths.runs(True) == tmp_path / "assaig_output" / "runs"


def test_a_value_json_cannot_write_fails_and_is_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(paths.plain({"bad": object()}))
