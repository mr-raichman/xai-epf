import datetime
import json

import pytest

from epf import paths


def test_a_new_run_is_named_by_its_set_and_day_and_listed_with_its_kind_and_set_alone(
    tmp_path, monkeypatch
):
    """Un nom equivocat no fallaria: faria continuar una altra passada."""
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(paths, "SMOKE_OUTPUT", tmp_path / "assaig_output")
    minute = datetime.datetime(2026, 10, 22, 14, 30, 59)
    for smoke in (False, True):
        for set_name in ("WITH_PRICE", "WITHOUT_PRICE"):
            paths.next_run(smoke, set_name, minute).mkdir(parents=True)
    # Una carpeta d'un altre nom no és cap passada.
    (paths.runs(False, "WITH_PRICE") / "baseline").mkdir()
    assert [path.name for path in paths.named_runs(False, "WITH_PRICE")] == [
        "with_price_22-10"
    ]
    assert [path.name for path in paths.named_runs(True, "WITHOUT_PRICE")] == [
        f"{paths.SMOKE_PREFIX}without_price_22-10"
    ]
    assert paths.runs(True, "WITHOUT_PRICE") == (
        tmp_path / "assaig_output" / "runs" / "without_price"
    )


def test_a_value_json_cannot_write_fails_and_is_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(paths.plain({"bad": object()}))
