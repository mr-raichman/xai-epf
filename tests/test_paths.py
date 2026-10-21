import datetime
import json

import pytest

from epf import paths


def test_the_next_run_counts_on_from_the_days_highest_and_smoke_apart(
    tmp_path, monkeypatch
):
    """Un nom equivocat no fallaria: faria continuar una altra passada."""
    # Un assaig no compta per a la mesura, ni un altre dia per a aquest, i una passada
    # esborrada a mà, la primera del dia, no fa tornar enrere el comptador.
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    smoke = paths.SMOKE_PREFIX
    for name in ("2026-10-14_03", "2026-10-15_02", f"{smoke}2026-10-15_07", "baseline"):
        (tmp_path / name).mkdir()
    day = datetime.date(2026, 10, 15)
    assert paths.next_run(False, day) == tmp_path / "2026-10-15_03"
    assert paths.next_run(True, day) == tmp_path / f"{smoke}2026-10-15_08"
    assert (
        paths.next_run(False, datetime.date(2026, 10, 16)) == tmp_path / "2026-10-16_01"
    )
    assert [path.name for path in paths.runs(False)] == [
        "2026-10-14_03",
        "2026-10-15_02",
    ]


def test_a_value_json_cannot_write_is_refused_and_not_silently_dropped():
    with pytest.raises(TypeError):
        json.dumps(paths.plain({"bad": object()}))
