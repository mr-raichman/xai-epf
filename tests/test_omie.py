"""Reducció de períodes a hores, i una baixada interrompuda."""
import pytest
import requests

from epf.data import omie
from epf.data.omie import to_hourly


def test_an_hourly_day_passes_through():
    day = list(range(24))
    assert to_hourly(day) == day


def test_the_two_daylight_saving_lengths_pass_through_unaveraged():
    assert len(to_hourly(list(range(23)))) == 23
    assert len(to_hourly(list(range(25)))) == 25


def test_ninety_six_periods_become_twenty_four_means():
    quarters = [float(i) for i in range(96)]
    hours = to_hourly(quarters)
    assert len(hours) == 24
    assert hours[0] == (0 + 1 + 2 + 3) / 4
    assert hours[23] == (92 + 93 + 94 + 95) / 4


def test_the_quarter_hourly_daylight_saving_days_keep_their_length():
    assert len(to_hourly([0.0] * 92)) == 23
    assert len(to_hourly([0.0] * 100)) == 25


def test_a_quarter_hourly_day_short_of_a_period_is_refused():
    # Agrupats de quatre en quatre, 95 períodes farien 23 hores i el residu es perdria
    # sense dir-ho: el dia d'un fitxer tallat passaria per un dia de primavera.
    with pytest.raises(ValueError, match="95"):
        to_hourly([0.0] * 95)


def test_an_interrupted_download_leaves_no_file_under_its_real_name(tmp_path, monkeypatch):
    # Un fitxer amb el seu nom no es torna a baixar mai, de manera que un de tallat
    # quedaria tallat per sempre.
    name = "marginalpdbc_20251001.1"
    response = requests.Response()
    response.status_code = 200
    response._content = b"MARGINALPDBC;\n" + b"2025;10;01;1;80.0;80.0;\n" * 96
    monkeypatch.setattr(omie.requests, "get", lambda *args, **kwargs: response)

    def interrupted(path, content):
        with path.open("wb") as handle:
            handle.write(content[:len(content) // 2])
        raise OSError("interromput")

    monkeypatch.setattr(omie.pathlib.Path, "write_bytes", interrupted)
    with pytest.raises(OSError):
        omie._fetch(name, tmp_path)
    assert not (tmp_path / name).exists()
