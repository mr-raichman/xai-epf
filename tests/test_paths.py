"""El nom de la passada següent: res no peta si se n'equivoca, i una passada que es digués
com una de més vella no seria l'última que es continua."""
import datetime

from epf import paths


def test_the_next_run_counts_on_from_the_days_highest_and_smoke_apart(tmp_path, monkeypatch):
    # Un assaig no compta per a la mesura, ni un altre dia per a aquest, i una passada
    # esborrada a mà, la primera del dia, no fa tornar enrere el comptador.
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    for name in ("2026-10-14_03", "2026-10-15_02", "smoke_2026-10-15_07", "baseline"):
        (tmp_path / name).mkdir()
    day = datetime.date(2026, 10, 15)
    assert paths.next_run(False, day) == tmp_path / "2026-10-15_03"
    assert paths.next_run(True, day) == tmp_path / "smoke_2026-10-15_08"
    assert paths.next_run(False, datetime.date(2026, 10, 16)) == tmp_path / "2026-10-16_01"
    assert [path.name for path in paths.runs(False)] == ["2026-10-14_03", "2026-10-15_02"]
