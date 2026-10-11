"""Quins pronòstics entren a la combinació, i amb quin pes.

Qui és membre ho diu el `run.json` de cada família. Abans sortia del `.npz` i la
combinació d'una execució acabada de fer comptava LEAR un sol cop en lloc d'un per finestra,
perquè l'atribució ja no s'escriu al camí endavant.
"""
import datetime

import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.results import run_files
from epf.results.combination import combined, members


def frame_of(days, value, start=datetime.date(2024, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    return pd.DataFrame(float(value), index=index, columns=HOUR_COLUMNS)


def written(folder, values, headline, days, headline_value=None):
    """Una família al disc: un CSV per membre, el de capçalera, i el seu registre."""
    folder.mkdir(parents=True)
    frames = {name: frame_of(days, value) for name, value in values.items()}
    if headline not in frames:
        frames[headline] = frame_of(days, headline_value)
    run_files.save_forecasts(folder, frames)
    run_files.save_record(folder, {"members": sorted(values), "headlines": [headline]})
    return frames[headline]


def one_model(folder, name, value, days=3):
    """Una família d'un sol model: el seu CSV és el membre i la capçalera."""
    return written(folder, {name: value}, headline=name, days=days)


def ensemble(folder, name, values, days=3):
    """Un conjunt com LEAR: un CSV per finestra i un altre amb la seva mitjana."""
    windows = {f"{name}_{number}": value for number, value in enumerate(values)}
    return written(folder, windows, headline=name, days=days,
                   headline_value=sum(values) / len(values))


def test_an_ensemble_counts_each_window_and_a_single_model_counts_once(tmp_path):
    # Si un conjunt comptés un sol cop, la combinació li donaria el pes d'una
    # família i no el dels models que ajusta, que és el que compta l'article.
    one_model(tmp_path / "tree", "tree", 10.0)
    ensemble(tmp_path / "lear", "lear", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(members(tmp_path / "tree")) == 1
    assert len(members(tmp_path / "lear")) == 5


def test_the_mean_of_an_ensemble_is_not_a_member_of_itself(tmp_path):
    # La capçalera d'un conjunt és la mitjana de les seves finestres i no cap model
    # ajustat: comptar-la hi donaria el doble de pes i la combinació seria una
    # mitjana ponderada que ningú no ha decidit.
    ensemble(tmp_path / "lear", "lear", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert "lear" not in run_files.load_record(tmp_path / "lear")["members"]
    assert len(members(tmp_path / "lear")) == 5


def test_a_single_model_family_contributes_its_one_csv(tmp_path):
    folder = tmp_path / "dnn"
    one_model(folder, "dnn", 7.0)
    found = members(folder)
    assert list(found) == ["dnn/dnn"]
    assert found["dnn/dnn"].to_numpy().min() == 7.0


def test_a_record_that_does_not_say_who_is_a_member_combines_nothing(tmp_path):
    # Amb una barreja de registres la combinació sortiria d'un subconjunt i no ho
    # diria enlloc, de manera que val més que no n'hi hagi cap.
    folder = tmp_path / "old"
    folder.mkdir(parents=True)
    run_files.save_forecasts(folder, {"old": frame_of(3, 5.0)})
    run_files.save_record(folder, {"family": "old"})
    assert members(folder) is None
    one_model(tmp_path / "tree", "tree", 10.0)
    frame, found = combined([folder, tmp_path / "tree"])
    assert frame is None and found == {}


def test_the_combination_is_the_simple_average_of_every_member(tmp_path):
    ensemble(tmp_path / "lear", "lear", [1.0, 2.0, 3.0, 4.0, 5.0])
    one_model(tmp_path / "tree", "tree", 10.0)
    frame, found = combined([tmp_path / "lear", tmp_path / "tree"])
    assert len(found) == 6
    # (1 + 2 + 3 + 4 + 5 + 10) / 6, i no (3 + 10) / 2, que és el que donaria comptar
    # famílies en lloc de membres.
    assert frame.to_numpy().min() == 25.0 / 6


def test_the_combination_is_never_a_member_of_itself(tmp_path):
    # El seu directori porta registre, de manera que `family_folders` el dóna com un
    # més i tot el que llegeix una execució el llegeix. Si `combined` no el saltés, la
    # combinació es mitjanaria amb els seus propis membres i el pes de cada un ja no
    # seria el que diu el registre.
    ensemble(tmp_path / "lear", "lear", [1.0, 3.0])
    one_model(tmp_path / "tree", "tree", 10.0)
    folders = [tmp_path / "lear", tmp_path / "tree"]
    frame, found = combined(folders)
    one_model(tmp_path / run_files.COMBINATION, run_files.COMBINATION,
              frame.to_numpy()[0, 0])
    again, found_again = combined([*folders, tmp_path / run_files.COMBINATION])
    assert list(found_again) == list(found)
    assert (again == frame).to_numpy().all()


def test_one_member_is_nothing_to_combine(tmp_path):
    one_model(tmp_path / "tree", "tree", 10.0)
    frame, found = combined([tmp_path / "tree"])
    assert frame is None and len(found) == 1


def test_only_the_days_every_member_has(tmp_path):
    ensemble(tmp_path / "lear", "lear", [1.0, 3.0], days=5)
    one_model(tmp_path / "tree", "tree", 10.0, days=3)
    frame, _ = combined([tmp_path / "lear", tmp_path / "tree"])
    assert len(frame) == 3
