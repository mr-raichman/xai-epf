"""Quins pronòstics entren a la combinació, i amb quin pes."""
import datetime

import numpy as np
import pandas as pd

from epf.results import run as run_files
from epf.calendar import HOUR_COLUMNS
from epf.runner import combined, members
from epf.models.attribution import ONE, PRICE_UNITS, Attribution, save


def frame_of(days, value, start=datetime.date(2024, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    return pd.DataFrame(float(value), index=index, columns=HOUR_COLUMNS)


def one_model(folder, name, value, days=3):
    """Una família d'un sol model: el CSV i una atribució sense membres propis."""
    folder.mkdir(parents=True)
    frame = frame_of(days, value)
    run_files.save_forecasts(folder, {name: frame})
    attributions = [Attribution(ONE, ["a"], np.zeros((1, 1, 24)), np.zeros((1, 24)),
                                None, PRICE_UNITS) for _ in range(days)]
    save(folder, name, frame.index, attributions)
    return frame


def ensemble(folder, name, values, days=3):
    """Un conjunt com LEAR: el CSV porta la mitjana i el `.npz` els membres."""
    folder.mkdir(parents=True)
    frame = frame_of(days, sum(values) / len(values))
    run_files.save_forecasts(folder, {name: frame})
    attributions = [
        Attribution(tuple(str(number) for number in range(len(values))), ["a"],
                    np.zeros((len(values), 1, 24)), np.zeros((len(values), 24)),
                    np.full((len(values), 24), 0.0) + np.array(values)[:, None],
                    "espai transformat")
        for _ in range(days)]
    save(folder, name, frame.index, attributions)
    return frame


def test_an_ensemble_counts_each_window_and_a_single_model_counts_once(tmp_path):
    # Si un conjunt comptés un sol cop, la combinació li donaria el pes d'una
    # família i no el dels models que ajusta, que és el que compta l'article.
    one_model(tmp_path / "tree", "tree", 10.0)
    ensemble(tmp_path / "lear", "lear", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(members(tmp_path / "tree")) == 1
    assert len(members(tmp_path / "lear")) == 5


def test_a_family_with_no_attribution_contributes_its_csv(tmp_path):
    folder = tmp_path / "mlp"
    folder.mkdir(parents=True)
    run_files.save_forecasts(folder, {"mlp": frame_of(3, 7.0)})
    found = members(folder)
    assert len(found) == 1
    assert found[0].to_numpy().min() == 7.0


def test_the_combination_is_the_simple_average_of_every_member(tmp_path):
    ensemble(tmp_path / "lear", "lear", [1.0, 2.0, 3.0, 4.0, 5.0])
    one_model(tmp_path / "tree", "tree", 10.0)
    frame, found = combined([tmp_path / "lear", tmp_path / "tree"])
    assert len(found) == 6
    # (1 + 2 + 3 + 4 + 5 + 10) / 6, and not (3 + 10) / 2, which is what counting
    # families rather than members would give.
    assert frame.to_numpy().min() == 25.0 / 6


def test_one_member_is_nothing_to_combine(tmp_path):
    one_model(tmp_path / "tree", "tree", 10.0)
    frame, found = combined([tmp_path / "tree"])
    assert frame is None and len(found) == 1


def test_only_the_days_every_member_has(tmp_path):
    ensemble(tmp_path / "lear", "lear", [1.0, 3.0], days=5)
    one_model(tmp_path / "tree", "tree", 10.0, days=3)
    frame, _ = combined([tmp_path / "lear", tmp_path / "tree"])
    assert len(frame) == 3
