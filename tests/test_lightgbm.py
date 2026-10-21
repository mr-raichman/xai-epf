import numpy as np
import pandas as pd
import pytest

from epf.contract.split import split
from epf.models.lightgbm import search, train
from epf.models.lightgbm.train import fit_rows, predict_rows
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La forma de la finestra on dos ajustos van donar arbres diferents, 1085 dies de 24
# hores i 98 columnes: una matriu petita no ho destapa mai.
ROWS, COLUMNS, TREES = 1085 * 24, 98, 200

# El punt amb què ho van fer: la taxa, inventada, i la resta, els valors per defecte de
# LightGBM.
POINT = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 1.0,
    "min_child_samples": 20,
}


def test_two_fits_on_the_same_rows_make_the_same_trees():
    """Sense `deterministic` i `force_row_wise`, fallava 8 vegades de 10 en repòs."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(ROWS, COLUMNS)), axis=0)
    y = X[:, :10].sum(axis=1) + rng.normal(scale=5, size=ROWS)
    first, second = (fit_rows(X, y, POINT, TREES, 0) for _ in range(2))
    assert np.array_equal(predict_rows(first, X[:500]), predict_rows(second, X[:500]))


def test_a_trial_and_the_run_hand_the_fit_the_same_seed(monkeypatch):
    """Amb una altra seed, la passada no ajustaria el que la cerca va puntuar."""

    class Stopped(Exception):
        """L'ajust fals s'atura aquí, un cop desada la seed."""

    seen = []

    def recorded(X, y, point, trees, seed, **keywords):
        seen.append(seed)
        raise Stopped

    monkeypatch.setattr(search, "fit_rows", recorded)
    monkeypatch.setattr(train, "fit_rows", recorded)
    days = pd.date_range(DEVELOPMENT[0], periods=14, freq="D")
    X = pd.DataFrame({"x": np.arange(len(days), dtype=float)}, index=days)
    y = pd.Series(np.zeros(len(days)), index=days)
    parts = split(3, DEVELOPMENT, BLACKOUT_WEEK)
    with pytest.raises(Stopped):
        search.trial("lightgbm", POINT, (X, y), (X, y), 3)
    with pytest.raises(Stopped):
        train.fitted(X, y, np.array(["x"]), parts, parts.test, 3, POINT, False)
    assert seen == [3, 3]
