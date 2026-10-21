import numpy as np
import pandas as pd
import pytest

from epf.contract.explanation import TOLERANCE
from epf.contract.record import Fitted, save_model
from epf.contract.split import Split
from epf.models.elastic_net import attribution as net
from epf.models.lightgbm import attribution as tree
from epf.models.nn.ddnn import attribution as network
from epf.parameters import HOURS_PER_DAY


# Tres dies de prova, dels quals s'expliquen els dos primers.
TEST_DAYS = pd.date_range("2019-01-01", periods=3, freq="D")
EXPLAINED = TEST_DAYS[:2]


def _saved(tmp_path, arrays, predicted, stored, explained=EXPLAINED):
    """Desa un model ajustat amb `arrays` i `predicted`, i en torna el camí."""
    none = TEST_DAYS[:0]
    path = tmp_path / "seed_0_0.npz"
    # Un preu real que varia, perquè l'R² en surti.
    actual = np.arange(len(predicted), dtype=float)
    fitted = Fitted(predicted, actual, arrays, 0.0, TOLERANCE, stored=stored)
    save_model(path, Split(none, none, TEST_DAYS), explained, fitted, 0)
    return path


def _tree(values):
    """Torna la predicció de prova que fan `values` amb una base de 50.

    Als dies que no s'expliquen val zero.
    """
    return np.concatenate([50.0 + values.sum(axis=1), np.zeros(HOURS_PER_DAY)])


def test_a_file_off_only_by_float32_rounding_takes_its_name(tmp_path):
    # Valors prou grans perquè desar-los a `float32` passi tot sol del marge.
    values = np.random.default_rng(0).normal(0, 1e5, (2 * HOURS_PER_DAY, 200))
    rounded = values.astype(np.float32).astype(np.float64)
    assert (
        np.abs(50.0 + rounded.sum(axis=1) - _tree(values)[: 2 * HOURS_PER_DAY]).max()
        > TOLERANCE
    )
    path = _saved(
        tmp_path,
        {"values": values, "base": 50.0},
        _tree(values),
        lambda found: tree.sums(found, with_interactions=False),
    )
    assert path.exists()


def test_a_file_that_does_not_add_up_is_refused_and_takes_no_name(tmp_path):
    """Un fitxer que no sumés el que explica faria figures d'una altra explicació."""
    # Les prediccions desades en un altre ordre, com les deixaria un fitxer mal escrit.
    values = np.random.default_rng(0).normal(0, 10, (2 * HOURS_PER_DAY, 20))
    with pytest.raises(ValueError):
        _saved(
            tmp_path,
            {"values": values, "base": 50.0},
            _tree(values)[::-1],
            lambda found: tree.sums(found, with_interactions=False),
        )
    assert not (tmp_path / "seed_0_0.npz").exists()


def test_a_file_holding_a_nan_is_refused(tmp_path):
    # Una xarxa que divergeix dona NaN.
    values = np.random.default_rng(0).normal(0, 10, (2 * HOURS_PER_DAY, 20))
    predicted = _tree(values)
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        _saved(
            tmp_path,
            {"values": values, "base": 50.0},
            predicted,
            lambda found: tree.sums(found, with_interactions=False),
        )


def test_explained_days_other_than_the_first_test_days_are_refused(tmp_path):
    # Amb uns altres dies explicats, la lectura compararia les sumes amb les prediccions
    # d'uns altres dies, i aquí, on totes valen igual, no se n'adonaria.
    values = np.zeros((2 * HOURS_PER_DAY, 3))
    with pytest.raises(ValueError):
        _saved(
            tmp_path,
            {"values": values, "base": 50.0},
            np.full(3 * HOURS_PER_DAY, 50.0),
            lambda found: tree.sums(found, with_interactions=False),
            TEST_DAYS[1:],
        )


def test_each_explained_familys_sums_read_its_own_arrays(tmp_path):
    rng = np.random.default_rng(0)
    # L'arbre a la seva millor configuració: cada fila d'interaccions suma la
    # contribució de la seva columna, aquí posada a la diagonal.
    values = rng.normal(0, 10, (2 * HOURS_PER_DAY, 4))
    sample = np.array([1, 5, 30])
    interactions = np.zeros((len(sample), 4, 4))
    interactions[:, np.arange(4), np.arange(4)] = values[sample]
    assert _saved(
        tmp_path,
        {
            "values": values,
            "base": 50.0,
            "interaction_rows": sample,
            "interactions": interactions,
        },
        _tree(values),
        lambda found: tree.sums(found, with_interactions=True),
    ).exists()
    # La xarxa distribucional: contribucions per fila, columna, hora i paràmetre, i els
    # paràmetres de tots els dies de prova.
    values = rng.normal(0, 10, (2, 5, HOURS_PER_DAY, 2))
    base = rng.normal(0, 10, (HOURS_PER_DAY, 2))
    parameters = np.concatenate(
        [base + values.sum(axis=1), np.zeros((1, HOURS_PER_DAY, 2))]
    )
    assert _saved(
        tmp_path,
        {"values": values, "base": base, "parameters": parameters},
        np.zeros(3 * HOURS_PER_DAY),
        network.sums,
    ).exists()
    # L'elastic net, a l'espai transformat.
    values = rng.normal(0, 1, (2 * HOURS_PER_DAY, 6))
    base = rng.normal(0, 1, 2 * HOURS_PER_DAY)
    assert _saved(
        tmp_path,
        {"values": values, "base": base, "transformed": base + values.sum(axis=1)},
        np.zeros(3 * HOURS_PER_DAY),
        net.sums,
    ).exists()
