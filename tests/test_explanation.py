import types

import numpy as np
import pandas as pd
import pytest

from epf import Refusal
from epf.contract.explanation import TOLERANCE
from epf.contract.record import Fitted, priced, save_model
from epf.contract.split import Split
from epf.models.dnn import attribution as network
from epf.models.elastic_net import attribution as net
from epf.models.lightgbm import attribution as tree
from epf.parameters import HOURS_PER_DAY


# Tres dies de prova, dels quals s'expliquen els dos primers.
TEST_DAYS = pd.date_range("2019-01-01", periods=3, freq="D")
EXPLAINED = TEST_DAYS[:2]


def _saved(tmp_path, arrays, predicted, stored, explained=EXPLAINED, gap=0.0):
    """Desa un model ajustat amb `arrays` i `predicted`, i en torna el camí.

    `gap` és el residu de l'explicació que l'ajust diu.
    """
    none = TEST_DAYS[:0]
    path = tmp_path / "seed_0_0.npz"
    # Un preu real que varia, perquè l'R² en surti.
    actual = np.arange(len(predicted), dtype=float)
    fitted = Fitted(predicted, actual, arrays, gap, TOLERANCE, stored=stored)
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


def test_a_fit_whose_explanation_misses_its_margin_is_refused_before_any_file(
    tmp_path,
):
    """Com l'ajust d'una xarxa que se n'allunyava 0.0011, amb marge 0.001."""
    values = np.zeros((2 * HOURS_PER_DAY, 3))
    with pytest.raises(
        Refusal, match="se n'allunya 0.0011, més que el marge de 0.001$"
    ):
        _saved(
            tmp_path,
            {"values": values, "base": 50.0},
            _tree(values),
            lambda found: tree.sums(found, with_interactions=False),
            gap=0.0011,
        )
    assert not list(tmp_path.iterdir())


def test_a_file_that_does_not_add_up_is_refused_and_takes_no_name(tmp_path):
    """Un fitxer que no sumés el que explica faria figures d'una altra explicació."""
    # Les prediccions desades en un altre ordre, com les deixaria un fitxer mal escrit.
    values = np.random.default_rng(0).normal(0, 10, (2 * HOURS_PER_DAY, 20))
    with pytest.raises(Refusal):
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
    with pytest.raises(Refusal):
        _saved(
            tmp_path,
            {"values": values, "base": 50.0},
            predicted,
            lambda found: tree.sums(found, with_interactions=False),
        )


def test_explained_days_other_than_the_first_test_days_fail(tmp_path):
    # Amb uns altres dies explicats, la lectura compararia les sumes amb les prediccions
    # d'uns altres dies, i aquí, on totes valen igual, no se n'adonaria.
    values = np.zeros((2 * HOURS_PER_DAY, 3))
    with pytest.raises(ValueError) as raised:
        _saved(
            tmp_path,
            {"values": values, "base": 50.0},
            np.full(3 * HOURS_PER_DAY, 50.0),
            lambda found: tree.sums(found, with_interactions=False),
            TEST_DAYS[1:],
        )
    assert not isinstance(raised.value, Refusal)


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
    # La xarxa neuronal: contribucions per dia, columna i hora, i la predicció de tots
    # els dies de prova.
    values = rng.normal(0, 10, (2, 5, HOURS_PER_DAY))
    base = rng.normal(0, 10, HOURS_PER_DAY)
    predicted = np.concatenate(
        [(base + values.sum(axis=1)).ravel(), np.zeros(HOURS_PER_DAY)]
    )
    assert _saved(
        tmp_path, {"values": values, "base": base}, predicted, network.sums
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


def test_a_file_with_a_reference_adds_up_once_the_reference_is_taken_away(tmp_path):
    """El fitxer desa la predicció en preu, i explica la distància a la referència."""
    values = np.random.default_rng(0).normal(0, 10, (2 * HOURS_PER_DAY, 20))
    built = types.SimpleNamespace(
        reference=pd.Series([60.0, 70.0, 80.0], index=TEST_DAYS),
        actual=pd.DataFrame(
            np.arange(3 * HOURS_PER_DAY, dtype=float).reshape(3, -1), index=TEST_DAYS
        ),
    )
    none = TEST_DAYS[:0]
    fitted = Fitted(
        _tree(values),
        np.zeros(3 * HOURS_PER_DAY),
        {"values": values, "base": 50.0},
        0.0,
        TOLERANCE,
        stored=lambda found: tree.sums(found, with_interactions=False),
    )
    with_reference = priced(fitted, built, TEST_DAYS)
    path = tmp_path / "seed_0_0.npz"
    save_model(path, Split(none, none, TEST_DAYS), EXPLAINED, with_reference, 0)
    assert path.exists()
    # La mateixa predicció en preu, sense la referència desada, no sumaria.
    lost = with_reference._replace(arrays={"values": values, "base": 50.0})
    with pytest.raises(Refusal, match="desada de la configuració 0 no torna"):
        save_model(
            tmp_path / "seed_0_1.npz", Split(none, none, TEST_DAYS), EXPLAINED, lost, 0
        )
