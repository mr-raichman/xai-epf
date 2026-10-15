"""Quines configuracions ajusta la passada a cada llavor, i si el fitxer desat d'un model
torna el que explica.

Res d'això no peta quan falla: una cerca a mitges barrejaria llavors cercades i sense cercar
en una sola dispersió, la prova de datació amb totes les configuracions en lloc de la
millor no seria la prova que diu, i un fitxer que no sumés el que explica faria figures
d'una explicació que no és la del model.
"""
import numpy as np
import pandas as pd
import pytest

from epf import describe, paths, search
from epf.market import HOURS_PER_DAY
from epf.split import Split, split
from test_search import DELETED, PERIOD, _searcher, _table

# Una família que es cerca i se n'ajusten les tres millors configuracions de cada llavor.
FAKE = describe.Describer("daily", (), None, {}, {"value": 0}, 3)


def _searched(tmp_path, monkeypatch, seeds):
    """Cerques de quatre proves de la família `fake` a l'assaig, per a `seeds`, i les
    columnes sobre les quals es van fer."""
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    X, y = _table()
    where = search.folder("smoke", "fake")
    where.mkdir(parents=True)
    for seed in seeds:
        study = search.run(where / f"seed_{seed}.db", "fake",
                           _searcher({"value": (4, 1, 3, 2)}, []), X, y,
                           split(seed, PERIOD, DELETED), seed, 4)
        search.save(search.record(study, "fake", seed, "smoke", "commit"), "smoke")
    return X.columns


def test_with_no_seed_searched_each_seed_fits_the_placeholders_marked(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, [])
    found = describe.configurations("smoke", "fake", FAKE, columns, (0, 1), False)
    assert found == {seed: [{"rank": 0, "numbers": {"value": 0}, "unsearched": True}]
                     for seed in (0, 1)}


def test_with_every_seed_searched_each_fits_its_best_and_the_date_probe_the_first(
        tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0, 1))
    found = describe.configurations("smoke", "fake", FAKE, columns, (0, 1), False)
    for seed in (0, 1):
        assert [one["rank"] for one in found[seed]] == [0, 1, 2]
        assert [one["numbers"]["value"] for one in found[seed]] == [1, 2, 3]
        assert all(one["trials"] == 4 for one in found[seed])
    alone = describe.configurations("smoke", "fake", FAKE, columns, (0, 1), True)
    assert {seed: [one["numbers"]["value"] for one in ones]
            for seed, ones in alone.items()} == {0: [1], 1: [1]}


def test_a_search_of_some_seeds_only_is_refused(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0,))
    with pytest.raises(ValueError):
        describe.configurations("smoke", "fake", FAKE, columns, (0, 1), False)


def test_a_family_never_searched_fits_once_with_no_numbers(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0,))
    never = FAKE._replace(placeholder=None, kept=0)
    assert describe.configurations("smoke", "fake", never, columns, (0, 1), False) == {
        0: [{"rank": 0}], 1: [{"rank": 0}]}


# Tres dies de prova, dels quals s'expliquen els dos primers.
TEST_DAYS = pd.date_range("2019-01-01", periods=3, freq="D")
EXPLAINED = TEST_DAYS[:2]


def _saved(tmp_path, arrays, predicted, stored, explained=EXPLAINED):
    """Desa un model amb `arrays` i `predicted` com ho fa la passada, i en torna el camí."""
    none = TEST_DAYS[:0]
    path = tmp_path / "seed_0_0.npz"
    described = describe.Described(predicted, predicted, arrays, 0.0, describe.TOLERANCE,
                                   stored=stored)
    describe._save(path, Split(none, none, TEST_DAYS), explained, described, {}, 0)
    return path


def _tree(values):
    """La predicció de prova de l'arbre que fan els seus valors amb una base de 50, als
    dies explicats, i qualsevol als altres."""
    return np.concatenate([50.0 + values.sum(axis=1), np.zeros(HOURS_PER_DAY)])


def test_a_file_off_only_by_float32_rounding_takes_its_name(tmp_path):
    # Valors prou grans perquè desar-los a `float32` passi tot sol del marge de la
    # memòria: la fita de l'arrodoniment és el que fa que no es refusi cap model per com
    # es desa.
    values = np.random.default_rng(0).normal(0, 1e5, (2 * HOURS_PER_DAY, 200))
    rounded = values.astype(np.float32).astype(np.float64)
    assert np.abs(50.0 + rounded.sum(axis=1) - _tree(values)[:2 * HOURS_PER_DAY]).max() \
        > describe.TOLERANCE
    path = _saved(tmp_path, {"values": values, "base": 50.0}, _tree(values),
                  lambda found: describe._tree_sums(found, best=False))
    assert path.exists()


def test_a_file_that_does_not_add_up_is_refused_and_takes_no_name(tmp_path):
    # Les prediccions desades en un altre ordre, com les deixaria un fitxer mal escrit.
    values = np.random.default_rng(0).normal(0, 10, (2 * HOURS_PER_DAY, 20))
    with pytest.raises(ValueError):
        _saved(tmp_path, {"values": values, "base": 50.0}, _tree(values)[::-1],
               lambda found: describe._tree_sums(found, best=False))
    assert not (tmp_path / "seed_0_0.npz").exists()


def test_a_file_holding_a_nan_is_refused(tmp_path):
    # Una xarxa que divergeix dona NaN, i un NaN no és més gran que cap marge: la lectura
    # l'ha de refusar igualment.
    values = np.random.default_rng(0).normal(0, 10, (2 * HOURS_PER_DAY, 20))
    predicted = _tree(values)
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        _saved(tmp_path, {"values": values, "base": 50.0}, predicted,
               lambda found: describe._tree_sums(found, best=False))

def test_explained_days_other_than_the_first_test_days_are_refused(tmp_path):
    # La lectura pren les primeres files de prova pels dies explicats. Amb una altra tria
    # de dies compararia les sumes amb les prediccions d'uns altres dies, i aquí, on totes
    # valen igual, no se n'adonaria.
    values = np.zeros((2 * HOURS_PER_DAY, 3))
    with pytest.raises(ValueError):
        _saved(tmp_path, {"values": values, "base": 50.0},
               np.full(3 * HOURS_PER_DAY, 50.0),
               lambda found: describe._tree_sums(found, best=False), TEST_DAYS[1:])


def test_each_explained_familys_sums_read_its_own_arrays(tmp_path):
    rng = np.random.default_rng(0)
    # L'arbre a la seva millor configuració: cada fila d'interaccions suma el valor de la
    # seva columna, aquí posat a la diagonal.
    values = rng.normal(0, 10, (2 * HOURS_PER_DAY, 4))
    sample = np.array([1, 5, 30])
    interactions = np.zeros((len(sample), 4, 4))
    interactions[:, np.arange(4), np.arange(4)] = values[sample]
    assert _saved(tmp_path, {"values": values, "base": 50.0, "interaction_rows": sample,
                             "interactions": interactions},
                  _tree(values), lambda found: describe._tree_sums(found, best=True)).exists()
    # La xarxa distribucional: valors per fila, columna, hora i paràmetre, i els paràmetres
    # de tots els dies de prova.
    values = rng.normal(0, 10, (2, 5, HOURS_PER_DAY, 2))
    base = rng.normal(0, 10, (HOURS_PER_DAY, 2))
    parameters = np.concatenate([base + values.sum(axis=1), np.zeros((1, HOURS_PER_DAY, 2))])
    assert _saved(tmp_path, {"values": values, "base": base, "parameters": parameters},
                  np.zeros(3 * HOURS_PER_DAY), describe._network_sums).exists()
    # La xarxa elàstica, a l'espai transformat.
    values = rng.normal(0, 1, (2 * HOURS_PER_DAY, 6))
    base = rng.normal(0, 1, 2 * HOURS_PER_DAY)
    assert _saved(tmp_path, {"values": values, "base": base,
                             "transformed": base + values.sum(axis=1)},
                  np.zeros(3 * HOURS_PER_DAY), describe._net_sums).exists()
