"""Quines configuracions ajusta la passada a cada llavor.

Res d'això no peta quan falla: una cerca a mitges barrejaria llavors cercades i sense cercar
en una sola dispersió, i la prova de datació amb totes les configuracions en lloc de la
millor no seria la prova que diu.
"""
import pytest

from epf import describe, paths, search
from epf.split import split
from tests.test_search import DELETED, PERIOD, _searcher, _table

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
