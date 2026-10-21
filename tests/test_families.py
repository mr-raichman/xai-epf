import importlib

from epf import paths
from epf.contract import record
from epf.run import families


def test_every_module_the_table_names_is_imported():
    """Un mòdul mogut deixa el seu nom vell a la taula, i cap linter no el llegeix."""
    for found in families.MODULES.values():
        for kind in ("train", "search", "figures"):
            if found[kind] is not None:
                importlib.import_module(found[kind])


def test_a_family_planned_for_some_seeds_alone_is_not_complete(tmp_path):
    """Planificada seed a seed, una família amb el pla d'una sola es donava per feta."""
    path = tmp_path / paths.SEED_RECORD
    for planned in ({"0": [{"rank": 0}]}, {"0": [{"rank": 0}], "1": [{"rank": 0}]}):
        record.write_record(
            path, {"settings": {"SEEDS": [0, 1]}, "configurations": planned}
        )
        for seed in planned:
            (tmp_path / paths.SEED.format(seed=seed, rank=0)).touch()
        assert families.complete(tmp_path) == (len(planned) == 2)


def test_a_record_without_settings_is_not_complete(tmp_path):
    """Un registre d'abans de `settings` aturava `epf run` amb un `KeyError`."""
    path = tmp_path / paths.SEED_RECORD
    record.write_record(path, {"seeds": [0], "configurations": {"0": [{"rank": 0}]}})
    (tmp_path / paths.SEED.format(seed=0, rank=0)).touch()
    assert not families.complete(tmp_path)
