from epf import paths
from epf.contract import record
from epf.run import families, figures, run, search


def test_every_family_a_run_fits_searches_or_draws_is_a_models_in_its_order():
    """Una família que cap model no té no es podria nomenar.

    Una passada, a més, es donaria per acabada sense ella, i en un altre ordre les
    famílies es farien i es mostrarien en un altre.
    """
    assert list(run.TRAINERS) == list(families.EVERY)
    assert list(figures.FIGURES) == list(families.EVERY)
    assert list(search.SEARCHERS) == [
        family for family in families.EVERY if family in search.SEARCHERS
    ]


def test_a_family_planned_for_some_seeds_alone_is_not_complete(tmp_path):
    """Planificada seed a seed, una família amb el pla d'una sola es donava per feta."""
    path = tmp_path / paths.SEED_RECORD
    for planned in ({"0": [{"rank": 0}]}, {"0": [{"rank": 0}], "1": [{"rank": 0}]}):
        record.write_record(path, {"seeds": [0, 1], "configurations": planned})
        for seed in planned:
            (tmp_path / paths.SEED.format(seed=seed, rank=0)).touch()
        assert families.complete(tmp_path) == (len(planned) == 2)
