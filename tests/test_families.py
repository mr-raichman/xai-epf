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
