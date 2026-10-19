from epf.run import families, figures, run, search


def test_every_family_a_run_fits_searches_or_draws_is_a_models():
    """Una família que cap model no té no es podria nomenar.

    Una passada, a més, es donaria per acabada sense ella.
    """
    assert set(run.TRAINERS) == families.EVERY
    assert set(figures.FIGURES) == families.EVERY
    assert set(search.SEARCHERS) <= families.EVERY
