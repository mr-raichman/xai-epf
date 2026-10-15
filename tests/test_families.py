"""Les famílies: totes les que una passada ajusta, cerca o dibuixa són d'un model.

Res no peta si no ho són: una família que una passada ajusta i cap model no té no es podria
nomenar, i una passada es donaria per acabada sense ella.
"""
from epf.run import families, figures, run, search


def test_every_family_a_run_fits_searches_or_draws_is_a_models():
    assert set(run.TRAINERS) == families.EVERY
    assert set(figures.FIGURES) == families.EVERY
    assert set(search.SEARCHERS) <= families.EVERY
