"""El conjunt del preu: què el separa de la resta de sèries d'ESIOS, i com s'uneix."""
import pandas as pd

from epf.data import esios, price


def test_the_price_indicator_is_outside_the_generic_series_table():
    # 600 porta diverses geografies que es trien en lloc de sumar-se, de manera que
    # `esios.load` no el pot llegir: si algun dia entrés a `SERIES`, el llegiria
    # sumant Espanya amb Portugal i amb França i no ho diria.
    assert price.ESIOS_PRICE not in esios.SERIES
    assert price.SPAIN in price.PRICE_COUNTRIES


def test_an_unknown_country_is_refused_rather_than_read_as_empty():
    try:
        price.from_esios("Spain")
    except KeyError as error:
        assert "Spain" in str(error)
    else:
        raise AssertionError("no ha refusat un país que no hi és")


def test_the_two_sources_meet_at_the_seam_without_overlapping():
    # El tall és el primer dia d'OMIE: ESIOS fins al dia abans, OMIE des d'aquell.
    seam = pd.Timestamp(price.OMIE_FROM)
    assert price.ESIOS_FIRST_DAY < price.OMIE_FROM
    assert seam.date() == price.OMIE_FROM
