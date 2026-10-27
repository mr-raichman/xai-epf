import pytest

from epf import Refusal, information
from epf.information import PRICE, SERIES, WITH_PRICE, InformationSet, Series


def _other(monkeypatch, series, reference=False):
    """Posa a `information` un information set més, `OTHER`, de `series`."""
    monkeypatch.setattr(
        information, "OTHER", InformationSet(series, reference), raising=False
    )


@pytest.mark.parametrize(
    "series, why",
    [
        (
            {"nuclear_sched": [1], "nuclear_again": [1]},
            "^l'indicador 4 és a dues sèries de OTHER, nuclear_sched i nuclear_again$",
        ),
        (
            {"oil_sched": [1]},
            "^OTHER llegeix la sèrie oil_sched, que SERIES no té$",
        ),
        (
            {PRICE: [1], "price_lag1": [0]},
            "^la columna price_lag1 es fa dues vegades a OTHER$",
        ),
        (
            {PRICE: [0, 1]},
            "^el preu entra a OTHER al retard 0, el dia que es prediu$",
        ),
        (
            {"demand_fcst": [-1]},
            "^la sèrie demand_fcst de OTHER té un retard negatiu, -1$",
        ),
    ],
    ids=[
        "indicador dues vegades",
        "sèrie que no hi és",
        "columna dues vegades",
        "preu al retard 0",
        "retard negatiu",
    ],
)
def test_a_set_that_cannot_be_used_is_refused_whichever_set_is_asked_for(
    monkeypatch, series, why
):
    # Les sèries d'un sol cas, a la llista de totes.
    added = {
        "nuclear_again": SERIES["nuclear_sched"],
        "price_lag1": Series([99], "forecast", ""),
    }
    monkeypatch.setattr(
        information,
        "SERIES",
        SERIES | {name: row for name, row in added.items() if name in series},
    )
    _other(monkeypatch, series)
    with pytest.raises(Refusal, match=why):
        information.resolved("WITH_PRICE")


def test_a_series_of_an_unknown_kind_is_refused(monkeypatch):
    monkeypatch.setattr(
        information,
        "SERIES",
        SERIES | {"demand_fcst": SERIES["demand_fcst"]._replace(kind="forcast")},
    )
    with pytest.raises(
        Refusal,
        match='^la sèrie demand_fcst té un tipus que no existeix, "forcast"; els '
        "tipus: price, forecast, schedule$",
    ):
        information.every()


def test_a_reference_weighed_by_a_series_the_list_lacks_is_refused(monkeypatch):
    monkeypatch.setattr(
        information, "TARGET", information.TARGET._replace(weights="load_fcst")
    )
    with pytest.raises(Refusal, match="^la referència es pesa amb load_fcst, que"):
        information.every()


def test_a_name_of_no_set_is_refused_with_the_sets():
    with pytest.raises(
        Refusal,
        match='^"WITHOUT" no és cap information set; els information sets: '
        "WITH_PRICE, WITHOUT_PRICE$",
    ):
        information.resolved("WITHOUT")


def test_the_gap_is_the_price_s_longest_lag_in_its_set_the_reference_s_among_them(
    monkeypatch,
):
    """Sense el preu, la referència el llegeix el dia abans: el gap no pot ser zero."""
    assert information.resolved("WITH_PRICE").gap == 1
    assert information.resolved("WITHOUT_PRICE").gap == 1
    _other(
        monkeypatch,
        {name: lags for name, lags in WITH_PRICE.series.items() if name != PRICE},
    )
    assert information.resolved("OTHER").gap == 0
