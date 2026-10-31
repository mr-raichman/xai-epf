import numpy as np
import pytest

from epf.contract.measures import (
    contributions,
    importances,
    pairs,
    part_ii_main,
    summarised,
)
from epf.parameters import HOUR_COLUMNS


def test_a_group_sums_its_columns_before_the_absolute_value():
    # Dues columnes del mateix grup que s'anul·len: per columna pesen, com a grup no.
    values = np.array([[1.0, -1.0, 2.0], [1.0, -1.0, -2.0]])
    by_column, by_group = importances(values, np.array(["a", "a", "b"]))
    assert np.allclose(by_column, [0.25, 0.25, 0.5])
    assert by_group == {"a": 0.0, "b": 1.0}


def test_the_networks_forecast_is_paired_with_its_own_hour():
    # La contribució de la columna de l'hora k a la sortida de l'hora h val 100 k + h:
    # la parella a l'hora només ha de prendre les de k igual a h. Una altra columna al
    # davant, perquè la posició d'una columna no coincideixi amb la seva hora.
    columns = ["other"] + [f"demand_fcst_{hour}" for hour in HOUR_COLUMNS]
    days = 2
    values = np.zeros((days, len(columns), 24))
    for k in range(24):
        for h in range(24):
            values[:, k + 1, h] = 100 * k + h + 1000 * np.arange(days)
    inputs = np.arange(days * len(columns), dtype=float).reshape(days, len(columns))
    model = {
        "values": values,
        "columns": np.array(columns),
        "inputs": inputs,
        "groups": np.array(["other"] + ["demand_fcst"] * len(HOUR_COLUMNS)),
    }
    x, y = pairs(model, contributions(model), "demand_fcst")
    expected_y = [101 * h + 1000 * d for d in range(days) for h in range(24)]
    expected_x = [inputs[d, h + 1] for d in range(days) for h in range(24)]
    assert np.array_equal(y, expected_y)
    assert np.array_equal(x, expected_x)


def test_a_model_s_shares_slope_and_doubled_interactions_by_hand():
    # Dues files. Per grup, les contribucions sumades: demanda 2 i 6, el preu 1 + 1 i
    # -1 + 3, el calendari -1 i 1; mitjanes del valor absolut 4, 2 i 1, parts de 7. La
    # demanda a 1 i 3 GW contribueix 2 i 6: pendent 2 per GW. Interaccions de la
    # demanda amb les dues columnes del preu, 1 + 1 i -1 + 3, i amb el calendari, -1 i
    # 1: el doble, de mitjana 4 i 2.
    values = np.array([[2.0, 1.0, 1.0, -1.0], [6.0, -1.0, 3.0, 1.0]])
    interactions = np.zeros((2, 4, 4))
    interactions[:, 0, 1] = [1.0, -1.0]
    interactions[:, 0, 2] = [1.0, 3.0]
    interactions[:, 0, 3] = [-1.0, 1.0]
    model = {
        "values": values,
        "inputs": np.array([[1000.0, 0, 0, 0], [3000.0, 0, 0, 0]]),
        "columns": np.array(
            ["demand_fcst", "price_lag1_h00", "price_lag1_h01", "hour"]
        ),
        "groups": np.array(["demand_fcst", "price_lag1", "price_lag1", "calendar"]),
        "unit": np.array("EUR/MWh"),
        "interactions": interactions,
    }
    found = summarised(model)
    assert found["shares"] == pytest.approx(
        {"demand_fcst": 4 / 7, "price_lag1": 2 / 7, "calendar": 1 / 7}
    )
    assert list(found["shares"]) == ["demand_fcst", "price_lag1", "calendar"]
    assert found["slopes"] == {"demand_fcst": pytest.approx(2.0)}
    assert found["partners"] == {"demand_fcst": {"price_lag1": 4.0, "calendar": 2.0}}
    assert summarised(model | {"values": np.zeros_like(values)}) is None


def test_a_column_is_selected_at_an_hour_where_any_row_contributes():
    # Dos dies de 24 hores: la columna `y` contribueix només a les 5 del segon.
    values = np.ones((48, 2))
    values[:, 1] = 0.0
    values[24 + 5, 1] = 0.3
    model = {
        "values": values,
        "inputs": np.zeros((48, 2)),
        "columns": np.array(["x", "y"]),
        "groups": np.array(["calendar", "calendar"]),
        "unit": np.array("espai transformat"),
    }
    selected = summarised(model, selection=True)["selected"]
    assert selected[:, 0].all() and np.flatnonzero(selected[:, 1]).tolist() == [5]


def _summary(shares, slopes, **fields):
    """Torna el que `summarised` tornaria d'un model, amb els grups `a`, `b` i `c`."""
    return {
        "shares": dict(zip("abc", shares, strict=True)),
        "slopes": dict(zip("bc", slopes, strict=True)),
        "unit": "espai transformat",
    } | fields


def _measured(rows, measure, **keys):
    """Torna el valor de cada fila d'una mesura que té les claus."""
    return [
        row["value"]
        for row in rows
        if row["measure"] == measure
        and all(row.get(key) == value for key, value in keys.items())
    ]


def test_each_seed_counts_once_and_one_showing_nothing_counts_against():
    """L'elastic net deixa grups sencers a zero, com el nuclear i el carbó."""
    # Seed 0, dos models: parts de `a` 0.6 i 0.4, mitjana 0.5, empatada amb `b`;
    # pendents de `b` 2 i 4, mitjana 3. Seed 1: un model, `a` 0.7, pendent de `b` -1, i
    # un sense contribució. Seed 2: cap model amb contribució. `c` és zero a tot arreu.
    # La mitjana del pendent de `b`, 1, positiva: en contra la seed 1, negativa, i la 2.
    # `a` per sobre de `b` a la mitjana, 0.6 contra 0.4: en contra la 0, empatada, i la
    # 2. El de `c`, 0, no té signe: les tres en contra.
    selected = np.zeros((24, 2), dtype=bool)
    selected[:, 0] = True
    fits = {
        (0, 0): _summary(
            (0.6, 0.4, 0.0), (2.0, 0.0), columns=["x", "y"], selected=selected
        ),
        (0, 1): _summary(
            (0.4, 0.6, 0.0),
            (4.0, 0.0),
            columns=["x", "y"],
            selected=selected | (np.arange(24) == 5)[:, None],
        ),
        (1, 0): _summary(
            (0.7, 0.3, 0.0), (-1.0, 0.0), columns=["x", "y"], selected=selected
        ),
        (1, 1): None,
        (2, 0): None,
        (2, 1): None,
    }
    rows = part_ii_main({"elastic_net": fits}, {"elastic_net": {"selection": True}})
    assert _measured(rows, "share", subject="a") == [0.5, 0.7]
    assert _measured(rows, "share_mean", subject="a") == [pytest.approx(0.6)]
    # Desviació mostral de 0.5 i 0.7: √((0.1² + 0.1²) / 1).
    assert _measured(rows, "share_std", subject="a") == [pytest.approx(0.02**0.5)]
    assert _measured(rows, "slope_mean", subject="b") == [1.0]
    assert _measured(rows, "sign", subject="b") == [1]
    assert _measured(rows, "sign_against", subject="b") == [2]
    assert _measured(rows, "sign", subject="c") == [0]
    assert _measured(rows, "sign_against", subject="c") == [3]
    assert _measured(rows, "order_against", subject="a", other_subject="b") == [2]
    assert _measured(rows, "order_against", subject="a", other_subject="c") == [1]
    assert _measured(rows, "models_without_contribution") == [0, 1, 2]
    assert _measured(rows, "selection", subject="y", hour=5) == [pytest.approx(1 / 3)]
    assert _measured(rows, "selection", subject="x") == [1.0] * 25
    assert all(row.get("rank") is None for row in rows if row["measure"] == "share")


def test_a_partner_counts_where_it_is_among_a_seed_s_three_strongest():
    # A la seed 0 els tres més forts amb `s` són `g1`, `g3` i `g2`; a la 1, `g2`, `g4` i
    # `g1`. La força de `g1`, 3 i 1, de mitjana 2.
    first = {"g1": 3.0, "g2": 1.0, "g3": 2.0, "g4": 0.5}
    second = {"g1": 1.0, "g2": 4.0, "g3": 0.2, "g4": 3.0}
    fits = {
        (0, 0): _summary((0.5, 0.5, 0.0), (1.0, 1.0), partners={"s": first}),
        (0, 1): _summary((0.5, 0.5, 0.0), (1.0, 1.0)),
        (1, 0): _summary((0.5, 0.5, 0.0), (1.0, 1.0), partners={"s": second}),
    }
    rows = part_ii_main({"lightgbm": fits}, {"lightgbm": {}})
    seeds = {
        group: _measured(rows, "partner_seeds", subject="s", other_subject=group)
        for group in first
    }
    assert seeds == {"g1": [2], "g2": [2], "g3": [1], "g4": [1]}
    assert _measured(rows, "partner_strength_mean", other_subject="g1") == [2.0]
    assert _measured(rows, "partner_strength", seed=1, other_subject="g4") == [3.0]
