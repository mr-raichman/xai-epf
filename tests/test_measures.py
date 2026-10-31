import numpy as np
import pytest

from epf.contract.measures import (
    _order,
    contributions,
    counter_share,
    importances,
    pairs,
    part_ii_main,
    part_ii_secondary,
    stability,
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


def test_a_seed_s_stability_by_hand():
    # Dos models, dues files, un grup: 1 i 3, i 3 i 3. Desviació mostral de cada fila,
    # √2 i 0, de mitjana √2 / 2; contribució absoluta mitjana (1 + 3 + 3 + 3) / 4.
    models = [
        {"rows": np.array([[1.0], [3.0]]), "magnitudes": {"a": 2.0}},
        {"rows": np.array([[3.0], [3.0]]), "magnitudes": {"a": 3.0}},
    ]
    deviation, magnitude = stability(models)["a"]
    assert deviation == pytest.approx(2**0.5 / 2) and magnitude == 2.5
    assert np.isnan(stability(models[:1])["a"][0])


def test_a_group_order_writes_a_tie_as_one():
    """L'elastic net deixa dos grups a zero, i cap no va per sobre de l'altre."""
    assert _order({"a": 0.25, "b": 0.5, "c": 0.25}) == "b > a = c"


def test_the_secondary_measures_by_hand():
    # Tres models, dos a la seed 0. Parts (0.6, 0.3, 0.1), (0.4, 0.5, 0.1) i
    # (0.7, 0.2, 0.1): rangs (3, 2, 1), (2, 3, 1) i (3, 2, 1), Spearman 0.5, 1 i 0.5,
    # de mitjana 2 / 3; dos ordres, un de dos models. Entre seeds, de la part de `a`,
    # 6, 4 i 7 desenes: 24 / 42 de la suma de quadrats. Magnituds de `a` i `b`, 2 i 1,
    # 1 i 2, 3 i 1: diferències 1, -1 i 2, de mitjana positiva, i un model en contra.
    fits = {
        (0, 0): _summary(
            (0.6, 0.3, 0.1), (1.0, 1.0), magnitudes={"a": 2.0, "b": 1.0, "c": 0.1}
        ),
        (0, 1): _summary(
            (0.4, 0.5, 0.1), (1.0, 1.0), magnitudes={"a": 1.0, "b": 2.0, "c": 0.1}
        ),
        (1, 0): _summary(
            (0.7, 0.2, 0.1), (1.0, 1.0), magnitudes={"a": 3.0, "b": 1.0, "c": 0.1}
        ),
        (1, 1): None,
    }
    stabilities = {"lightgbm": {0: {name: (0.1, 1.0) for name in "abc"}}}
    rows = part_ii_secondary({"lightgbm": fits}, stabilities, {})
    assert _measured(rows, "spearman_mean") == [pytest.approx(2 / 3)]
    assert _measured(rows, "group_order") == [2, 1]
    assert _measured(rows, "group_order", subject="b > a > c") == [1]
    assert _measured(rows, "between_seeds_share", subject="a") == [pytest.approx(4 / 7)]
    assert _measured(rows, "flip_rate", subject="a", other_subject="b") == [
        pytest.approx(1 / 3)
    ]
    assert _measured(rows, "stability_index", subject="a") == [pytest.approx(0.1)]
    assert len(_measured(rows, "model_share")) == 9
    # Els pendents, tots 1, no tenen dispersió.
    assert np.isnan(_measured(rows, "between_seeds_slope", subject="b")[0])


def test_two_families_agreements_by_hand():
    # Parts (0.5, 0.3, 0.2) i (0.2, 0.3, 0.5): rànquings oposats, -1, cap parell
    # d'acord, els tres primers els mateixos però només el del mig al seu lloc; pendents
    # (1, -1) i (1, 1), un signe de dos.
    magnitudes = {"a": 1.0, "b": 2.0, "c": 3.0}
    first = {(0, 0): _summary((0.5, 0.3, 0.2), (1.0, -1.0), magnitudes=magnitudes)}
    second = {(0, 0): _summary((0.2, 0.3, 0.5), (1.0, 1.0), magnitudes=magnitudes)}
    stabilities = {
        family: {0: {name: (0.1, 1.0) for name in "abc"}}
        for family in ("dnn", "lightgbm")
    }
    rows = part_ii_secondary({"dnn": first, "lightgbm": second}, stabilities, {})
    found = {
        name: _measured(rows, name, family="dnn", other_family="lightgbm", seed=0)
        for name in (
            "rank_correlation",
            "pairwise_rank_agreement",
            "feature_agreement",
            "rank_agreement",
            "sign_agreement",
        )
    }
    assert found == {
        "rank_correlation": [pytest.approx(-1.0)],
        "pairwise_rank_agreement": [0.0],
        "feature_agreement": [1.0],
        "rank_agreement": [pytest.approx(1 / 3)],
        "sign_agreement": [0.5],
    }


def test_the_late_evening_and_the_own_hour_by_hand():
    # Un dia: la columna de l'hora k del preu contribueix k + 100 h a l'hora prevista
    # h. A la mateixa hora, 101 h; de mitjana a les 21, 22 i 23, 22 + 100 h.
    hours = np.arange(24)
    values = hours[None, :] + 100.0 * hours[:, None]
    model = {
        "values": values,
        "inputs": np.zeros((24, 24)),
        "columns": np.array([f"price_lag1_{hour}" for hour in HOUR_COLUMNS]),
        "groups": np.array(["price_lag1"] * 24),
        "unit": np.array("EUR/MWh"),
    }
    found = summarised(model)
    assert found["own_hour"] == pytest.approx(101.0 * hours)
    assert found["late_evening"] == pytest.approx(22 + 100.0 * hours)


def test_the_date_probe_s_counter_share_by_hand():
    # Mitjanes del valor absolut 1, 2 i 2: la part del comptador és 2 de 5.
    model = {
        "values": np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -1.0]]),
        "columns": np.array(["price_lag1_h00", "day_counter", "hour"]),
        "groups": np.array(["price_lag1", "date", "calendar"]),
    }
    assert counter_share(model, "day_counter") == pytest.approx(0.4)


def test_a_group_no_model_reads_has_no_stability_index():
    """Un grup que l'elastic net deixa a zero a tot arreu no té cap importància."""
    fits = {
        (0, 0): _summary(
            (0.6, 0.4, 0.0), (1.0, 0.0), magnitudes={"a": 2.0, "b": 1.0, "c": 0.0}
        ),
        (0, 1): _summary(
            (0.5, 0.5, 0.0), (2.0, 0.0), magnitudes={"a": 1.0, "b": 1.0, "c": 0.0}
        ),
    }
    stabilities = {
        "elastic_net": {0: {"a": (0.1, 1.5), "b": (0.0, 1.0), "c": (0.0, 0.0)}}
    }
    rows = part_ii_secondary({"elastic_net": fits}, stabilities, {})
    assert np.isnan(_measured(rows, "stability_index", subject="c")[0])
    assert _measured(rows, "stability_index", subject="a") == [pytest.approx(0.1 / 1.5)]
