import warnings

import numpy as np

from epf.contract import figure
from epf.contract.figure import importances, outputs, pairs
from epf.parameters import HOUR_COLUMNS


def test_a_group_sums_its_columns_before_the_absolute_value():
    # Dues columnes del mateix grup que s'anul·len: per columna pesen, com a grup no.
    values = np.array([[1.0, -1.0, 2.0], [1.0, -1.0, -2.0]])
    by_column, by_group = importances(values, np.array(["a", "a", "b"]))
    assert np.allclose(by_column, [0.5, 0.5, 1.0])
    assert by_group == {"a": 0.0, "b": 1.0}


def test_a_distributional_networks_forecast_is_paired_with_its_own_hour():
    # La contribució de la columna de l'hora k a la sortida de l'hora h val 100 k + h:
    # la parella a l'hora només ha de prendre les de k igual a h. Una altra columna al
    # davant, perquè la posició d'una columna no coincideixi amb la seva hora.
    columns = ["other"] + [f"demand_fcst_{hour}" for hour in HOUR_COLUMNS]
    days = 2
    values = np.zeros((days, len(columns), 24, 1))
    for k in range(24):
        for h in range(24):
            values[:, k + 1, h, 0] = 100 * k + h + 1000 * np.arange(days)
    inputs = np.arange(days * len(columns), dtype=float).reshape(days, len(columns))
    model = {
        "values": values,
        "names": np.array(["location"]),
        "columns": np.array(columns),
        "inputs": inputs,
        "groups": np.array(["other"] + ["demand_fcst"] * len(HOUR_COLUMNS)),
    }
    x, y = pairs(model, outputs(model)["location"], "demand_fcst")
    expected_y = [101 * h + 1000 * d for d in range(days) for h in range(24)]
    expected_x = [inputs[d, h + 1] for d in range(days) for h in range(24)]
    assert np.array_equal(y, expected_y)
    assert np.array_equal(x, expected_x)


def test_a_model_with_no_contribution_is_left_out_of_the_importance(tmp_path):
    """Una xarxa amb tots els pesos d'entrada a zero ho prediu tot igual."""
    groups = np.array(["price", "price", "calendar"])
    model = {
        "values": np.array([[1.0, -1.0, 2.0]]),
        "groups": groups,
        "columns": np.array(["x", "y", "z"]),
    }
    constant = model | {"values": np.zeros((1, 3))}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure.importance(tmp_path, [model, constant], figure.OTHER_COLOUR)
        figure.importance(tmp_path, [constant], figure.OTHER_COLOUR)
    assert {path.stem for path in tmp_path.iterdir()} >= {"importance", "groups"}


def test_a_number_is_written_with_the_memorias_separators():
    # Xifres que les figures escriuen: la demanda en MW, una diferència amb signe, el
    # menys que matplotlib ja hi posa, una R², un pendent minúscul i un nom de mes, que
    # no és cap nombre.
    assert figure.catalan("30000") == "30.000"
    assert figure.catalan("1234567.5") == "1.234.567,5"
    assert figure.catalan("-12.3") == "\N{MINUS SIGN}12,3"
    assert figure.catalan("\N{MINUS SIGN}150") == "\N{MINUS SIGN}150"
    assert figure.catalan("0.87") == "0,87"
    assert figure.catalan("999") == "999"
    assert figure.catalan("-3.98e-05") == "\N{MINUS SIGN}3,98e\N{MINUS SIGN}05"
    assert figure.catalan("gener") == "gener"
