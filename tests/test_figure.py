"""Les figures dels models explicats: la importància per grups i la parella a l'hora."""
import numpy as np

from epf.contract.figure import outputs, pairs, shares
from epf.parameters import HOUR_COLUMNS


def test_a_group_sums_its_columns_before_the_absolute_value():
    # Dues columnes del mateix grup que s'anul·len: per columna pesen, com a grup no.
    values = np.array([[1.0, -1.0, 2.0], [1.0, -1.0, -2.0]])
    by_column, by_group = shares(values, np.array(["a", "a", "b"]))
    assert np.allclose(by_column, [0.25, 0.25, 0.5])
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
    seed = {"values": values, "names": np.array(["location"]),
            "columns": np.array(columns), "inputs": inputs,
            "groups": np.array(["other"] + ["demand_fcst"] * len(HOUR_COLUMNS))}
    x, y = pairs(seed, "location", outputs(seed)["location"], "demand_fcst")
    expected_y = [101 * h + 1000 * d for d in range(days) for h in range(24)]
    expected_x = [inputs[d, h + 1] for d in range(days) for h in range(24)]
    assert np.array_equal(y, expected_y)
    assert np.array_equal(x, expected_x)
