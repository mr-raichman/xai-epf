"""La transformació dels models lineals: per columna, d'anada i tornada, i què fa amb una
columna sense MAD."""
import numpy as np
import pytest

from epf.models.scaling import Scaler

# La previsió solar a les 20 h: zero més de la meitat dels dies, milers de MW la resta,
# de manera que la seva MAD és zero tot i que es mou.
EVENING_SOLAR = np.array([0.0] * 6 + [900.0, 2500.0, 4641.0, 1200.0])


def test_the_scaler_round_trips():
    values = np.array([[0.0, 5.0], [100.0, -3.0]])
    scaler = Scaler(values)
    assert np.allclose(scaler.back(scaler.forward(values)), values)


def test_each_column_gets_its_own_median_and_spread():
    # Una columna al voltant de 0 i una altra al voltant de 1000. Amb una mediana
    # compartida, la primera columna quedaria tota desplaçada cap avall.
    values = np.array([[0.0, 1000.0], [1.0, 1002.0], [2.0, 1004.0], [3.0, 1006.0]])
    scaler = Scaler(values)
    assert list(scaler.median) == [1.5, 1003.0]
    transformed = scaler.forward(values)
    assert abs(transformed[:, 0].mean()) < 1e-9
    assert abs(transformed[:, 1].mean()) < 1e-9
    assert np.allclose(scaler.back(transformed), values)


def test_a_window_without_spread_fails_loudly():
    with pytest.raises(ValueError):
        Scaler(np.zeros((10, 24)))


def test_a_column_without_mad_is_refused_unless_the_caller_declares_the_fallback():
    values = np.column_stack([np.arange(10.0), EVENING_SOLAR])
    with pytest.raises(ValueError):
        Scaler(values)
    scaler = Scaler(values, std_where_flat=True)
    assert scaler.scale[1] == np.std(EVENING_SOLAR)
    assert np.allclose(scaler.back(scaler.forward(values)), values)


def test_the_fallback_still_refuses_a_column_that_never_moves():
    with pytest.raises(ValueError):
        Scaler(np.column_stack([np.arange(10.0), np.zeros(10)]), std_where_flat=True)
