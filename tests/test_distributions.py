import datetime

import keras
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from epf import Refusal
from epf.contract.metrics import PERCENTILES, crps
from epf.models.nn.ddnn.distributions import JOHNSON_SU, NORMAL
from epf.models.nn.network import parameters
from epf.parameters import HOUR_COLUMNS

DAYS = 40


def frame_of(values):
    """Torna `values` com a preus de `DAYS` dies des del 2024-01-01."""
    index = pd.DatetimeIndex(
        [
            datetime.date(2024, 1, 1) + datetime.timedelta(days=day)
            for day in range(DAYS)
        ],
        name="day",
    )
    return pd.DataFrame(values, index=index, columns=HOUR_COLUMNS)


def normal_parameters(location, scale):
    return np.stack(
        [
            np.full((DAYS, len(HOUR_COLUMNS)), location),
            np.full((DAYS, len(HOUR_COLUMNS)), scale),
        ],
        axis=-1,
    )


def test_a_symmetric_johnson_su_has_its_location_as_median():
    """Amb l'asimetria i el pes de la cua canviats de lloc, res més no se n'adonaria."""
    # Amb asimetria zero, la de Johnson SU és simètrica al voltant de la localització.
    parameters = np.stack(
        [
            np.full((DAYS, 24), 60.0),
            np.full((DAYS, 24), 5.0),
            np.zeros((DAYS, 24)),
            np.full((DAYS, 24), 2.0),
        ],
        axis=-1,
    )
    assert np.allclose(JOHNSON_SU.point(parameters), 60.0, atol=1e-9)


def test_a_positive_skew_moves_the_median_one_way_and_a_negative_the_other():
    def median(skew):
        parameters = np.stack(
            [
                np.full((DAYS, 24), 60.0),
                np.full((DAYS, 24), 5.0),
                np.full((DAYS, 24), skew),
                np.full((DAYS, 24), 2.0),
            ],
            axis=-1,
        )
        return JOHNSON_SU.point(parameters)[0, 0]

    assert median(1.0) < 60.0 < median(-1.0)


def test_the_johnson_su_loss_is_its_density_far_below_the_location():
    """Amb asinh com a `log(z + sqrt(1 + z^2))`, en float32 la pèrdua es desviava.

    Ho feia a z = -1000; a z = -4097 hi faltava més d'una unitat sencera d'asinh, i a
    z = -10000 era infinita. La de Marcjasz et al. (2023) és la densitat de TFP.
    """
    z = np.array([-1000.0, -4097.0, -10000.0], dtype="float32")[:, None]
    raw = np.zeros((len(z), len(JOHNSON_SU.names) * len(HOUR_COLUMNS)), dtype="float32")
    raw[:, len(HOUR_COLUMNS) : 2 * len(HOUR_COLUMNS)] = -5.0
    location, scale, skew, tail = np.moveaxis(parameters(JOHNSON_SU, raw), -1, 0)
    target = (location + z * scale).astype("float32")
    loss = keras.ops.convert_to_numpy(JOHNSON_SU.loss(target, raw))
    # La pèrdua deixa fora la constant de la normal.
    density = stats.johnsonsu.logpdf(
        target.astype(float), skew, tail, loc=location, scale=scale
    )
    expected = (-density - 0.5 * np.log(2 * np.pi)).mean(axis=1)
    np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_the_normals_median_is_its_location():
    parameters = normal_parameters(60.0, 5.0)
    assert np.allclose(NORMAL.point(parameters), 60.0)


def test_the_quantiles_come_out_in_order_and_straddle_the_median():
    quantiles = NORMAL.quantiles(normal_parameters(60.0, 5.0), PERCENTILES)
    assert quantiles.shape == (DAYS, len(HOUR_COLUMNS), len(PERCENTILES))
    assert (np.diff(quantiles, axis=-1) > 0).all()
    assert quantiles[0, 0, len(PERCENTILES) // 2] == pytest.approx(60.0)


def test_a_sharper_distribution_scores_better_when_it_is_centred_right():
    """Un pinball amb el signe al revés donaria un CRPS plausible i equivocat.

    Cap CRPS de referència per a aquest mercat no ho delataria.
    """
    actual = frame_of(60.0)
    sharp = crps(actual, NORMAL.quantiles(normal_parameters(60.0, 1.0), PERCENTILES))
    wide = crps(actual, NORMAL.quantiles(normal_parameters(60.0, 20.0), PERCENTILES))
    assert sharp < wide


def test_a_sharper_distribution_scores_worse_when_it_is_centred_wrong():
    """La meitat que fa del CRPS una mesura, i no un premi a la confiança."""
    actual = frame_of(60.0)
    sharp = crps(actual, NORMAL.quantiles(normal_parameters(90.0, 1.0), PERCENTILES))
    wide = crps(actual, NORMAL.quantiles(normal_parameters(90.0, 20.0), PERCENTILES))
    assert sharp > wide


def test_the_best_scale_is_the_one_the_prices_really_have():
    rng = np.random.default_rng(0)
    actual = frame_of(rng.normal(60.0, 8.0, size=(DAYS, len(HOUR_COLUMNS))))
    scores = {
        scale: crps(
            actual, NORMAL.quantiles(normal_parameters(60.0, scale), PERCENTILES)
        )
        for scale in (1.0, 4.0, 8.0, 16.0, 40.0)
    }
    assert min(scores, key=scores.get) == 8.0


def test_a_non_finite_quantile_is_refused_rather_than_averaged_away():
    actual = frame_of(60.0)
    quantiles = NORMAL.quantiles(normal_parameters(60.0, 5.0), PERCENTILES)
    quantiles[3, 2, 10] = np.inf
    with pytest.raises(Refusal, match="no finits"):
        crps(actual, quantiles)
