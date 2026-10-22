import datetime

import keras
import numpy as np
import optuna
import pandas as pd
import pytest
from scipy import stats

from epf import Refusal
from epf.contract.matrix import DAY_OF_WEEK_COLUMNS
from epf.contract.metrics import PERCENTILES, crps
from epf.contract.split import split
from epf.models.nn import network, search
from epf.models.nn.ddnn import train as ddnn
from epf.models.nn.ddnn.attribution import attribute_rows
from epf.models.nn.ddnn.distributions import JOHNSON_SU, NORMAL
from epf.models.nn.dnn import train as dnn
from epf.models.nn.network import build, fit_rows, parameters, parameters_rows
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, HOUR_COLUMNS

# Una xarxa petita i un entrenament curt: el que es comprova és la suma, que no depèn de
# com d'ajustada estigui.
TUNED = {
    "hidden": [8, 8],
    "activation": ["relu", "softplus"],
    "dropout": 0.1,
    "l1": [0.0, 1e-4],
    "activity_l1": [1e-4, 0.0],
    "learning_rate": 0.01,
}


def tuned(distribution):
    """Torna els números de prova amb una L1 per paràmetre de `distribution`."""
    return TUNED | {"parameter_l1": [1e-4] * len(distribution.names)}


def rows(count, seed=0):
    """Torna `count` files, sis columnes i les dummies del dia, i els seus preus.

    La xarxa llegeix les set dummies en una sola columna.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(50, 20, size=(count, 6)), columns=[f"x{n}" for n in range(6)]
    )
    weights = rng.normal(size=(6, 24))
    y = pd.DataFrame(
        X.to_numpy() @ weights / 6 + rng.normal(size=(count, 24)), columns=HOUR_COLUMNS
    )
    week = len(DAY_OF_WEEK_COLUMNS)
    days = pd.DataFrame(
        np.eye(week)[np.arange(count) % week], columns=DAY_OF_WEEK_COLUMNS
    )
    return pd.concat([X, days], axis="columns"), y


@pytest.mark.parametrize("distribution", [JOHNSON_SU, NORMAL], ids=["jsu", "normal"])
def test_each_parameters_base_and_contributions_add_up_to_it(distribution):
    # La base i les contribucions de DeepExplainer contra la predicció de la xarxa, a
    # través de la normalització per lots de l'entrada i de les L1 de la sortida.
    X, y = rows(150)
    fitted = fit_rows(
        distribution,
        (X[:120], y[:120]),
        (X[120:], y[120:]),
        tuned(distribution),
        0,
        batch=16,
        epochs=5,
        patience=2,
    )
    explained = attribute_rows(distribution, fitted, X[:5], background=X[:120])
    direct = parameters_rows(distribution, fitted, X[:5])
    assert explained.values.shape == (5, 7, 24, len(distribution.names))
    rebuilt = explained.base + explained.values.sum(axis=1)
    assert np.allclose(rebuilt, direct, atol=1e-3)


def test_a_parameter_l1_for_another_distribution_fails():
    """La pèrdua llegiria menys sortides de les que la xarxa treu, sense fallar.

    Amb quatre caps i una pèrdua de dos paràmetres, en llegiria dos, i la xarxa
    n'aprendria quatre.
    """
    with pytest.raises(ValueError) as raised:
        build(NORMAL, 6, tuned(JOHNSON_SU))
    assert not isinstance(raised.value, Refusal)


def test_the_tail_weight_never_drops_below_one():
    """Marcjasz et al. (2023) no permeten un pes de la cua sota 1."""
    # Com més negativa la sortida crua, més a prop d'1 queda el pes de la cua.
    raw = np.full((3, len(JOHNSON_SU.names) * 24), -50.0, dtype="float32")
    tail = parameters(JOHNSON_SU, raw)[..., JOHNSON_SU.names.index("tail_weight")]
    assert (tail >= 1).all()
    assert np.allclose(tail, 1)


# Una xarxa de l'espai, sense cap L1 als pesos ni a les sortides de les capes amagades.
PARAMS = {
    "neurons_1": 32,
    "neurons_2": 16,
    "activation_1": "relu",
    "activation_2": "elu",
    "dropout": False,
    "regularize_h1_kernel": False,
    "regularize_h2_kernel": False,
    "regularize_h1_activation": False,
    "regularize_h2_activation": False,
    "learning_rate": 1e-3,
}


def test_each_parameters_l1_is_the_one_named_after_it():
    """Per posició, les L1 de l'asimetria i del pes de la cua s'intercanviarien."""
    rates = {"location": 1e-4, "scale": 1e-3, "skewness": 1e-2, "tail_weight": 1e-1}
    params = PARAMS | {f"regularize_{name}": True for name in rates}
    params |= {f"{name}_l1": rate for name, rate in rates.items()}
    tuned = search.drawn("ddnn_jsu", optuna.trial.FixedTrial(params))
    assert tuned["parameter_l1"] == [rates[name] for name in JOHNSON_SU.names]


def test_the_point_networks_space_asks_for_no_output_penalty():
    # Una L1 a la sortida faria fallar `FixedTrial`, que no en té el paràmetre.
    tuned = search.drawn("dnn", optuna.trial.FixedTrial(PARAMS))
    assert tuned["parameter_l1"] == [0.0]


def test_a_trial_and_the_run_hand_each_networks_fit_the_same_seed(monkeypatch):
    """Amb una altra seed, la passada no ajustaria el que la cerca va puntuar."""

    class Stopped(Exception):
        """L'ajust fals s'atura aquí, un cop desada la seed."""

    seen = []

    def recorded(distribution, train, validate, tuned, seed, **keywords):
        seen.append(seed)
        raise Stopped

    monkeypatch.setattr(network, "fit_rows", recorded)
    X, y = rows(10)
    parts = split(3, DEVELOPMENT, BLACKOUT_WEEK)
    numbers = tuned(JOHNSON_SU)
    for fit in (
        lambda: search.trial("ddnn_jsu", numbers, (X, y), (X, y), 3),
        lambda: ddnn._fitted(JOHNSON_SU, X, y, parts, parts.test, 3, numbers),
        lambda: dnn._fitted(X, y, parts, 3, tuned(dnn.POINT)),
    ):
        with pytest.raises(Stopped):
            fit()
    assert seen == [3, 3, 3]


DAYS = 40


def prices_of(values):
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
    actual = prices_of(60.0)
    sharp = crps(actual, NORMAL.quantiles(normal_parameters(60.0, 1.0), PERCENTILES))
    wide = crps(actual, NORMAL.quantiles(normal_parameters(60.0, 20.0), PERCENTILES))
    assert sharp < wide


def test_a_sharper_distribution_scores_worse_when_it_is_centred_wrong():
    """La meitat que fa del CRPS una mesura, i no un premi a la confiança."""
    actual = prices_of(60.0)
    sharp = crps(actual, NORMAL.quantiles(normal_parameters(90.0, 1.0), PERCENTILES))
    wide = crps(actual, NORMAL.quantiles(normal_parameters(90.0, 20.0), PERCENTILES))
    assert sharp > wide


def test_the_best_scale_is_the_one_the_prices_really_have():
    rng = np.random.default_rng(0)
    actual = prices_of(rng.normal(60.0, 8.0, size=(DAYS, len(HOUR_COLUMNS))))
    scores = {
        scale: crps(
            actual, NORMAL.quantiles(normal_parameters(60.0, scale), PERCENTILES)
        )
        for scale in (1.0, 4.0, 8.0, 16.0, 40.0)
    }
    assert min(scores, key=scores.get) == 8.0


def test_a_non_finite_quantile_is_refused_rather_than_averaged_away():
    actual = prices_of(60.0)
    quantiles = NORMAL.quantiles(normal_parameters(60.0, 5.0), PERCENTILES)
    quantiles[3, 2, 10] = np.inf
    with pytest.raises(Refusal, match="no finits"):
        crps(actual, quantiles)
