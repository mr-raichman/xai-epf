"""Model LEAR de Lago et al. (2021), la referència que els altres models han de superar.

Un LASSO per hora sobre preus transformats, reajustat cada dia sobre una finestra
de calibratge, i la mitjana aritmètica de les prediccions de cada finestra. La
lambda surt de LARS amb l'AIC de la mostra i el model final es reajusta per
descens de coordenades, tal com descriu l'article.

Implementació pròpia a partir de la descripció de l'article, no del seu codi.
"""
import collections

import numpy as np
from sklearn.linear_model import Lasso, LassoLarsIC
from threadpoolctl import threadpool_limits

from epf.models.lear.lear_hyperparameters import MAX_ITER, SHORT_LAGS, WINDOWS
from epf.parameters import DAYS_PER_WEEK, LAGS
from epf.market import HOURS_PER_DAY
from epf.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK_COLUMNS, calibration
from epf.models.scaling import Scaler

# Nom de la finestra que agafa tot l'historial.
WHOLE_HISTORY = "all"

# Nom del conjunt, que és la mitjana de les seves finestres, i nom de cada finestra.
# Cada finestra és un pronòstic amb el seu CSV i les seves mètriques, perquè el que
# una combinació compta són models ajustats: la mitjana no és un model, és una mitjana.
NAME = "lear"


def window_name(window):
    """Nom del pronòstic d'una finestra: `lear_49`, i `lear_all` per tot l'historial."""
    return f"{NAME}_{window if window else WHOLE_HISTORY}"


def window_names(windows=WINDOWS):
    return tuple(window_name(window) for window in windows)

# El que queda ajustat d'una finestra, i tot el que la predicció necessita. Els 24 LASSO
# hi són tal com es van ajustar i no com a matrius de coeficients, de manera que predir
# és exactament la crida que els va predir. Hi va la mida de la finestra i no els seus
# retards ni les seves columnes, que se'n deriven amb `lags_of`. `prices` és l'escalador
# de les entrades i `target` el de les 24 sortides, que és qui torna a EUR/MWh. No
# s'explica: LEAR és la referència, i les referències no s'interpreten.
Window = collections.namedtuple("Window", "window models prices target")

def _design(rows, scaler):
    """Files de `matrix.build` com la matriu de LEAR: preus transformats i indicadors.

    Els indicadors no s'escalen: ja són zeros i uns, i els escriu la matriu.
    """
    values = rows.to_numpy()
    return np.hstack([scaler.forward(values[:, :-DAYS_PER_WEEK]),
                      values[:, -DAYS_PER_WEEK:]])


def _fit_hour(features, target, max_iter):
    """Un LASSO per una hora: lambda per LARS amb AIC, ajust per descens de coordenades.

    El sostre d'iteracions ve de la configuració: un ajust tallat a mitges no es
    queixa, i a 2500 la meitat dels de les finestres llargues ho estaven.
    """
    alpha = LassoLarsIC(criterion="aic", max_iter=max_iter).fit(features, target).alpha_
    return Lasso(alpha=alpha, max_iter=max_iter).fit(features, target)


def lags_of(window):
    """Retards que porta una finestra: els de l'etapa, o menys si és curta."""
    return SHORT_LAGS.get(window, LAGS)


def feature_columns(lags):
    """Columnes de `matrix.build` que fa servir una finestra, en el mateix ordre.

    Són també les de la matriu de disseny: l'escalat no canvia cap nom i els
    indicadors ja arriben oberts.
    """
    return ([f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS]
            + DAY_OF_WEEK_COLUMNS)


def _fit_window(X, y, day, window, max_iter):
    """Els 24 LASSO d'una finestra, ajustats només amb dies anteriors, i els dos
    escaladors que cal per llegir-los."""
    lags = lags_of(window)
    columns = feature_columns(lags)
    features, target = calibration(X[columns], y, day, window)
    width = len(columns)
    if len(features) <= width + 1:
        raise ValueError(f"{len(features)} dies de calibratge per a {day}, "
                         f"insuficients per a {width} columnes")

    prices = Scaler(features.to_numpy()[:, :-DAYS_PER_WEEK])
    prices_target = Scaler(target.to_numpy())
    train = _design(features, prices)
    scaled = prices_target.forward(target.to_numpy())

    # Un fil, a l'ajust i a la lectura. La matriu de disseny és de 1085 per 103 i a
    # OpenBLAS li costa més repartir-la entre setze fils que fer-la en un: mesurat
    # sobre tres dies, un fil va 2,05 a 2,18 vegades més ràpid i dóna el mateix
    # resultat bit a bit. El límit és d'aquesta família i no de l'execució, perquè les
    # altres sí que aprofiten els setze.
    with threadpool_limits(limits=1):
        models = [_fit_hour(train, scaled[:, hour], max_iter)
                  for hour in range(HOURS_PER_DAY)]
    return Window(window, models, prices, prices_target)


def _rows_of(fitted, X, day):
    """La fila del dia que es prediu, escalada amb l'escalador de la finestra."""
    columns = feature_columns(lags_of(fitted.window))
    return _design(X.loc[[day], columns], fitted.prices)


def _window_prices(fitted, rows):
    """Els 24 preus d'una finestra, en EUR/MWh: els models donen l'espai transformat i
    l'escalador de l'objectiu el desfà."""
    with threadpool_limits(limits=1):
        outputs = np.array([model.predict(rows)[0] for model in fitted.models])
    return fitted.target.back(outputs)


def fit(X, y, day, windows=WINDOWS, max_iter=MAX_ITER):
    """El conjunt ajustat: una finestra per cada mida de `windows`.

    `X` i `y` són els de `matrix.build`, i una finestra són dies de calibratge, com a
    tot el projecte.
    """
    return tuple(_fit_window(X, y, day, window, max_iter) for window in windows)


def predict(model, X, day):
    """Cada finestra i la mitjana aritmètica de totes, per nom.

    La mitjana és el que l'article anomena LEAR i és la que es compara amb els altres
    models; les finestres hi són perquè cada una és un model ajustat, i és això el que
    entra a la combinació d'una execució.
    """
    prices = {window_name(fitted.window): _window_prices(fitted, _rows_of(fitted, X, day))
              for fitted in model}
    return {NAME: np.mean(list(prices.values()), axis=0), **prices}

