"""Model LEAR de Lago et al. (2021), la referència que els altres models han de superar.

Un LASSO per hora sobre preus transformats, reajustat cada dia sobre una finestra
de calibratge, i la mitjana aritmètica de les prediccions de cada finestra. La
lambda surt de LARS amb l'AIC de la mostra i el model final es reajusta per
descens de coordenades, tal com descriu l'article.

Implementació pròpia a partir de la descripció de l'article, no del seu codi.
"""
import numpy as np
from sklearn.linear_model import Lasso, LassoLarsIC

from epf.config.market import HOURS_PER_DAY
from epf.features.matrix import PRICE_LAGS, calibration

DAYS_PER_WEEK = 7

# Dies que es perden al principi de qualsevol tram: cap fila es pot formar fins que
# hi ha tots els retards.
WARM_UP = max(PRICE_LAGS)

# Finestres de calibratge, en files. Els "anys" de l'article són de 364 dies, 52
# setmanes exactes que mantenen alineat el dia de la setmana; les seves finestres
# de 1092 i 1456 dies de preus donen aquestes files un cop descomptat el començament.
# None vol dir tot l'historial disponible, que l'article no fa servir.
WINDOWS = (364 * 3 - WARM_UP, 364 * 4 - WARM_UP, None)

# La mediana de les desviacions absolutes dividida per aquesta constant estima la
# desviació típica d'una normal; és el que fa `statsmodels.robust.mad`, i el que
# la referència utilitza encara que l'article només digui "MAD".
NORMAL_CONSISTENCY = 0.6744897501960817

MAX_ITER = 2500


class Scaler:
    """Transformació de variància estabilitzada: asinh del valor centrat i escalat.

    Mediana i MAD de cada columna per separat, preses de la finestra de calibratge i
    mai del dia que es prediu.
    """

    def __init__(self, values):
        self.median = np.median(values, axis=0)
        deviation = np.median(np.abs(values - self.median), axis=0)
        if not deviation.all():
            raise ValueError("alguna columna de la finestra de calibratge no té dispersió")
        self.mad = deviation / NORMAL_CONSISTENCY

    def forward(self, values):
        return np.arcsinh((values - self.median) / self.mad)

    def back(self, values):
        return np.sinh(values) * self.mad + self.median


def week_dummies(day_of_week):
    """El dia de la setmana, enter, en set indicadors, que és el que LEAR necessita."""
    return np.eye(DAYS_PER_WEEK)[np.asarray(day_of_week, dtype=int)]


def _design(rows, scaler):
    """Files de `matrix.build` com la matriu de LEAR: preus transformats i indicadors.

    Els indicadors no s'escalen: ja són zeros i uns.
    """
    values = rows.to_numpy()
    return np.hstack([scaler.forward(values[:, :-1]), week_dummies(values[:, -1])])


def _fit_hour(features, target):
    """Un LASSO per una hora: lambda per LARS amb AIC, ajust per descens de coordenades."""
    alpha = LassoLarsIC(criterion="aic", max_iter=MAX_ITER).fit(features, target).alpha_
    return Lasso(alpha=alpha, max_iter=MAX_ITER).fit(features, target)


def forecast(X, y, day, window=None):
    """Els 24 preus de `day`, amb el model ajustat només amb dies anteriors.

    `X` i `y` són els de `features.matrix.build`, i la finestra són files, com a tot
    el projecte.
    """
    features, target = calibration(X, y, day, window)
    if len(features) <= X.shape[1] + DAYS_PER_WEEK:
        raise ValueError(f"{len(features)} dies de calibratge per a {day}, "
                         f"insuficients per a {X.shape[1] + DAYS_PER_WEEK} columnes")

    prices = Scaler(features.to_numpy()[:, :-1])
    prices_target = Scaler(target.to_numpy())
    train = _design(features, prices)
    test = _design(X.loc[[day]], prices)
    scaled = prices_target.forward(target.to_numpy())
    hours = [_fit_hour(train, scaled[:, hour]).predict(test)[0] for hour in range(HOURS_PER_DAY)]
    return prices_target.back(np.array(hours))


def forecast_ensemble(X, y, day, windows=WINDOWS):
    """Mitjana aritmètica de les prediccions de cada finestra de calibratge."""
    return np.mean([forecast(X, y, day, window) for window in windows], axis=0)
