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
from epf.features.matrix import calibration

# Finestres de calibratge en dies. Els "anys" de l'article són de 364 dies, que
# són 52 setmanes exactes i mantenen alineat el dia de la setmana. None vol dir
# tot l'historial disponible, que l'article no fa servir.
WINDOWS = (364 * 3, 364 * 4, None)

DAYS_PER_WEEK = 7

# El MAD de l'article és la mediana de les desviacions absolutes, sense cap
# constant. `statsmodels.robust.mad` divideix per 0.6745 per defecte, i com que
# asinh no és lineal les dues normalitzacions donen models diferents.
MAD_SCALE = 1.0

MAX_ITER = 5000


class Scaler:
    """Transformació de variància estabilitzada: asinh del preu centrat i escalat.

    La mediana i el MAD es prenen de la finestra de calibratge, mai del dia que es
    prediu.
    """

    def __init__(self, values):
        self.median = float(np.median(values))
        deviation = float(np.median(np.abs(values - self.median)))
        if deviation == 0:
            raise ValueError("la finestra de calibratge no té dispersió: MAD zero")
        self.mad = deviation / MAD_SCALE

    def forward(self, values):
        return np.arcsinh((values - self.median) / self.mad)

    def back(self, values):
        return np.sinh(values) * self.mad + self.median


def week_dummies(day_of_week):
    """El dia de la setmana, enter, en set indicadors, que és el que LEAR necessita."""
    return np.eye(DAYS_PER_WEEK)[np.asarray(day_of_week, dtype=int)]


def _design(rows, scaler):
    """Files de `matrix.build` com la matriu de LEAR: preus transformats i indicadors."""
    values = rows.to_numpy()
    return np.hstack([scaler.forward(values[:, :-1]), week_dummies(values[:, -1])])


def _fit_hour(features, target):
    """Un LASSO per una hora: lambda per LARS amb AIC, ajust per descens de coordenades."""
    alpha = LassoLarsIC(criterion="aic").fit(features, target).alpha_
    return Lasso(alpha=alpha, max_iter=MAX_ITER).fit(features, target)


def forecast(X, y, day, window=None):
    """Els 24 preus de `day`, amb el model ajustat només amb dies anteriors.

    `X` i `y` són els de `features.matrix.build`. La finestra són els últims
    `window` dies disponibles abans de `day`, o tots si és None.
    """
    features, target = calibration(X, y, day, window)
    if len(features) <= X.shape[1] + DAYS_PER_WEEK:
        raise ValueError(f"{len(features)} dies de calibratge per a {day}, "
                         f"insuficients per a {X.shape[1] + DAYS_PER_WEEK} columnes")

    scaler = Scaler(target.to_numpy())
    train = _design(features, scaler)
    test = _design(X.loc[[day]], scaler)
    scaled = scaler.forward(target.to_numpy())
    hours = [_fit_hour(train, scaled[:, hour]).predict(test)[0] for hour in range(HOURS_PER_DAY)]
    return scaler.back(np.array(hours))


def forecast_ensemble(X, y, day, windows=WINDOWS):
    """Mitjana aritmètica de les prediccions de cada finestra de calibratge."""
    return np.mean([forecast(X, y, day, window) for window in windows], axis=0)
