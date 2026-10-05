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
from epf.data.calendar import HOUR_COLUMNS
from epf.features.matrix import DAY_OF_WEEK, PRICE_LAGS, calibration
from epf.models.attribution import Attribution

DAYS_PER_WEEK = 7

# Dies que es perden al principi de qualsevol tram: cap fila es pot formar fins que
# hi ha tots els retards.
WARM_UP = max(PRICE_LAGS)

# Finestres de calibratge, en dies de calibratge. Els "anys" de l'article són de 364
# dies, 52 setmanes exactes que mantenen alineat el dia de la setmana; les seves
# finestres de 1092 i 1456 dies de preus donen aquests dies un cop descomptat el
# començament. None vol dir tot l'historial disponible, que l'article no fa servir.
EIGHT_WEEKS = 8 * DAYS_PER_WEEK - WARM_UP
TWELVE_WEEKS = 12 * DAYS_PER_WEEK - WARM_UP
WINDOWS = (EIGHT_WEEKS, TWELVE_WEEKS, 364 * 3 - WARM_UP, 364 * 4 - WARM_UP, None)

# Retards que porta cada finestra curta. Amb els quatre de l'article la matriu de
# disseny té 103 columnes i LassoLarsIC demana dues files més que columnes, que cap
# de les dues finestres curtes no té: cada una porta els retards que hi caben.
SHORT_LAGS = {EIGHT_WEEKS: (1,), TWELVE_WEEKS: (1, 2)}

# Nom de la finestra que agafa tot l'historial, per a l'atribució desada.
WHOLE_HISTORY = "all"

# La mediana de les desviacions absolutes dividida per aquesta constant estima la
# desviació típica d'una normal; és el que fa `statsmodels.robust.mad`, i el que
# la referència utilitza encara que l'article només digui "MAD".
NORMAL_CONSISTENCY = 0.6744897501960817

# Prou iteracions perquè el descens de coordenades convergeixi també sobre les
# finestres llargues que travessen la crisi del 2021-22, on amb 2500 la meitat dels
# ajusts es tallaven a mitges i la predicció es movia fins a 0,566 EUR/MWh. Un ajust
# que ja convergia no es mou gens si s'apuja el sostre.
MAX_ITER = 10000


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


def lags_of(window):
    """Retards que porta una finestra: els quatre de l'article, o menys si és curta."""
    return SHORT_LAGS.get(window, PRICE_LAGS)


def feature_columns(lags):
    """Columnes de `matrix.build` que fa servir una finestra, en el mateix ordre."""
    return [f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS] + [DAY_OF_WEEK]


def design_columns(lags):
    """Noms de la matriu de disseny: els retards escalats i els set indicadors."""
    return ([f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS]
            + [f"{DAY_OF_WEEK}_{number}" for number in range(DAYS_PER_WEEK)])


DESIGN_COLUMNS = design_columns(PRICE_LAGS)


def _window_forecast(X, y, day, window):
    """(24 preus, contribucions, base) d'una finestra, ajustada només amb dies
    anteriors.

    Les contribucions són els valors de SHAP exactes d'un model lineal, el
    coeficient per la distància de la columna a la mitjana de la finestra, i van en
    l'espai transformat del model, que no és EUR/MWh. La base és la predicció a
    aquella mitjana, en el mateix espai, i cada hora té la seva perquè l'objectiu
    s'escala per columna.
    """
    columns = feature_columns(lags_of(window))
    features, target = calibration(X[columns], y, day, window)
    width = len(columns) - 1 + DAYS_PER_WEEK
    if len(features) <= width + 1:
        raise ValueError(f"{len(features)} dies de calibratge per a {day}, "
                         f"insuficients per a {width} columnes")

    prices = Scaler(features.to_numpy()[:, :-1])
    prices_target = Scaler(target.to_numpy())
    train = _design(features, prices)
    test = _design(X.loc[[day], columns], prices)
    scaled = prices_target.forward(target.to_numpy())
    background = train.mean(axis=0)

    hours, contributions, base = [], [], []
    for hour in range(HOURS_PER_DAY):
        model = _fit_hour(train, scaled[:, hour])
        hours.append(model.predict(test)[0])
        contributions.append(model.coef_ * (test[0] - background))
        # El valor del qual parteixen les contribucions: la predicció a la mitjana de
        # la finestra. Base més la suma de les contribucions torna la predicció.
        base.append(model.intercept_ + model.coef_ @ background)
    return (prices_target.back(np.array(hours)), np.array(contributions).T,
            np.array(base))


def forecast(X, y, day, window=None):
    """Els 24 preus de `day`, amb el model ajustat només amb dies anteriors.

    `X` i `y` són els de `features.matrix.build`, i la finestra són dies de
    calibratge, com a tot el projecte.
    """
    return _window_forecast(X, y, day, window)[0]


def forecast_ensemble(X, y, day, windows=WINDOWS):
    """Mitjana aritmètica de les prediccions de cada finestra de calibratge."""
    return np.mean([forecast(X, y, day, window) for window in windows], axis=0)


def explain(X, y, day, windows=WINDOWS):
    """Predicció del conjunt i contribucions de cada finestra, desades per separat.

    Cada finestra escala la seva pròpia finestra amb la seva mediana i la seva MAD,
    de manera que dues finestres no tenen les contribucions en les mateixes unitats i
    mitjanar-les no voldria dir res. Les columnes que una finestra curta no porta hi
    són a zero.
    """
    forecasts, values, bases = [], [], []
    for window in windows:
        prices, contributions, base = _window_forecast(X, y, day, window)
        forecasts.append(prices)
        bases.append(base)
        whole = np.zeros((len(DESIGN_COLUMNS), HOURS_PER_DAY))
        whole[[DESIGN_COLUMNS.index(name) for name in design_columns(lags_of(window))]] = contributions
        values.append(whole)
    members = tuple(str(window) if window else WHOLE_HISTORY for window in windows)
    return (np.mean(forecasts, axis=0),
            Attribution(members, DESIGN_COLUMNS, np.array(values), np.array(bases),
                        np.array(forecasts), "espai transformat"))
