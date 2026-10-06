"""Model LEAR de Lago et al. (2021), la referència que els altres models han de superar.

Un LASSO per hora sobre preus transformats, reajustat cada dia sobre una finestra
de calibratge, i la mitjana aritmètica de les prediccions de cada finestra. La
lambda surt de LARS amb l'AIC de la mostra i el model final es reajusta per
descens de coordenades, tal com descriu l'article.

Implementació pròpia a partir de la descripció de l'article, no del seu codi.
"""
import numpy as np
from sklearn.linear_model import Lasso, LassoLarsIC
from threadpoolctl import threadpool_limits

from epf.models.lear.lear_hyperparameters import MAX_ITER, SHORT_LAGS, WINDOWS
from epf.parameters import DAYS_PER_WEEK, LAGS
from epf.market import HOURS_PER_DAY
from epf.data.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK, calibration
from epf.models.attribution import Attribution

# Nom de la finestra que agafa tot l'historial, per a l'atribució desada.
WHOLE_HISTORY = "all"

# La mediana de les desviacions absolutes dividida per aquesta constant estima la
# desviació típica d'una normal; és el que fa `statsmodels.robust.mad`, i el que
# la referència utilitza encara que l'article només digui "MAD".
NORMAL_CONSISTENCY = 0.6744897501960817

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
    """Columnes de `matrix.build` que fa servir una finestra, en el mateix ordre."""
    return [f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS] + [DAY_OF_WEEK]


def design_columns(lags):
    """Noms de la matriu de disseny: els retards escalats i els set indicadors."""
    return ([f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS]
            + [f"{DAY_OF_WEEK}_{number}" for number in range(DAYS_PER_WEEK)])


DESIGN_COLUMNS = design_columns(LAGS)


def _window_forecast(X, y, day, window, max_iter):
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
    # Un fil. La matriu de disseny és de 1085 per 103 i a OpenBLAS li costa més
    # repartir-la entre setze fils que fer-la en un: mesurat sobre tres dies, un fil
    # va 2,05 a 2,18 vegades més ràpid i dóna el mateix resultat bit a bit. El límit
    # és d'aquesta família i no de l'execució, perquè les altres sí que aprofiten els
    # setze.
    with threadpool_limits(limits=1):
        for hour in range(HOURS_PER_DAY):
            model = _fit_hour(train, scaled[:, hour], max_iter)
            hours.append(model.predict(test)[0])
            contributions.append(model.coef_ * (test[0] - background))
            # El valor del qual parteixen les contribucions: la predicció a la
            # mitjana de la finestra. Base més la suma de les contribucions torna
            # la predicció.
            base.append(model.intercept_ + model.coef_ @ background)
    return (prices_target.back(np.array(hours)), np.array(contributions).T,
            np.array(base))


def forecast(X, y, day, window, max_iter):
    """Els 24 preus de `day`, amb el model ajustat només amb dies anteriors.

    `X` i `y` són els de `features.matrix.build`, i la finestra són dies de
    calibratge, com a tot el projecte.
    """
    return _window_forecast(X, y, day, window, max_iter)[0]


def explain(X, y, day, windows=WINDOWS, max_iter=MAX_ITER):
    """Predicció del conjunt i contribucions de cada finestra, desades per separat.

    Cada finestra escala la seva pròpia finestra amb la seva mediana i la seva MAD,
    de manera que dues finestres no tenen les contribucions en les mateixes unitats i
    mitjanar-les no voldria dir res. Les columnes que una finestra curta no porta hi
    són a zero.
    """
    forecasts, values, bases = [], [], []
    for window in windows:
        prices, contributions, base = _window_forecast(X, y, day, window, max_iter)
        forecasts.append(prices)
        bases.append(base)
        whole = np.zeros((len(DESIGN_COLUMNS), HOURS_PER_DAY))
        whole[[DESIGN_COLUMNS.index(name) for name in design_columns(lags_of(window))]] = contributions
        values.append(whole)
    members = tuple(str(window) if window else WHOLE_HISTORY for window in windows)
    return (np.mean(forecasts, axis=0),
            Attribution(members, DESIGN_COLUMNS, np.array(values), np.array(bases),
                        np.array(forecasts), "espai transformat"))
