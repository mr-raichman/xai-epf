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
from epf.matrix import DAY_OF_WEEK, calibration
from epf.attribution.stored import Attribution

# Nom de la finestra que agafa tot l'historial, per a l'atribució desada.
WHOLE_HISTORY = "all"

# El que queda ajustat d'una finestra, i tot el que la predicció i les contribucions
# necessiten. Els 24 LASSO hi són tal com es van ajustar i no com a matrius de
# coeficients, de manera que predir és exactament la crida que els va predir. Hi va la
# mida de la finestra i no els seus retards ni les seves columnes, que se'n deriven:
# els retards amb `lags_of` i les columnes dues vegades, per llegir la fila i per
# col·locar les contribucions. També és el nom del membre a l'atribució. `background` és
# la mitjana de cada columna de la matriu de disseny, `prices` l'escalador de les
# entrades i `target` el de les 24 sortides, que és qui torna a EUR/MWh.
Window = collections.namedtuple("Window", "window models background prices target")

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


def _fit_window(X, y, day, window, max_iter):
    """Els 24 LASSO d'una finestra, ajustats només amb dies anteriors, i el que cal
    per llegir-los: la mitjana de la finestra i els dos escaladors."""
    lags = lags_of(window)
    columns = feature_columns(lags)
    features, target = calibration(X[columns], y, day, window)
    width = len(columns) - 1 + DAYS_PER_WEEK
    if len(features) <= width + 1:
        raise ValueError(f"{len(features)} dies de calibratge per a {day}, "
                         f"insuficients per a {width} columnes")

    prices = Scaler(features.to_numpy()[:, :-1])
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
    return Window(window, models, train.mean(axis=0), prices, prices_target)


def _rows_of(fitted, X, day):
    """La fila del dia que es prediu, escalada amb l'escalador de la finestra."""
    columns = feature_columns(lags_of(fitted.window))
    return _design(X.loc[[day], columns], fitted.prices)


def _window_prices(fitted, rows):
    """Els 24 preus d'una finestra, en EUR/MWh."""
    with threadpool_limits(limits=1):
        hours = [model.predict(rows)[0] for model in fitted.models]
    return fitted.target.back(np.array(hours))


def _window_contributions(fitted, rows):
    """(contribucions, base) d'una finestra, en el seu espai transformat.

    Les contribucions són els valors de SHAP exactes d'un model lineal, el
    coeficient per la distància de la columna a la mitjana de la finestra, i no van
    en EUR/MWh. La base és la predicció a aquella mitjana, en el mateix espai, i
    cada hora té la seva perquè l'objectiu s'escala per columna: base més la suma de
    les contribucions torna la predicció del membre.
    """
    with threadpool_limits(limits=1):
        contributions = [model.coef_ * (rows[0] - fitted.background)
                         for model in fitted.models]
        base = [model.intercept_ + model.coef_ @ fitted.background
                for model in fitted.models]
    return np.array(contributions).T, np.array(base)


def fit(X, y, day, windows=WINDOWS, max_iter=MAX_ITER):
    """El conjunt ajustat: una finestra per cada mida de `windows`.

    `X` i `y` són els de `matrix.build`, i una finestra són dies de calibratge, com a
    tot el projecte.
    """
    return tuple(_fit_window(X, y, day, window, max_iter) for window in windows)


def predict(model, X, day):
    """La predicció del conjunt: la mitjana aritmètica de les seves finestres."""
    return np.mean([_window_prices(fitted, _rows_of(fitted, X, day))
                    for fitted in model], axis=0)


def attribute(model, X, day):
    """Contribucions de cada finestra, desades per separat.

    Cada finestra escala la seva pròpia finestra amb la seva mediana i la seva MAD,
    de manera que dues finestres no tenen les contribucions en les mateixes unitats i
    mitjanar-les no voldria dir res. Les columnes que una finestra curta no porta hi
    són a zero. `forecasts` porta la predicció de cada finestra perquè la del CSV és
    la mitjana i la base de cada membre s'ha de poder comprovar contra la seva.
    """
    forecasts, values, bases = [], [], []
    for fitted in model:
        rows = _rows_of(fitted, X, day)
        contributions, base = _window_contributions(fitted, rows)
        forecasts.append(_window_prices(fitted, rows))
        bases.append(base)
        whole = np.zeros((len(DESIGN_COLUMNS), HOURS_PER_DAY))
        carried = design_columns(lags_of(fitted.window))
        whole[[DESIGN_COLUMNS.index(name) for name in carried]] = contributions
        values.append(whole)
    # Els noms surten de les finestres que s'han ajustat i no de `WINDOWS`: una crida
    # amb un subconjunt n'ha de donar els seus i no cinc.
    members = tuple(str(fitted.window) if fitted.window else WHOLE_HISTORY
                    for fitted in model)
    return Attribution(members, DESIGN_COLUMNS, np.array(values), np.array(bases),
                       np.array(forecasts), "espai transformat")
