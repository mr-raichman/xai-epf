"""La cerca de la xarxa elàstica: la graella d'Uniejewski et al. (2016), sencera.

Les tres barreges que l'article fixa a mà, i per a cada una els 34 càstigs de la
seqüència que fa `lasso.m`, l'eina amb què l'article les ajusta: geomètrica, del càstig
més petit que deixa a zero tots els coeficients fins a `RATIO` d'ell, el valor per
defecte de l'eina, que l'article no canvia. Un sol càstig serveix les 24 hores, com a
l'article, de manera que el més gran de cada seqüència és el més gran de les 24.

La seqüència es calcula sobre les files d'entrenament de la prova, i se'n prova cada
punt una vegada amb la prova de la part A: un ajust amb els quatre anys anteriors al
primer dia de prova, puntuat amb l'error absolut mitjà en EUR/MWh de la quarta part
final dels seus dies. Les files i la partició són les de `models/search.py`.

Una cerca interrompuda no perd res: l'estudi es desa a cada prova, i tornar-la a cridar
prova els punts que falten. `--trials 0` no en fa cap de nou i desa el millor que
l'estudi ja té.
"""
import datetime

import numpy as np
import optuna

from epf import matrix
from epf.data import forecasts, price
from epf.models.linear import linear, linear_hyperparameters
from epf.models.linear.linear_hyperparameters import NAME
from epf.models.search import completed, parts, through_grid, window
from epf.parameters import SEED

# Les barreges d'Uniejewski et al. (2016), els càstigs de cada una, i la raó entre el més
# petit i el més gran, el `LambdaRatio` per defecte de `lasso.m`.
MIXINGS = (0.25, 0.5, 0.75)
STEPS = 34
RATIO = 1e-4
POINTS = len(MIXINGS) * STEPS


def penalties(X, y):
    """Per a cada barreja, els seus `STEPS` càstigs, del que deixa a zero tots els
    coeficients de les 24 hores cap avall, sobre les files donades."""
    found = {}
    for mixing in MIXINGS:
        largest = linear.largest_penalty(X, y, mixing)
        found[mixing] = np.geomspace(largest, largest * RATIO, STEPS)
    return found


def space(trial):
    """El punt de la graella d'una prova: la barreja i el pas de la seva seqüència."""
    return (trial.suggest_categorical("mixing", MIXINGS),
            trial.suggest_categorical("step", tuple(range(STEPS))))


def score(penalty, mixing, X, y):
    """El MAE en EUR/MWh d'un càstig i una barreja sobre els dies de validació."""
    train, validate = parts(X)
    model = linear.fit_rows(X[train], y[train], penalty, mixing)
    predicted = linear.predict_rows(model, X[validate])
    return float(np.mean(np.abs(predicted.to_numpy() - y[validate].to_numpy())))


def study(store):
    """L'estudi de la cerca, desat a `store`, amb un mostrejador que recorre la graella
    sense repetir cap punt, també si es reprèn."""
    sampler = optuna.samplers.GridSampler(
        {"mixing": list(MIXINGS), "step": list(range(STEPS))}, seed=SEED)
    return optuna.create_study(study_name=NAME, direction="minimize", load_if_exists=True,
                               storage=f"sqlite:///{store}", sampler=sampler)


def main(arguments):
    if arguments.member is not None:
        raise SystemExit("la xarxa elàstica no té membres: --member només és per a la "
                         "xarxa de l'article")
    X, y = matrix.lago(price.load(), forecasts.load()).hourly
    features, target = window(X, y)
    train, validate = parts(features)
    grid = penalties(features[train], target[train])
    days, validated = features.index.unique(), features.index[validate].unique()
    print(f"la graella de {POINTS} punts, cada un d'un ajust sobre {len(days)} dies, de "
          f"{days[0].date()} a {days[-1].date()}, puntuat de {validated[0].date()} a "
          f"{validated[-1].date()}", flush=True)

    def objective(trial):
        mixing, step = space(trial)
        trial.set_user_attr("penalty", float(grid[mixing][step]))
        return score(grid[mixing][step], mixing, features, target)

    store = linear_hyperparameters.path(arguments.inputs).with_suffix(".db")
    store.parent.mkdir(parents=True, exist_ok=True)
    searching = study(store)
    through_grid(searching, objective, POINTS, arguments.trials)

    done = completed(searching)
    if not done:
        print("cap prova acabada: no es desa res")
        return 1

    best = searching.best_trial
    linear_hyperparameters.save(arguments.inputs, {
        "taken": str(datetime.date.today()),
        "trials": len(done),
        "first": str(days[0].date()),
        "last": str(days[-1].date()),
        "validated_from": str(validated[0].date()),
        "stage": arguments.inputs,
        "mae": best.value,
        "step": best.params["step"],
        "hyperparameters": {"penalty": best.user_attrs["penalty"],
                            "mixing": best.params["mixing"]},
    })
    print(f"millor MAE {best.value:.3f} EUR/MWh de {len(done)} punts, desat a "
          f"{linear_hyperparameters.path(arguments.inputs)}")
    return 0
