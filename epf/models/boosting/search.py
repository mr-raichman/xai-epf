"""La cerca de LightGBM: la graella del grup de Trebbien et al. (2023), sencera.

L'article no publica la seva cerca, i el seu codi, que la porta, no és enlloc. La graella
és la del codi de dos dels seus autors per a un altre problema, amb LightGBM, i se'n prova
cada punt una vegada, com fa la seva cerca en graella, amb la prova de la part A: un
ajust amb les files dels quatre anys anteriors al primer dia de prova, aturat després de
`PATIENCE` arbres sense millorar sobre la quarta part final dels seus dies en ordre de
calendari, i puntuat amb l'error absolut mitjà d'aquells dies en EUR/MWh. Les files i la
partició són les de `models/search.py`, les mateixes que les de la xarxa.

L'error amb què s'atura és el quadràtic, el de LightGBM per a la seva pèrdua, i el que
puntua és l'absolut: la pèrdua de l'entrenament i la mesura de l'avaluació de Trebbien et
al. (2023).

Una cerca interrompuda no perd res: l'estudi es desa a cada prova, i tornar-la a cridar
prova els punts que falten. `--trials 0` no en fa cap de nou i desa el millor que
l'estudi ja té.
"""
import datetime

import numpy as np
import optuna

from epf import matrix
from epf.data import forecasts, price
from epf.models.boosting import boosting, boosting_hyperparameters
from epf.models.boosting.boosting_hyperparameters import MOST_TREES, NAME
from epf.models.search import split, window
from epf.parameters import SEED

# La graella del codi de Kruse, Schäfer i Witthaut per a l'activació de la reserva
# secundària, `scripts/6_model_fit.py` de `johkruse/XAI-for-aFRR-activation`: 288 punts.
GRID = {
    "num_leaves": (10, 30, 200, 900),
    "max_depth": (10, 5),
    "subsample": (1.0, 0.9, 0.5),
    "learning_rate": (0.001, 0.05, 0.01, 0.1),
    "min_child_samples": (50, 100, 500),
}
POINTS = int(np.prod([len(values) for values in GRID.values()]))


def space(trial):
    """El punt de la graella d'una prova."""
    return {name: trial.suggest_categorical(name, values) for name, values in GRID.items()}


def parts(X):
    """Les files d'entrenament i les de validació, com a màscares, partides per dies
    sencers: les 24 hores d'un dia van totes a la mateixa banda."""
    days = X.index.unique()
    train, validate = split(len(days))
    return X.index.isin(days[train]), X.index.isin(days[validate])


def score(point, X, y):
    """El MAE en EUR/MWh d'un punt sobre els dies de validació, i quants arbres va fer
    l'ajust abans d'aturar-se."""
    train, validate = parts(X)
    model = boosting.fit_rows(X[train], y[train], point, MOST_TREES,
                              validation=[(X[validate], y[validate])])
    predicted = boosting.predict_rows(model, X[validate])
    return float(np.mean(np.abs(predicted - y[validate].to_numpy()))), model.best_iteration_


def completed(searching):
    """Les proves acabades d'un estudi."""
    return [one for one in searching.trials
            if one.state == optuna.trial.TrialState.COMPLETE]


def study(store):
    """L'estudi de la cerca, desat a `store`, amb un mostrejador que recorre la graella
    sense repetir cap punt, també si es reprèn."""
    sampler = optuna.samplers.GridSampler(
        {name: list(values) for name, values in GRID.items()}, seed=SEED)
    return optuna.create_study(study_name=NAME, direction="minimize", load_if_exists=True,
                               storage=f"sqlite:///{store}", sampler=sampler)


def main(arguments):
    if arguments.member is not None:
        raise SystemExit("LightGBM no té membres: --member només és per a la xarxa")
    X, y = matrix.lago(price.load(), forecasts.load()).hourly
    features, target = window(X, y)
    days = features.index.unique()
    validated = days[split(len(days))[1]]
    print(f"la graella de {POINTS} punts, cada un d'un ajust sobre {len(days)} dies, de "
          f"{days[0].date()} a "
          f"{days[-1].date()}, puntuat de {validated[0].date()} a "
          f"{validated[-1].date()}", flush=True)

    def objective(trial):
        found, trees = score(space(trial), features, target)
        trial.set_user_attr("trees", trees)
        return found

    store = boosting_hyperparameters.path(arguments.inputs).with_suffix(".db")
    store.parent.mkdir(parents=True, exist_ok=True)
    searching = study(store)
    # Els punts que falten, o menys si `--trials` ho diu. Una graella ja sencera no en
    # fa cap: el mostrejador d'Optuna en tornaria a provar un a l'atzar.
    remaining = POINTS - len(completed(searching))
    searching.optimize(objective, n_trials=remaining if arguments.trials is None
                       else min(arguments.trials, remaining))

    done = completed(searching)
    if not done:
        print("cap prova acabada: no es desa res")
        return 1

    best = searching.best_trial
    boosting_hyperparameters.save(arguments.inputs, {
        "taken": str(datetime.date.today()),
        "trials": len(done),
        "first": str(days[0].date()),
        "last": str(days[-1].date()),
        "validated_from": str(validated[0].date()),
        "stage": arguments.inputs,
        "mae": best.value,
        "hyperparameters": space(optuna.trial.FixedTrial(best.params))
        | {"trees": best.user_attrs["trees"]},
    })
    print(f"millor MAE {best.value:.3f} EUR/MWh de {len(done)} punts, desat a "
          f"{boosting_hyperparameters.path(arguments.inputs)}")
    return 0
