"""La cerca d'hiperparàmetres de la xarxa, com la fa el codi de Lago et al. (2021).

**Una prova és un ajust.** La xarxa s'ajusta amb els quatre anys anteriors al primer
dia de prova, s'atura amb la seva quarta part final en ordre de calendari, i es puntua
amb l'error absolut mitjà d'aquella quarta part en EUR/MWh, desfent l'escalat de la
sortida; les mateixes files que l'aturen la puntuen, com al codi. És el codi de
l'article amb la barreja apagada, que el seu text també descriu: barrejant les setmanes
dels quatre anys, més de la meitat de les de validació serien de la crisi o del topall
del gas, que no s'assemblen a cap dels dos anys de prova. El codi puntua també cada
prova sobre el període de prova i ho imprimeix; aquí no se'l toca mai.

Les files d'una prova i la seva partició són les de `models/search.py`, que comparteix
amb la cerca de l'arbre, i l'estimador i el pressupost també, que comparteix amb la de
les xarxes distribucionals. L'espai és el del codi, `_build_space`, amb els seus intervals
i els seus noms d'opció.

Cada membre de la xarxa és una cerca pròpia, `epf search --family dnn --member`, amb el
seu estudi, el seu
registre i la seva llavor inicial, que és com l'article obté les seves quatre xarxes. Una
cerca interrompuda no perd res: l'estudi es desa a cada prova, i tornar-la a cridar la
reprèn. `--trials 0` no en fa cap de nova i desa el millor que l'estudi ja té, que és
com es tanca una cerca que va caure abans d'arribar-hi.
"""
import datetime

import numpy as np
import optuna

from epf import matrix
from epf.data import forecasts, price
from epf.models.blocks import blocks, inputs
from epf.models.dnn import dnn, dnn_hyperparameters
from epf.models.dnn.dnn_hyperparameters import NAME
from epf.models.search import TRIALS, completed, guided, split, switched_on, window

# L'espai del codi de Lago et al. (2021): les neurones de cada capa amagada, la segona
# fora si en té menys de `dnn_hyperparameters.SMALLEST_LAYER`; les set activacions i les
# sis inicialitzacions, pels seus noms; la L1, apagada o en aquest interval logarítmic;
# el dropout; la taxa d'aprenentatge, també logarítmica; i la llavor.
NEURONS = ((50, 500), (25, 400))
ACTIVATIONS = ("relu", "softplus", "tanh", "selu", "LeakyReLU", "PReLU", "sigmoid")
INITIALISERS = ("Orthogonal", "lecun_uniform", "glorot_uniform", "glorot_normal",
                "he_uniform", "he_normal")
L1 = (1e-5, 1.0)
DROPOUT = (0.0, 1.0)
LEARNING_RATE = (5e-4, 0.1)
SEEDS = (1, 1000)


def space(trial, names):
    """Els hiperparàmetres d'una prova, demanats un per un a Optuna, sobre els blocs
    `names`.

    La L1 es demana com al codi: primer si n'hi ha, i el coeficient només si n'hi ha.
    """
    return {
        "blocks": switched_on(trial, names),
        "hidden": [trial.suggest_int(f"neurons_{layer}", *bounds)
                   for layer, bounds in enumerate(NEURONS)],
        "activation": trial.suggest_categorical("activation", ACTIVATIONS),
        "l1": (trial.suggest_float("l1", *L1, log=True)
               if trial.suggest_categorical("l1_on", (False, True)) else 0.0),
        "dropout": trial.suggest_float("dropout", *DROPOUT),
        "batch_normalisation": trial.suggest_categorical("batch_normalisation",
                                                         (False, True)),
        "initialiser": trial.suggest_categorical("initialiser", INITIALISERS),
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
        "scale_inputs": trial.suggest_categorical("scale_inputs", tuple(dnn.SCALINGS)),
        "scale_outputs": trial.suggest_categorical("scale_outputs", tuple(dnn.SCALINGS)),
        "seed": trial.suggest_int("seed", *SEEDS),
    }


def score(tuned, features, target):
    """El MAE en EUR/MWh d'un conjunt: un ajust amb `features`, puntuat a la seva quarta
    part final, la mateixa que l'atura."""
    values = inputs(features, tuned["blocks"]).to_numpy()
    found = target.to_numpy(dtype=float)
    train, validate = split(len(values))
    fitted = dnn.fit_rows((values[train], found[train]),
                          (values[validate], found[validate]), tuned, tuned["seed"])
    return float(np.mean(np.abs(dnn.prices(fitted, values[validate]) - found[validate])))


def main(arguments):
    # `--trials 0` no és «cap valor»: és demanar que es desi el que l'estudi ja sap,
    # que és com es recupera una cerca que va caure abans d'arribar a desar res.
    member = arguments.member
    if member is None:
        raise SystemExit("la xarxa es cerca membre a membre: falta --member")
    trials = TRIALS if arguments.trials is None else arguments.trials
    X, y = matrix.lago(price.load(), forecasts.load()).daily
    features, target = window(X, y)
    names = list(blocks(features.columns))
    validated = features.index[split(len(features))[1]]
    print(f"membre {member}: {trials} proves, cada una d'un ajust sobre {len(features)} "
          f"dies, de {features.index[0].date()} a {features.index[-1].date()}, puntuat "
          f"de {validated[0].date()} a {validated[-1].date()}", flush=True)

    def objective(trial):
        return score(space(trial, names), features, target)

    # L'estudi es desa al costat del resultat, de manera que una cerca interrompuda es
    # reprèn en lloc de tornar a començar: una de sencera són hores.
    store = dnn_hyperparameters.path(arguments.inputs, member).with_suffix(".db")
    store.parent.mkdir(parents=True, exist_ok=True)
    searching = guided(store, f"{NAME}_{member}")
    searching.optimize(objective, n_trials=trials)

    done = completed(searching)
    if not done:
        print("cap prova acabada: no es desa res")
        return 1

    dnn_hyperparameters.save(arguments.inputs, member, {
        "taken": str(datetime.date.today()),
        "member": member,
        "trials": len(done),
        "first": str(features.index[0].date()),
        "last": str(features.index[-1].date()),
        "validated_from": str(validated[0].date()),
        "stage": arguments.inputs,
        "mae": searching.best_value,
        "hyperparameters": space(optuna.trial.FixedTrial(searching.best_params), names),
    })
    print(f"millor MAE {searching.best_value:.3f} EUR/MWh de {len(done)} proves, desat a "
          f"{dnn_hyperparameters.path(arguments.inputs, member)}")
    return 0
