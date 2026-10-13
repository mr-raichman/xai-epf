"""La cerca d'hiperparàmetres de la xarxa, com la fa el codi de Lago et al. (2021).

**Una prova és un ajust.** La xarxa s'ajusta amb els quatre anys anteriors al primer
dia de prova, s'atura amb la seva quarta part final en ordre de calendari, i es puntua
amb l'error absolut mitjà d'aquella quarta part en EUR/MWh, desfent l'escalat de la
sortida; les mateixes files que l'aturen la puntuen, com al codi. És el codi de
l'article amb la barreja apagada, que el seu text també descriu: barrejant les setmanes
dels quatre anys, més de la meitat de les de validació serien de la crisi o del topall
del gas, que no s'assemblen a cap dels dos anys de prova. El codi puntua també cada
prova sobre el període de prova i ho imprimeix; aquí no se'l toca mai.

L'espai és el del codi, `_build_space`, amb els seus intervals i els seus noms d'opció.
L'estimador és el de Parzen en arbre que l'article fa servir a través de hyperopt, aquí
el d'Optuna amb els paràmetres de hyperopt allà on els dos no coincideixen.

Una cerca interrompuda no perd res: l'estudi es desa a cada prova, i tornar-la a
cridar la reprèn. `--trials 0` no en fa cap de nova i desa el millor que l'estudi ja
té, que és com es tanca una cerca que va caure abans d'arribar-hi.
"""
import datetime
import math

import numpy as np
import optuna
import pandas as pd

from epf import matrix
from epf.data import forecasts, price
from epf.matrix import calibration
from epf.models.dnn import dnn, dnn_hyperparameters
from epf.models.dnn.dnn_hyperparameters import NAME, TRIALS, WINDOW
from epf.parameters import SEED, TEST_PERIODS

# El primer dia de prova. La cerca s'ajusta amb els quatre anys d'abans i no en toca cap.
FIRST_TEST_DAY = pd.Timestamp(min(first for first, _ in TEST_PERIODS.values()))

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

# Els paràmetres del `tpe.suggest` de hyperopt que no són els d'Optuna: les proves a
# l'atzar abans que comenci a guiar-se per les anteriors, 20 on Optuna en fa 10, i
# quantes de les fetes compten com a bones, la part `GOOD` de l'arrel del seu nombre i
# mai més de `GOOD_CAP`, on Optuna en compta la desena part. La resta del seu estimador
# ja és la d'Optuna. L'argument amb què Optuna pren aquesta regla, `gamma`, desapareix a
# la seva versió 6: el projecte la fixa a la 5.0.0, i moure-la obliga a tornar a triar.
STARTUP = 20
GOOD = 0.25
GOOD_CAP = 25


def good(trials):
    """Quantes de les `trials` proves fetes compten com a bones, amb la regla de hyperopt."""
    return min(math.ceil(GOOD * math.sqrt(trials)), GOOD_CAP)


def space(trial, names):
    """Els hiperparàmetres d'una prova, demanats un per un a Optuna, sobre els blocs
    `names`.

    La L1 es demana com al codi: primer si n'hi ha, i el coeficient només si n'hi ha. Una
    prova que no deixa cap bloc engegat es poda, perquè donaria a la xarxa una entrada
    buida.
    """
    chosen = [name for name in names
              if trial.suggest_categorical(f"use_{name}", (False, True))]
    if not chosen:
        raise optuna.TrialPruned("cap bloc d'entrada engegat")
    return {
        "blocks": chosen,
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


def study(store):
    """L'estudi de la cerca, desat a `store`, amb un estimador que no repeteix el que
    l'estudi ja ha provat.

    Cada crida en fa un de nou, i amb la mateixa llavor tornaria a proposar les mateixes
    proves a l'atzar del començament: una cerca represa repetiria les primeres i no en
    faria de noves. La llavor compta, per això, les proves que l'estudi ja porta.
    """
    found = optuna.create_study(study_name=NAME, direction="minimize",
                                load_if_exists=True, storage=f"sqlite:///{store}")
    found.sampler = optuna.samplers.TPESampler(n_startup_trials=STARTUP, gamma=good,
                                               seed=SEED + len(found.trials))
    return found


def window(X, y):
    """Les files de totes les proves: els `WINDOW` dies anteriors al primer de prova."""
    return calibration(X, y, FIRST_TEST_DAY, WINDOW)


def score(tuned, features, target):
    """El MAE en EUR/MWh d'un conjunt: un ajust amb `features`, puntuat a la seva quarta
    part final, la mateixa que l'atura."""
    values, prices = dnn.inputs(features, tuned["blocks"]), target.to_numpy(dtype=float)
    train, validate = dnn.split(len(values))
    fitted = dnn.fit_rows((values[train], prices[train]),
                          (values[validate], prices[validate]), tuned, tuned["seed"])
    predicted = fitted.outputs.back(fitted.network.predict(
        fitted.inputs.forward(values[validate]), verbose=0))
    return float(np.mean(np.abs(predicted - prices[validate])))


def main(arguments):
    # `--trials 0` no és «cap valor»: és demanar que es desi el que l'estudi ja sap,
    # que és com es recupera una cerca que va caure abans d'arribar a desar res.
    trials = TRIALS if arguments.trials is None else arguments.trials
    X, y = matrix.lago(price.load(), forecasts.load()).daily
    features, target = window(X, y)
    names = list(dnn.blocks(features.columns))
    validated = features.index[dnn.split(len(features))[1]]
    print(f"{trials} proves, cada una d'un ajust sobre {len(features)} dies, de "
          f"{features.index[0].date()} a {features.index[-1].date()}, puntuat de "
          f"{validated[0].date()} a {validated[-1].date()}", flush=True)

    def objective(trial):
        return score(space(trial, names), features, target)

    # L'estudi es desa al costat del resultat, de manera que una cerca interrompuda es
    # reprèn en lloc de tornar a començar: una de sencera són hores.
    store = dnn_hyperparameters.path(arguments.inputs).with_suffix(".db")
    store.parent.mkdir(parents=True, exist_ok=True)
    searching = study(store)
    searching.optimize(objective, n_trials=trials)

    done = [one for one in searching.trials
            if one.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("cap prova acabada: no es desa res")
        return 1

    dnn_hyperparameters.save(arguments.inputs, {
        "taken": str(datetime.date.today()),
        "trials": len(done),
        "first": str(features.index[0].date()),
        "last": str(features.index[-1].date()),
        "validated_from": str(validated[0].date()),
        "stage": arguments.inputs,
        "mae": searching.best_value,
        "hyperparameters": space(optuna.trial.FixedTrial(searching.best_params), names),
    })
    print(f"millor MAE {searching.best_value:.3f} EUR/MWh de {len(done)} proves, desat a "
          f"{dnn_hyperparameters.path(arguments.inputs)}")
    return 0
