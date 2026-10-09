"""La cerca d'hiperparàmetres de la xarxa: què es prova, com es puntua i què es desa.

Lago et al. (2021) en trien dinou amb un estimador de Parzen i no diu els seus
intervals. Els intervals, el conjunt d'activacions i l'esquema de cada prova són de
Marcjasz et al. (2023), que cerca els de la mateixa xarxa i els escriu, i que la cerca
amb Optuna.

**Una prova no és un ajust.** Marcjasz et al. (2023) diuen per què: amb una sola
estimació "the results of such an experiment are very volatile" i el conjunt que en
surt no garanteix bon rendiment, mentre que reajustar cada dia "would take roughly 364
times longer". El seu compromís, que és el d'aquí, són tretze reajustos sobre lots de
28 dies, amb la finestra rodant 28 dies a cada pas.

Això és l'esquema de la cerca i no el del pronòstic: el camí endavant sí que reajusta
cada dia. Dins d'un lot, l'últim dia el prediu un ajust de fa quatre setmanes, cosa que
fa la puntuació una mica pessimista i la cerca assumible. Cap dia d'un lot no entra a
l'ajust que el prediu, que és el que `tests/test_search.py` fixa.

Una cerca interrompuda no perd res: l'estudi es desa a cada prova, i tornar-la a
cridar la reprèn. `--trials 0` no en fa cap de nova i desa el millor que l'estudi ja
té, que és com es tanca una cerca que va caure abans d'arribar-hi.

Es cerca sobre l'últim any del període de validació, que acaba el dia abans del primer
de prova. Els anys de prova no es toquen mai; aquell any és el més semblant a ells que
es pot fer servir, perquè el topall del gas va caure el 2023-12-31 i nou dels seus
dotze mesos ja en són a fora. El preu és que l'escombrat compara famílies sobre tots
dos anys de validació i la xarxa ha triat els seus números en un.
"""
import datetime

import numpy as np
import optuna

from epf import matrix, parameters
from epf.data import price
from epf.models.dnn import dnn, dnn_hyperparameters
from epf.models.dnn.dnn_hyperparameters import (BATCH_DAYS, RECALIBRATIONS, TRIALS,
                                                NAME)
from epf.parameters import SEED, VALIDATION

# Capes amagades, que l'article fixa a dues i no cerca. Cada una té les seves
# neurones, la seva activació i la seva L1.
LAYERS = 2

# Les activacions que Marcjasz et al. (2023) proven. No són les set de la
# implementació de Lago et al. (2021): les seves inclouen `LeakyReLU` i `PReLU`, que
# són una capa a part darrere d'una `Dense` lineal, i `selu`, que vol `AlphaDropout`.
# Cap d'aquestes sis no demana res especial a `network.build`.
ACTIVATIONS = ("elu", "relu", "sigmoid", "softmax", "softplus", "tanh")

# Les inicialitzacions que prova la implementació de Lago et al. (2021). Marcjasz et
# al. (2023) no diuen que la cerquin.
INITIALISERS = ("Orthogonal", "lecun_uniform", "glorot_uniform", "glorot_normal",
                "he_uniform", "he_normal")

# Intervals de Marcjasz et al. (2023) secció 4.2.1.
NEURONS = (16, 1024)
DROPOUT = (0.0, 1.0)
L1 = (1e-5, 10.0)
LEARNING_RATE = (1e-5, 1e-1)


def space(trial):
    """Els hiperparàmetres d'una prova, demanats un per un a Optuna.

    Els condicionals es demanen com l'article els compta: primer si la capa porta
    dropout o L1, i la taxa només si en porta. Una prova que no deixa cap bloc
    d'entrada engegat no és una xarxa i es poda.
    """
    chosen = [name for name in dnn.blocks()
              if trial.suggest_categorical(f"use_{name}", (False, True))]
    if not chosen:
        raise optuna.TrialPruned("cap bloc d'entrada engegat")

    dropout = (trial.suggest_float("dropout", *DROPOUT)
               if trial.suggest_categorical("dropout_on", (False, True)) else 0.0)
    return {
        "blocks": chosen,
        "hidden": [trial.suggest_int(f"neurons_{layer}", *NEURONS)
                   for layer in range(LAYERS)],
        "activation": [trial.suggest_categorical(f"activation_{layer}", ACTIVATIONS)
                       for layer in range(LAYERS)],
        "l1": [trial.suggest_float(f"l1_{layer}", *L1, log=True)
               if trial.suggest_categorical(f"l1_on_{layer}", (False, True)) else 0.0
               for layer in range(LAYERS)],
        "dropout": dropout,
        "batch_normalisation": trial.suggest_categorical("batch_normalisation",
                                                         (False, True)),
        "initialiser": trial.suggest_categorical("initialiser", INITIALISERS),
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE,
                                             log=True),
    }


def lots(days, count=RECALIBRATIONS, size=BATCH_DAYS):
    """Els `count` lots de `size` dies consecutius amb què acaba el període.

    Acaben al final i no comencen al principi: el que s'ha de puntuar són els dies més
    propers als de prova.
    """
    if len(days) < count * size:
        raise ValueError(f"{len(days)} dies no donen {count} lots de {size}")
    kept = days[-count * size:]
    return [kept[start:start + size] for start in range(0, len(kept), size)]


def score(tuned, X, y, actual, batches):
    """El MAE d'un conjunt: un ajust per lot i tots els dies del lot amb aquell ajust.

    `dnn.fit` calibra amb dies anteriors al primer del lot, de manera que cap dia que
    es puntua no ha entrat a l'ajust que el prediu. Un membre per prova: a l'article
    cada membre de la mitjana surt de la seva pròpia cerca.
    """
    errors = []
    for batch in batches:
        model = dnn.fit(X, y, batch[0], tuned=tuned, members=1)
        for day in batch:
            predicted = dnn.predict(model, X, day)[NAME]
            errors.append(np.abs(predicted - actual.loc[day].to_numpy()))
    return float(np.mean(errors))


def main(arguments):
    # `--trials 0` no és «cap valor»: és demanar que es desi el que l'estudi ja sap,
    # que és com es recupera una cerca que va caure abans d'arribar a desar res.
    trials = TRIALS if arguments.trials is None else arguments.trials
    data = matrix.price_only(price.load())
    X, y = data.daily
    period = (X.index >= str(VALIDATION[0])) & (X.index <= str(VALIDATION[1]))
    batches = lots(X.index[period].unique())
    print(f"{trials} proves, cada una de {RECALIBRATIONS} ajustos sobre "
          f"lots de {BATCH_DAYS} dies, de {batches[0][0].date()} a "
          f"{batches[-1][-1].date()}", flush=True)

    def objective(trial):
        return score(space(trial), X, y, data.actual, batches)

    # L'estudi es desa al costat del resultat, de manera que una cerca interrompuda es
    # reprèn en lloc de tornar a començar: una de sencera són hores.
    store = dnn_hyperparameters.path().with_suffix(".db")
    store.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=NAME, direction="minimize", load_if_exists=True,
        storage=f"sqlite:///{store}",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)

    done = [one for one in study.trials
            if one.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("cap prova acabada: no es desa res")
        return 1

    dnn_hyperparameters.save({
        "taken": str(datetime.date.today()),
        "trials": len(done),
        "recalibrations": RECALIBRATIONS,
        "batch_days": BATCH_DAYS,
        "first": str(batches[0][0].date()),
        "last": str(batches[-1][-1].date()),
        "stage": parameters.PRICE_ONLY,
        "mae": study.best_value,
        "hyperparameters": space(optuna.trial.FixedTrial(study.best_params)),
    })
    print(f"millor MAE {study.best_value:.3f} EUR/MWh de {len(done)} proves, desat a "
          f"{dnn_hyperparameters.path()}")
    return 0
