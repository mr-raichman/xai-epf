"""La cerca d'hiperparàmetres de les xarxes distribucionals: l'espai del codi de Marcjasz et
al. (2023) amb la prova de la part A.

**Una prova és un ajust**, el de totes les cerques de la part A: la xarxa s'ajusta amb
els quatre anys anteriors al primer dia de prova, s'atura amb la seva quarta part final en
ordre de calendari, i es puntua sobre aquella mateixa quarta part amb la pèrdua de Keras,
la log-versemblança negativa més les L1 de la xarxa. És el que torna el codi de l'article,
`model.evaluate`, i el que mira l'aturada, de manera que la prova s'atura i es puntua amb
el mateix número. Una prova paga, doncs, la seva pròpia penalització.

L'espai és el del codi de l'article, `nn.py`, amb els seus intervals, i amb un selector per
cada bloc del conjunt de Lago et al. (2021), onze, on l'article en té un per cada bloc de
les seves dades, que no porten l'eòlica més solar de fa una setmana. L'estimador i el
pressupost són els de `models/search.py`, els de la xarxa de punt. L'ajust de cada prova
pren la llavor de l'execució, perquè el codi de l'article no en fixa cap.

Cada distribució és una cerca pròpia, `epf search --family ddnn_jsu` o `ddnn_normal`, amb
el seu estudi i el seu registre. Una cerca interrompuda no perd res: l'estudi es desa a
cada prova, i tornar-la a cridar la reprèn. `--trials 0` no en fa cap de nova i desa el
millor que l'estudi ja té.
"""
import datetime

import optuna

from epf import matrix
from epf.data import forecasts, price
from epf.models.blocks import blocks, columns_of, inputs
from epf.models.ddnn import ddnn, ddnn_hyperparameters
from epf.models.ddnn.distributions import FAMILIES
from epf.models.search import TRIALS, completed, guided, split, switched_on, window

# L'espai del codi de Marcjasz et al. (2023): dues capes amagades, les neurones de cada
# una, les sis activacions pels seus noms, el dropout de després de l'entrada, el
# coeficient de cada L1, tots en el mateix interval logarítmic, i la taxa d'aprenentatge,
# també logarítmica.
LAYERS = 2
NEURONS = (16, 1024)
ACTIVATIONS = ("sigmoid", "relu", "elu", "tanh", "softplus", "softmax")
DROPOUT = (0.0, 1.0)
L1 = (1e-5, 10.0)
LEARNING_RATE = (1e-5, 0.1)


def penalty(trial, name):
    """Una L1 com la demana el codi: primer si n'hi ha, i el coeficient només si n'hi ha.
    Sense, és zero, que és una L1 que no fa res."""
    if not trial.suggest_categorical(f"regularize_{name}", (False, True)):
        return 0.0
    return trial.suggest_float(f"{name}_l1", *L1, log=True)


def space(trial, names, distribution):
    """Els hiperparàmetres d'una prova sobre els blocs `names`, en la forma en què els
    llegeix `ddnn.build`.

    L'L1 de cada paràmetre es demana pel nom que té a `distribution`, i en el seu ordre:
    el codi de l'article posa el pes de la cua de Johnson SU abans de l'asimetria.
    """
    layers = range(1, LAYERS + 1)
    return {
        "blocks": switched_on(trial, names),
        "hidden": [trial.suggest_int(f"neurons_{layer}", *NEURONS) for layer in layers],
        "activation": [trial.suggest_categorical(f"activation_{layer}", ACTIVATIONS)
                       for layer in layers],
        "dropout": (trial.suggest_float("dropout_rate", *DROPOUT)
                    if trial.suggest_categorical("dropout", (False, True)) else 0.0),
        "l1": [penalty(trial, f"h{layer}_kernel") for layer in layers],
        "activity_l1": [penalty(trial, f"h{layer}_activation") for layer in layers],
        "parameter_l1": [penalty(trial, name) for name in distribution.names],
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
    }


def score(distribution, tuned, features, target):
    """La pèrdua de Keras d'un conjunt: un ajust amb `features`, puntuat a la seva quarta
    part final, la mateixa que l'atura, amb la xarxa de la seva millor època."""
    X = features[columns_of(features.columns, tuned["blocks"])]
    train, validate = split(len(X))
    fitted = ddnn.fit_rows(distribution, (X.iloc[train], target.iloc[train]),
                           (X.iloc[validate], target.iloc[validate]), tuned)
    return float(fitted.network.evaluate(
        inputs(X.iloc[validate], fitted.names).to_numpy(),
        target.iloc[validate].to_numpy(dtype=float), verbose=0))


def main(arguments):
    # `--trials 0` no és «cap valor»: és demanar que es desi el que l'estudi ja sap,
    # que és com es recupera una cerca que va caure abans d'arribar a desar res.
    if arguments.member is not None:
        raise SystemExit("cada distribució és una sola xarxa: --member només és per a "
                         "la xarxa de punt")
    name = arguments.family
    distribution = FAMILIES[name]
    trials = TRIALS if arguments.trials is None else arguments.trials
    X, y = matrix.lago(price.load(), forecasts.load()).daily
    features, target = window(X, y)
    names = list(blocks(features.columns))
    validated = features.index[split(len(features))[1]]
    print(f"{name}: {trials} proves, cada una d'un ajust sobre {len(features)} dies, de "
          f"{features.index[0].date()} a {features.index[-1].date()}, puntuat de "
          f"{validated[0].date()} a {validated[-1].date()}", flush=True)

    def objective(trial):
        return score(distribution, space(trial, names, distribution), features, target)

    # L'estudi es desa al costat del resultat, de manera que una cerca interrompuda es
    # reprèn en lloc de tornar a començar.
    store = ddnn_hyperparameters.path(arguments.inputs, name).with_suffix(".db")
    store.parent.mkdir(parents=True, exist_ok=True)
    searching = guided(store, name)
    searching.optimize(objective, n_trials=trials)

    done = completed(searching)
    if not done:
        print("cap prova acabada: no es desa res")
        return 1

    ddnn_hyperparameters.save(arguments.inputs, name, {
        "taken": str(datetime.date.today()),
        "trials": len(done),
        "first": str(features.index[0].date()),
        "last": str(features.index[-1].date()),
        "validated_from": str(validated[0].date()),
        "stage": arguments.inputs,
        "loss": searching.best_value,
        "hyperparameters": space(optuna.trial.FixedTrial(searching.best_params), names,
                                 distribution),
    })
    print(f"millor pèrdua {searching.best_value:.4f} de {len(done)} proves, desat a "
          f"{ddnn_hyperparameters.path(arguments.inputs, name)}")
    return 0
