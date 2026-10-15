"""La cerca de Trebbien et al. (2023): a cada llavor i família, configuracions a l'atzar de
l'espai de la família, cada una ajustada amb les files d'entrenament de la llavor i
puntuada sobre les de validació, sense tocar mai les de prova, i les millors desades.

Cada família es puntua com ja es puntuava la seva cerca: l'arbre i la xarxa elàstica pel
MAE en EUR/MWh, i les xarxes per la seva pèrdua de Keras sobre la validació, amb les L1 a
dins, que és el que torna el codi de Marcjasz et al. (2023). Una graella es recorre sense
repetir cap punt; un espai continu es mostreja a l'atzar. Quina taula llegeix, quina
graella té i com es prova cada punt ho diu el `search.py` del model de la família.
"""
import importlib

import optuna

from epf import parameters, paths
from epf.contract import features, record, study
from epf.contract.split import split
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families

# Cada família que es cerca, en l'ordre en què es cerquen, amb el `search.py` del seu
# model, que s'importa quan la família es cerca o es planifica.
SEARCHERS = {
    "lightgbm": "epf.models.lightgbm.search",
    "elastic_net": "epf.models.elastic_net.search",
    "ddnn_normal": "epf.models.nn.search",
    "ddnn_jsu": "epf.models.nn.search",
    "dnn": "epf.models.nn.search",
}


def searcher(family):
    """El `search.py` del model d'una família que es cerca."""
    return importlib.import_module(SEARCHERS[family])


def main(arguments):
    values = parameters.chosen(arguments.smoke)
    seeds = values["SEEDS"]
    # Tot el que es pot refusar, abans de cercar res.
    try:
        chosen = families.named(arguments.family, SEARCHERS)
        if arguments.seed:
            outside = sorted(set(arguments.seed) - set(seeds))
            if outside:
                raise ValueError(f"les llavors {outside} no hi són: "
                                 f"{'un assaig' if arguments.smoke else 'una mesura'} en "
                                 f"té {list(seeds)}")
            seeds = [seed for seed in seeds if seed in arguments.seed]
        tables = features.tables(features.series())
        fingerprints = record.fingerprints()
    except ValueError as refused:
        print(refused)
        return 1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in chosen:
        module = searcher(family)
        X, y = getattr(tables, module.TABLE)
        where = study.folder(arguments.smoke, family)
        where.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            made = study.run(where / study.STUDY.format(seed=seed), family, module, X, y,
                             split(seed, DEVELOPMENT, BLACKOUT_WEEK), seed,
                             values["TRIALS"], fingerprints)
            found = study.record(made, family, seed, arguments.smoke)
            study.save(found, arguments.smoke)
            print(f"{family} llavor {seed}: {found['trials']} proves, la millor "
                  f"{found['best'][0]['score']:.4g}", flush=True)
    print(f"les cerques són a {paths.SEARCHES}")
    return 0
