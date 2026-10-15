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
import shutil

import optuna

from epf import paths
from epf.contract import features
from epf.contract.split import split
from epf.contract.study import STUDY, folder, record, run, save
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, MEASUREMENT, SEARCH_TRIALS, SMOKE
from epf.results import run_files

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
    commit = run_files.git("rev-parse", "HEAD")
    if arguments.tier == MEASUREMENT and run_files.git("status", "--porcelain"):
        print("l'arbre té canvis sense desar. Una cerca de mesura desa el commit que la fa, "
              "i amb canvis sense desar no diria quin codi l'ha feta. Fes un commit, o fes "
              f"servir --tier {SMOKE}.")
        return 1
    chosen = arguments.family or list(SEARCHERS)
    unknown = sorted(set(chosen) - set(SEARCHERS))
    if unknown:
        print(f"no es cerca cap família {unknown}; n'hi ha {sorted(SEARCHERS)}")
        return 1

    seeds, trials = SEARCH_TRIALS[arguments.tier]
    if arguments.seed:
        outside = sorted(set(arguments.seed) - set(seeds))
        if outside:
            print(f"les llavors {outside} no són del nivell {arguments.tier}, que en té "
                  f"{list(seeds)}")
            return 1
        seeds = [seed for seed in seeds if seed in arguments.seed]
    # Reprendre és de la mesura i prou, com a la passada.
    if arguments.tier == SMOKE or arguments.replace:
        for family in chosen:
            shutil.rmtree(folder(arguments.tier, family), ignore_errors=True)

    tables = features.tables(*features.series())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in chosen:
        module = searcher(family)
        X, y = getattr(tables, module.TABLE)
        where = folder(arguments.tier, family)
        where.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            study = run(where / STUDY.format(seed=seed), family, module, X, y,
                        split(seed, DEVELOPMENT, BLACKOUT_WEEK), seed, trials)
            found = record(study, family, seed, arguments.tier, commit)
            save(found, arguments.tier)
            print(f"{family} llavor {seed}: {found['trials']} proves, la millor "
                  f"{found['best'][0]['score']:.4g}", flush=True)
    print(f"les cerques són a {paths.SEARCHES / arguments.tier}")
    return 0
