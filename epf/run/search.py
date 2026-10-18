"""La cerca de Trebbien et al. (2023): a cada llavor i família, configuracions a l'atzar
de l'espai de la família, cada una ajustada amb les files d'entrenament de la llavor i
puntuada sobre les de validació, sense tocar mai les de prova, i les millors desades.

Cada família es puntua com ja es puntuava la seva cerca: l'arbre i la xarxa elàstica pel
MAE en EUR/MWh, i les xarxes per la seva pèrdua de Keras sobre la validació, amb les L1
a dins, que és el que torna el codi de Marcjasz et al. (2023). Una graella es recorre
sense repetir cap punt; un espai continu es mostreja a l'atzar. Quina matriu llegeix,
quina graella té i com es prova cada punt ho diu el `search.py` del model de la família.

Cada família en cada llavor és una tasca de `run/pool.py`, que en fa unes quantes
alhora; si alguna peta, les altres acaben i l'ordre ho diu en sortir.
"""

import importlib

import optuna

from epf import parameters, paths
from epf.contract import matrix, record, searches
from epf.contract.split import split
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families, pool

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


def _search(family, seed, smoke, trials, fingerprints):
    """La cerca d'una família en una llavor, una tasca de `pool`: l'estudi fins a
    `trials` intents, el seu registre desat, i la línia que el resumeix."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    module = searcher(family)
    X, y = getattr(matrix.matrices(matrix.series()), module.MATRIX)
    made = searches.run(
        searches.folder(smoke, family) / searches.STUDY.format(seed=seed),
        family,
        module,
        X,
        y,
        split(seed, DEVELOPMENT, BLACKOUT_WEEK),
        seed,
        trials,
        fingerprints,
    )
    found = searches.record(made, family, seed, smoke)
    searches.save(found, smoke)
    print(
        f"{family} llavor {seed}: {found['trials']} intents, el millor "
        f"{found['best'][0]['score']:.4g}",
        flush=True,
    )


def main(arguments):
    values = parameters.chosen(arguments.smoke)
    seeds = values["SEEDS"]
    # Tot el que es pot refusar, abans de cercar res. Les matrius les torna a llegir
    # cada tasca, i aquí només es refusen.
    try:
        chosen = families.named(arguments.family, SEARCHERS)
        if arguments.seed:
            outside = sorted(set(arguments.seed) - set(seeds))
            if outside:
                raise ValueError(
                    f"les llavors {outside} no hi són: "
                    f"{'un assaig' if arguments.smoke else 'una mesura'} en "
                    f"té {list(seeds)}"
                )
            seeds = [seed for seed in seeds if seed in arguments.seed]
        matrix.matrices(matrix.series())
        fingerprints = record.fingerprints()
    except ValueError as refused:
        print(refused)
        return 1

    for family in chosen:
        searches.folder(arguments.smoke, family).mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            f"{family} llavor {seed}",
            _search,
            (family, seed, arguments.smoke, values["TRIALS"], fingerprints),
        )
        for family in chosen
        for seed in seeds
    ]
    _, failed = pool.done(tasks, values["WORKERS"], values["THREADS"])
    if failed:
        print(f"han petat {len(failed)} cerques: {', '.join(failed)}")
        return 1
    print(f"les cerques són a {paths.SEARCHES}")
    return 0
