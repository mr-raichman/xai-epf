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
import sys

import optuna

from epf import Refusal, parameters, paths
from epf.contract import matrix, record, searches
from epf.contract.split import split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families, pool, progress

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
    """Fa la cerca d'una família en una llavor, una tasca de `pool`: l'estudi fins a
    `trials` intents i el seu registre desat."""
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
    searches.save(searches.record(made, family, seed, smoke), smoke)


def main(arguments):
    values = parameters.chosen(arguments.smoke)
    seeds = values["SEEDS"]
    # Tot el que es pot refusar, abans de cercar res. Les matrius les torna a llegir
    # cada tasca, i aquí només es refusen.
    chosen = families.named(arguments.family, SEARCHERS)
    if not chosen:
        named = ", ".join(f'"{name}"' for name in arguments.family)
        raise Refusal(
            f"no es pot fer una cerca de {named}; les famílies que es cerquen: "
            f"{', '.join(SEARCHERS)}"
        )
    if arguments.seed:
        outside = sorted(set(arguments.seed) - set(seeds))
        if outside:
            kind = "un assaig" if arguments.smoke else "una mesura"
            asked = (
                f"la llavor {outside[0]}"
                if len(outside) == 1
                else f"les llavors {', '.join(map(str, outside))}"
            )
            raise Refusal(
                f"{kind} no té {asked}; les que té: {', '.join(map(str, seeds))}"
            )
        seeds = [seed for seed in seeds if seed in arguments.seed]
    matrix.matrices(matrix.series())
    split(seeds[0], DEVELOPMENT, BLACKOUT_WEEK)
    validation_parts(seeds[0], DEVELOPMENT, BLACKOUT_WEEK)
    fingerprints = record.fingerprints()

    for family in chosen:
        searches.folder(arguments.smoke, family).mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            (family, seed),
            _search,
            (family, seed, arguments.smoke, values["TRIALS"], fingerprints),
        )
        for family in chosen
        for seed in seeds
    ]
    block = progress.Block(
        "Cerca d'assaig" if arguments.smoke else "Cerca",
        {family: len(seeds) for family in chosen},
    )
    prefix = paths.SMOKE_PREFIX if arguments.smoke else ""
    log = paths.SEARCHES / f"{prefix}{paths.WORKERS_LOG}"
    _, failed = pool.done(tasks, values["WORKERS"], values["THREADS"], log, block)
    if failed:
        counted = progress.counted(failed, chosen)
        print(f"error: la cerca no és acabada: {counted}", file=sys.stderr)
        return 1
    return 0
