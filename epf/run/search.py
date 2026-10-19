"""La comanda `epf search`, la cerca de Trebbien et al. (2023)."""

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
    """Torna el `search.py` del model d'una família que es cerca."""
    return importlib.import_module(SEARCHERS[family])


def _search(family, seed, smoke, trials, fingerprints):
    """Fa la cerca d'una família en una llavor, una tasca de `pool`.

    Fa el study fins a `trials` intents i en desa el registre.
    """
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
    """Fa una cerca, la comanda `epf search`, i torna 0, o 1 si alguna llavor ha fallat.

    Fa la cerca de cada família triada a cada llavor, o a les de `--seed`, unes quantes
    alhora, i en desa el study i el registre. Tot el que refusa, ho refusa abans de
    cercar res.
    """
    values = parameters.chosen(arguments.smoke)
    seeds = values["SEEDS"]
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
    # Les matrius, la partició i les parts de validació, només per refusar-les abans de
    # cercar res: cada tasca les torna a fer.
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
