"""La comanda `epf search`, la cerca de Trebbien et al. (2023)."""

import importlib

import optuna

from epf import Refusal, parameters, paths
from epf.contract import matrix, record, searches
from epf.contract.split import split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families, pool, progress

# Cada família que es cerca, en l'ordre en què es cerquen, amb el `search.py` del seu
# model, que s'importa quan la família es cerca o es planifica.
SEARCHERS = {
    "ddnn_jsu": "epf.models.nn.search",
    "ddnn_normal": "epf.models.nn.search",
    "dnn": "epf.models.nn.search",
    "lightgbm": "epf.models.lightgbm.search",
    "elastic_net": "epf.models.elastic_net.search",
}


def searcher(family):
    """Torna el `search.py` del model d'una família que es cerca."""
    return importlib.import_module(SEARCHERS[family])


def _trial(family, seed, drawn):
    """Ajusta els valors d'un trial d'una família en una seed, una tasca de `pool`.

    Torna els números i la puntuació.
    """
    module = searcher(family)
    X, y = getattr(matrix.matrices(matrix.series()), module.MATRIX)
    return searches.tried(
        module, family, drawn, X, y, split(seed, DEVELOPMENT, BLACKOUT_WEEK), seed
    )


def main(arguments):
    """Fa una cerca, la comanda `epf search`.

    Fa els trials que falten a cada família triada a cada seed, o a les de `--seed`,
    uns quants alhora, i desa el registre de cada seed que els té tots. Refusa abans de
    cercar res el que no pot fer, i acaba amb un refús si alguna seed ha fallat.
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
                f"la seed {outside[0]}"
                if len(outside) == 1
                else f"les seeds {', '.join(map(str, outside))}"
            )
            raise Refusal(
                f"{kind} no té {asked}; les que té: {', '.join(map(str, seeds))}"
            )
        seeds = [seed for seed in seeds if seed in arguments.seed]
    # Al pare, cada study que crea seria una línia d'Optuna a sobre del bloc.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    matrices = matrix.matrices(matrix.series())
    matrix.refuse_missing_days(matrices, DEVELOPMENT, BLACKOUT_WEEK)
    # La partició i les parts de validació, només per refusar-les abans de cercar res.
    split(seeds[0], DEVELOPMENT, BLACKOUT_WEEK)
    validation_parts(seeds[0], DEVELOPMENT, BLACKOUT_WEEK)
    fingerprints = record.fingerprints()
    trials = values["TRIALS"]
    studies = {}
    for family in chosen:
        module = searcher(family)
        X, _ = getattr(matrices, module.MATRIX)
        folder = searches.folder(arguments.smoke, family)
        folder.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            studies[family, seed] = searches.opened(
                folder / searches.STUDY.format(seed=seed),
                family,
                module,
                seed,
                X.columns,
                fingerprints["dataset"],
            )
    done = {
        key: min(len(searches.completed(study)), trials)
        for key, study in studies.items()
    }
    # Una seed que ja els té tots, com quan la cerca s'atura entre el seu últim trial i
    # el seu registre.
    for (family, seed), study in studies.items():
        if done[family, seed] == trials:
            searches.save(
                searches.record(study, family, seed, arguments.smoke), arguments.smoke
            )
    block = progress.Block(
        "Cerca d'assaig" if arguments.smoke else "Cerca",
        "trial",
        {
            family: {seed: (trials, done[family, seed]) for seed in seeds}
            for family in chosen
        },
    )
    # El trial que el pare ha demanat per cada tasca que encara no ha tornat.
    pending = {}

    def tasks():
        # Un generador: el pool en llegeix una tasca quan un procés queda lliure, i el
        # trial es demana llavors; una llista els demanaria tots d'entrada.
        for (family, seed), study in studies.items():
            for _ in range(trials - done[family, seed]):
                if block.stopped(family, seed):
                    break
                optuna_trial, drawn = searches.asked(study, searcher(family), family)
                label = (family, seed, optuna_trial.number)
                pending[label] = optuna_trial
                yield label, _trial, (family, seed, drawn)

    def returned(label, found):
        family, seed, _ = label
        study = studies[family, seed]
        numbers, score = found
        searches.told(study, pending.pop(label), numbers, score, fingerprints)
        if len(searches.completed(study)) == trials:
            searches.save(
                searches.record(study, family, seed, arguments.smoke), arguments.smoke
            )

    def failed(label):
        family, seed, _ = label
        searches.abandoned(studies[family, seed], pending.pop(label))

    prefix = paths.SMOKE_PREFIX if arguments.smoke else ""
    log = paths.SEARCHES / f"{prefix}{paths.WORKERS_LOG}"
    pool.done(
        tasks(), values["WORKERS"], values["THREADS"], log, block, returned, failed
    )
    if block.failed:
        counted = progress.counted(block.failed, chosen)
        raise Refusal(f"la cerca no és acabada: {counted}")
