"""La comanda `epf run`, la passada del protocol de Trebbien et al. (2023)."""

import datetime
import importlib
import json
import time

from epf import Refusal, parameters, paths
from epf.contract import matrix, record
from epf.contract.searches import configurations
from epf.contract.split import split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT
from epf.run import families, figures, pool, progress, search

# Com diu el refús d'una passada continuada cada empremta que no és la d'ara: la que la
# va començar i aquella amb què s'ha d'acabar.
DIFFERENT = {
    "code": ("un altre codi", "aquell codi"),
    "dataset": ("un altre dataset", "aquell dataset"),
    "environment": ("unes altres biblioteques", "aquelles biblioteques"),
}

# Cada família, en l'ordre en què una passada les ajusta, amb el mòdul del seu model que
# l'ajusta, que s'importa quan es tria la família.
TRAINERS = {
    "ddnn_jsu": "epf.models.nn.ddnn.train",
    "ddnn_normal": "epf.models.nn.ddnn.train",
    "dnn": "epf.models.nn.dnn.train",
    "lightgbm": "epf.models.lightgbm.train",
    "date_probe": "epf.models.lightgbm.date_probe",
    "elastic_net": "epf.models.elastic_net.train",
    "floor": "epf.models.floor.train",
    "naive_similar_day": "epf.models.naive.train",
    "naive_previous_week": "epf.models.naive.train",
    "naive_previous_day": "epf.models.naive.train",
}


def _where(smoke, fingerprints):
    """Torna l'última passada del tipus si no és acabada, o la següent d'avui.

    Refusa continuar-ne una que van començar un altre codi, un altre dataset o unes
    altres biblioteques, perquè les seves famílies no es podrien comparar, i una
    d'assaig començada amb una altra variant que la d'ara, `smoke`, que hi barrejaria
    els valors de dues. Un registre sense una de les empremtes d'ara és d'un altre codi.
    """
    found = paths.runs(smoke)
    if not found or families.finished(found[-1]):
        return paths.next_run(smoke, datetime.date.today())
    latest = found[-1]
    differ, begun = set(), set()
    for path in sorted(latest.glob(f"*/{paths.SEED_RECORD}")):
        made = json.loads(path.read_text(encoding="utf-8"))
        differ |= {
            name for name, value in fingerprints.items() if made.get(name) != value
        }
        begun.add(made.get("smoke"))
    if differ:
        then = [DIFFERENT[name] for name in DIFFERENT if name in differ]
        raise Refusal(
            f"la passada {latest.name} es va començar amb "
            f"{' i '.join(other for other, _ in then)}; continuar amb "
            f"{' i '.join(that for _, that in then)}, o esborrar "
            f"{paths.relative(latest)}/"
        )
    variants = sorted(map(str, begun - {smoke}))
    if variants:
        flags = {name: flag for flag, name in parameters.variants().items()}
        flag = flags.get(variants[0], variants[0])
        raise Refusal(
            f'la passada {latest.name} es va començar amb "{flag}"; continuar amb '
            f'"uv run epf run {flag}", o esborrar {paths.relative(latest)}/'
        )
    return latest


def _train(module, family, folder, seed, configuration, explained_days):
    """Ajusta una configuració d'una família en una seed, una tasca de `pool`.

    Crida el `train` del mòdul del model, `module`, amb un pla d'aquesta seed i aquesta
    configuració soles. Torna què ha desat i quants segons ha trigat.
    """
    started = time.time()
    described = importlib.import_module(module).train(
        family, folder, {seed: [configuration]}, explained_days
    )
    return {"described": described, "seconds": time.time() - started}


def main(arguments):
    """Fa una passada, la comanda `epf run`.

    Continua l'última del seu tipus si no és acabada. Les famílies triades que encara no
    hi tenen pla en reben un, que queda al seu registre i ja no canvia: les millors
    configuracions de la cerca de cada seed o, sense cerca, uns números que no s'han
    cercat. Ajusta cada configuració que encara no té el seu fitxer, unes quantes
    alhora, i dibuixa les figures si cap seed no ha fallat. Refusa abans d'escriure res
    el que no pot fer, i acaba amb un refús si alguna seed ha fallat.
    """
    values = parameters.chosen(arguments.smoke)
    chosen = families.named(arguments.family, TRAINERS)
    matrices = matrix.matrices(matrix.series())
    matrix.refuse_missing_days(matrices, DEVELOPMENT, BLACKOUT_WEEK)
    # La partició i les parts de validació, només per refusar-les abans d'escriure res.
    split(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    validation_parts(values["SEEDS"][0], DEVELOPMENT, BLACKOUT_WEEK)
    fingerprints = record.fingerprints()
    where = _where(arguments.smoke, fingerprints)
    continued = where.exists()
    trainers = {family: importlib.import_module(TRAINERS[family]) for family in chosen}
    plans = {}
    for family in chosen:
        if families.recorded(where / family) is not None:
            continue
        plan = trainers[family].FAMILIES[family]
        columns = None
        # La cerca es compara amb les columnes de la matriu del seu `search.py`.
        if plan["kept"]:
            X, _ = getattr(matrices, search.searcher(plan["searched_as"]).MATRIX)
            columns = X.columns
        plans[family] = configurations(
            arguments.smoke,
            plan,
            columns,
            values["SEEDS"],
            values["BEST"],
        )
    where.mkdir(parents=True, exist_ok=True)
    matrix.write(matrices, where)
    # Els trials són els de la cerca, que porta cada configuració, i el calendari no
    # canvia mai.
    left_out = {"TRIALS", "DAYS_PER_WEEK", "HOURS_PER_DAY", "HOUR_COLUMNS"}
    written = {name: value for name, value in values.items() if name not in left_out}
    for family, planned in plans.items():
        record.write_record(
            where / family / paths.SEED_RECORD,
            {
                "family": family,
                "smoke": arguments.smoke,
                **fingerprints,
                "seeds": values["SEEDS"],
                "explained_days": values["EXPLAINED_DAYS"],
                "fixed": trainers[family].FAMILIES[family]["fixed"],
                "configurations": planned,
                "described_now": None,
                "seconds_now": None,
                "parameters": written,
            },
        )
    # Cada ajust, el primer també, llegeix el pla del registre.
    recorded = {family: families.recorded(where / family) for family in chosen}
    missing = {family: families.missing(where / family) for family in chosen}
    block = progress.Block(
        f"Passada {where.name}{', continuada' if continued else ''}",
        "fit",
        {
            family: {
                seed: (len(planned), len(planned) - len(missing[family].get(seed, [])))
                for seed, planned in recorded[family].items()
            }
            for family in chosen
        },
    )
    # Un generador: el pool en llegeix una tasca quan un procés queda lliure, i ja no
    # llegeix les d'una seed que ha fallat.
    tasks = (
        (
            (family, seed, one["rank"]),
            _train,
            (
                TRAINERS[family],
                family,
                where / family,
                seed,
                one,
                values["EXPLAINED_DAYS"],
            ),
        )
        for family in chosen
        for seed, ones in missing[family].items()
        for one in ones
        if not block.stopped(family, seed)
    )
    results = pool.done(
        tasks, values["WORKERS"], values["THREADS"], where / paths.WORKERS_LOG, block
    )
    for family in chosen:
        found = sorted(
            (label, one) for label, one in results.items() if label[0] == family
        )
        if not found:
            continue
        seconds = {}
        for (_, seed, _), one in found:
            seconds[seed] = seconds.get(seed, 0) + one["seconds"]
        path = where / family / paths.SEED_RECORD
        record.write_record(
            path,
            json.loads(path.read_text(encoding="utf-8"))
            | {
                "described_now": [
                    pair for _, one in found for pair in one["described"]
                ],
                "seconds_now": {
                    seed: round(total, 1) for seed, total in seconds.items()
                },
            },
        )
    if block.failed:
        counted = progress.counted(block.failed, chosen)
        raise Refusal(f"la passada {where.name} no és acabada: {counted}")
    figures.draw(where)
