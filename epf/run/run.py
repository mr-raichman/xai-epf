"""La comanda `epf run`, la passada del protocol de Trebbien et al. (2023)."""

import datetime
import importlib
import json
import sys
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
    "dataset": ("un altre conjunt de dades", "aquell conjunt de dades"),
    "environment": ("unes altres biblioteques", "aquelles biblioteques"),
}

# Cada família, en l'ordre en què una passada les ajusta, amb el mòdul del seu model que
# l'ajusta, que s'importa quan es tria la família.
TRAINERS = {
    "lightgbm": "epf.models.lightgbm.train",
    "date_probe": "epf.models.lightgbm.date_probe",
    "elastic_net": "epf.models.elastic_net.train",
    "ddnn_normal": "epf.models.nn.ddnn.train",
    "ddnn_jsu": "epf.models.nn.ddnn.train",
    "dnn": "epf.models.nn.dnn.train",
    "floor": "epf.models.floor.train",
    "naive_previous_day": "epf.models.naive.train",
    "naive_previous_week": "epf.models.naive.train",
    "naive_similar_day": "epf.models.naive.train",
}


def _where(smoke, fingerprints):
    """Torna l'última passada del tipus si no és acabada, o la següent d'avui.

    Refusa continuar-ne una que van començar un altre codi, un altre conjunt de dades o
    unes altres biblioteques, perquè les seves famílies no es podrien comparar. Un
    registre sense una de les empremtes d'ara és d'un altre codi.
    """
    found = paths.runs(smoke)
    if not found or families.finished(found[-1]):
        return paths.next_run(smoke, datetime.date.today())
    latest = found[-1]
    differ = set()
    for path in sorted(latest.glob(f"*/{paths.SEED_RECORD}")):
        made = json.loads(path.read_text(encoding="utf-8"))
        differ |= {
            name for name, value in fingerprints.items() if made.get(name) != value
        }
    if differ:
        then = [DIFFERENT[name] for name in DIFFERENT if name in differ]
        raise Refusal(
            f"la passada {latest.name} es va començar amb "
            f"{' i '.join(other for other, _ in then)}; continuar amb "
            f"{' i '.join(that for _, that in then)}, o esborrar "
            f"{paths.relative(latest)}/"
        )
    return latest


def _train(module, family, folder, seed, planned, explained_days):
    """Ajusta una família en una llavor, una tasca de `pool`.

    Crida el `train` del mòdul del model, `module`, amb el pla d'aquesta llavor sola,
    `planned`. Torna què ha desat i quant ha trigat.
    """
    started = time.time()
    described = importlib.import_module(module).train(
        family, folder, {seed: planned}, explained_days
    )
    return {
        "family": family,
        "seed": seed,
        "described": described,
        "seconds": round(time.time() - started, 1),
    }


def main(arguments):
    """Fa una passada, la comanda `epf run`, i torna 0, o 1 si alguna llavor ha fallat.

    Continua l'última del seu tipus si no és acabada. Les famílies triades que encara no
    hi tenen pla en reben un, que queda al seu registre i ja no canvia: les millors
    configuracions de la cerca de cada llavor o, sense cerca, uns números que no s'han
    cercat. Ajusta cada llavor a la qual falta algun fitxer, unes quantes alhora, i
    dibuixa les figures si cap no ha fallat. Tot el que refusa, ho refusa abans
    d'escriure res.
    """
    values = parameters.chosen(arguments.smoke)
    chosen = families.named(arguments.family, TRAINERS)
    matrices = matrix.matrices(matrix.series())
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
        plans[family] = configurations(arguments.smoke, plan, columns, values["SEEDS"])
    where.mkdir(parents=True, exist_ok=True)
    matrix.write(matrices, where)
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
                "parameters": values,
            },
        )
    # Cada ajust, el primer també, llegeix el pla del registre.
    missing = {family: families.missing(where / family) for family in chosen}
    tasks = [
        (
            (family, seed),
            _train,
            (
                TRAINERS[family],
                family,
                where / family,
                seed,
                planned,
                values["EXPLAINED_DAYS"],
            ),
        )
        for family in chosen
        for seed, planned in missing[family].items()
    ]
    totals = {family: len(families.recorded(where / family)) for family in chosen}
    block = progress.Block(
        f"Passada {where.name}{', continuada' if continued else ''}",
        totals,
        {family: totals[family] - len(missing[family]) for family in chosen},
    )
    results, failed = pool.done(
        tasks, values["WORKERS"], values["THREADS"], where / paths.WORKERS_LOG, block
    )
    for family in chosen:
        found = sorted(
            (one for one in results.values() if one["family"] == family),
            key=lambda one: one["seed"],
        )
        if not found:
            continue
        path = where / family / paths.SEED_RECORD
        record.write_record(
            path,
            json.loads(path.read_text(encoding="utf-8"))
            | {
                "described_now": [pair for one in found for pair in one["described"]],
                "seconds_now": {one["seed"]: one["seconds"] for one in found},
            },
        )
    if failed:
        counted = progress.counted(failed, chosen)
        print(
            f"error: la passada {where.name} no és acabada: {counted}", file=sys.stderr
        )
        return 1
    figures.draw(where)
    return 0
