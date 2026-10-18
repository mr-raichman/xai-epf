"""El protocol de Trebbien et al. (2023) sobre el període que es descriu, amb la
partició de `contract/split.py`: la robustesa és la dispersió entre llavors.

`epf run` fa de cada família triada, a cada llavor a la qual falta algun fitxer, una
tasca de `run/pool.py`, que en fa unes quantes alhora: el `train.py` del model de la
família l'ajusta en aquella llavor, la puntua sobre les setmanes de prova i n'explica
les files de prova si el model s'explica. Si alguna tasca peta, les altres acaben, la
passada queda a mitges i les figures no es dibuixen. Continua l'última passada del seu
tipus mentre no és acabada, amb totes les famílies, i si ho és en comença una altra; en
diu el nom primer. Abans d'ajustar-ne cap, planifica amb `contract/searches.py` les
famílies triades que encara no hi tenen pla: les millors configuracions de la cerca de
cada llavor o, mentre la cerca no s'ha fet, uns números que no s'han cercat. El pla
queda al registre de la família i ja no canvia, i cada ajust, el primer també, el
llegeix d'allà. Tot es desa per llavor, de manera que les figures surten del que hi ha
al disc i no d'un ajust nou.
"""

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
    """La passada on s'escriu: l'última del tipus si no és acabada, i si no la següent
    d'avui. Continuar-ne una que han fet un altre codi, un altre conjunt de dades o unes
    altres biblioteques es refusa, perquè les seves famílies no es podrien comparar. Un
    registre sense una de les empremtes d'ara és d'un altre codi."""
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
    """L'ajust d'una família en una llavor, una tasca de `pool`: el `train.py` del seu
    model, `module`, amb el pla d'aquesta llavor sola. Torna què ha desat i quant ha
    trigat."""
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
    values = parameters.chosen(arguments.smoke)
    # Tot el que es pot refusar, abans d'escriure res: les famílies, el conjunt de
    # dades, la partició, la passada, i el pla de cada família que encara no en té, de
    # manera que una cerca a mitges o feta sobre unes altres files es refusa abans de
    # gastar-hi res. Una cerca es compara amb les columnes de la matriu que llegeix la
    # seva.
    chosen = families.named(arguments.family, TRAINERS)
    matrices = matrix.matrices(matrix.series())
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
    # Una llavor que ja hi té tots els fitxers no s'ajusta, i el registre de la seva
    # família guarda el que va desar i va trigar cada llavor l'última vegada.
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
