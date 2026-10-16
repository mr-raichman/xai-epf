"""El protocol de Trebbien et al. (2023) sobre el període que es descriu, amb la
partició de `contract/split.py`: la robustesa és la dispersió entre llavors.

`epf run` crida el `train.py` del model de cada família triada, l'una rere l'altra, que
l'ajusta a cada llavor, la puntua sobre les setmanes de prova i n'explica les files de
prova si el model s'explica. Continua l'última passada del seu tipus mentre no és
acabada, amb totes les famílies, i si ho és en comença una altra; en diu el nom primer.
Abans d'ajustar-ne cap, planifica amb `contract/searches.py` les famílies triades que
encara no hi tenen pla: les millors configuracions de la cerca de cada llavor o, mentre
la cerca no s'ha fet, uns números que no s'han cercat. El pla queda al registre de la
família i ja no canvia, i cada ajust, el primer també, el llegeix d'allà. Tot es desa
per llavor, de manera que les figures surten del que hi ha al disc i no d'un ajust nou.
"""

import datetime
import importlib
import json
import time

from epf import parameters, paths
from epf.contract import matrix, record
from epf.contract.searches import configurations
from epf.run import families, figures, search

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
    for path in sorted(latest.glob(f"*/{paths.SEED_RECORD}")):
        made = json.loads(path.read_text(encoding="utf-8"))
        if any(made.get(name) != value for name, value in fingerprints.items()):
            raise ValueError(
                f"la passada {latest.name} es va fer amb un altre codi, un altre "
                f"conjunt de dades o unes altres biblioteques: acaba-la amb "
                f"aquells, o esborra-la"
            )
    return latest


def main(arguments):
    values = parameters.chosen(arguments.smoke)
    # Tot el que es pot refusar, abans d'escriure res: les famílies, el conjunt de
    # dades, la passada, i el pla de cada família que encara no en té, de manera que una
    # cerca a mitges o feta sobre unes altres files es refusa abans de gastar-hi res.
    # Una cerca es compara amb les columnes de la matriu que llegeix la seva. El nom de
    # la passada surt tan bon punt se sap, abans que els mòduls de les xarxes, en
    # importar-se, escriguin.
    try:
        chosen = families.named(arguments.family, TRAINERS)
        matrices = matrix.matrices(matrix.series())
        fingerprints = record.fingerprints()
        where = _where(arguments.smoke, fingerprints)
        print(f"passada {where.name}", flush=True)
        trainers = {
            family: importlib.import_module(TRAINERS[family]) for family in chosen
        }
        plans = {}
        for family in chosen:
            if families.recorded(where / family) is not None:
                continue
            plan = trainers[family].FAMILIES[family]
            columns = None
            if plan["kept"]:
                X, _ = getattr(matrices, search.searcher(plan["searched_as"]).MATRIX)
                columns = X.columns
            plans[family] = configurations(
                arguments.smoke, plan, columns, values["SEEDS"]
            )
    except ValueError as refused:
        print(refused)
        return 1
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
    # Una família que ja hi té tots els fitxers no s'ajusta, i el seu registre guarda el
    # temps de l'ajust que la va fer.
    for family in chosen:
        folder = where / family
        if families.complete(folder):
            continue
        started = time.time()
        described_now = trainers[family].train(
            family, folder, families.recorded(folder), values["EXPLAINED_DAYS"]
        )
        path = folder / paths.SEED_RECORD
        record.write_record(
            path,
            json.loads(path.read_text(encoding="utf-8"))
            | {
                "described_now": described_now,
                "seconds_now": round(time.time() - started, 1),
            },
        )
    figures.draw(where)
    print(f"la passada és a {where}, amb les seves figures")
    return 0
