"""El protocol de Trebbien et al. (2023) sobre el període que es descriu, amb la
partició de `contract/split.py`: la robustesa és la dispersió entre llavors.

`epf describe` crida el `train.py` del model de cada família triada, l'una rere l'altra,
que l'ajusta a cada llavor, la puntua sobre les setmanes de prova i n'explica les files
de prova si el model s'explica. Abans d'ajustar-ne cap, planifica totes les famílies
triades amb `contract/study.py`: les millors configuracions de la cerca de cada llavor o,
mentre la cerca no s'ha fet, uns números que no s'han cercat, i el registre de cada
família ho diu. Tot es desa per llavor, de manera que les figures surten del que hi ha al
disc i no d'un ajust nou.
"""
import importlib
import json
import shutil
import time

from epf import parameters, paths
from epf.contract import features
from epf.contract.record import plain
from epf.contract.study import configurations
from epf.parameters import DESCRIPTIONS, MEASUREMENT, SMOKE
from epf.results import run_files
from epf.run import figures, search

# La carpeta de les passades dins de cada nivell, `runs/<nivell>/describe/<commit>/`,
# amb les famílies en carpetes per sota.
STAGE = "describe"

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


def main(arguments):
    commit = run_files.git("rev-parse", "HEAD")
    if arguments.tier == MEASUREMENT and run_files.git("status", "--porcelain"):
        print("l'arbre té canvis sense desar. El directori d'una mesura es diu pel "
              "commit, i amb canvis sense desar el nom no diria quin codi l'ha fet. "
              f"Fes un commit, o fes servir --tier {SMOKE}.")
        return 1
    chosen = arguments.family or list(TRAINERS)
    unknown = sorted(set(chosen) - set(TRAINERS))
    if unknown:
        print(f"no es descriu cap família {unknown}; n'hi ha {sorted(TRAINERS)}")
        return 1

    seeds, explained_days = DESCRIPTIONS[arguments.tier]
    where = paths.run_folder(arguments.tier, STAGE, commit[:7])
    # Reprendre és de la mesura i prou: l'assaig admet l'arbre brut, i dues còpies
    # diferents del codi en compartirien el directori.
    if arguments.tier == SMOKE or arguments.replace:
        for family in chosen:
            shutil.rmtree(where / family, ignore_errors=True)

    # Les configuracions de totes les famílies abans d'ajustar-ne cap, de manera que una
    # cerca a mitges o feta sobre unes altres files es refusa abans de gastar-hi res. Una
    # cerca es compara amb les columnes de la taula que llegeix la seva.
    trainers = {family: importlib.import_module(TRAINERS[family]) for family in chosen}
    tables = features.tables(*features.series())
    plans = {}
    try:
        for family in chosen:
            plan = trainers[family].FAMILIES[family]
            columns = None
            if plan["kept"]:
                X, _ = getattr(tables, search.searcher(plan["searched_as"]).TABLE)
                columns = X.columns
            plans[family] = configurations(arguments.tier, plan, columns, seeds)
    except ValueError as refused:
        print(refused)
        return 1
    swept = {name: getattr(parameters, name) for name in dir(parameters)
             if name.isupper() and not name.startswith("_")}
    for family in chosen:
        folder = where / family
        started = time.time()
        described_now = trainers[family].train(family, folder, plans[family],
                                               explained_days)
        record = {"commit": commit, "tier": arguments.tier, "family": family,
                  "seeds": list(seeds), "explained_days": explained_days,
                  "fixed": trainers[family].FAMILIES[family]["fixed"],
                  "configurations": plans[family], "described_now": described_now,
                  "seconds_now": round(time.time() - started, 1), "parameters": swept}
        (folder / paths.SEED_RECORD).write_text(json.dumps(plain(record), indent=2))
    figures.draw(where)
    print(f"la passada és a {where}, amb les seves figures")
    return 0
