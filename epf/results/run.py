"""Què hi ha dins del directori d'una execució, i on és.

Un directori per configuració i família, perquè el que es compara és una execució
sencera: totes les famílies d'una configuració, de cop. El període és dins de
`run.json` i dins del nom que se li posa a la configuració.

Tot el que hi ha aquí es deriva de les prediccions i del preu, de manera que es pot
tornar a calcular sense tornar a ajustar res. L'atribució no: aquella surt del model
que va predir el dia i la desa `models/attribution.py`.
"""
import json

import pandas as pd

from epf import paths

FORECASTS = "forecasts_{name}.csv"
METRICS = "metrics.csv"
RECORD = "run.json"


def run_folder(tier, stage, commit):
    """On viu una execució sencera: hi pengen les famílies i les figures que les
    comparen. El commit la nomena, perquè és el que fixa tots els seus números."""
    return paths.RUNS / tier / stage / commit


def family_folders(paths):
    """Els directoris de família que hi ha darrere dels camins donats.

    Un camí que ja ho és val tal qual; un que no ho és aporta els seus
    subdirectoris que sí que ho són, de manera que tant el directori d'una
    configuració com un glob que l'obre valen igual. El glob també agafa les
    figures de la configuració, que no són directoris i se salten.
    """
    found = []
    for path in paths:
        if not path.is_dir():
            continue
        if (path / RECORD).exists():
            found.append(path)
        else:
            found += sorted(child for child in path.iterdir() if (child / RECORD).exists())
    if not found:
        raise ValueError(f"cap directori de família a {[str(path) for path in paths]}")
    return found


def save_forecasts(folder, forecasts):
    for name, frame in sorted(forecasts.items()):
        frame.to_csv(folder / FORECASTS.format(name=name))


def load_forecasts(folder):
    """Els pronòstics que hi ha desats, pel nom que duien quan es van escriure."""
    found = sorted(folder.glob(FORECASTS.format(name="*")))
    if not found:
        raise ValueError(f"cap fitxer de prediccions a {folder}")
    prefix = FORECASTS.format(name="")[:-len(".csv")]
    return {path.stem.removeprefix(prefix): pd.read_csv(path, index_col=0, parse_dates=True)
            for path in found}


def save_metrics(folder, table):
    table.to_csv(folder / METRICS, index=False)


def load_metrics(folder):
    return pd.read_csv(folder / METRICS)


def summary(table):
    """Les mètriques d'un sol valor, un pronòstic per fila i una mètrica per columna.

    L'ordre de les columnes és el que porta la taula i no l'alfabètic, de manera
    que qui decideix com es llegeix una execució és `evaluation.metrics`, que és qui
    les calcula.
    """
    # La combinació és a `metrics.csv` de cada família, així que llegint-ne
    # diverses arriba repetida. Files idèntiques es queden en una; si dues
    # execucions donessin valors diferents, totes dues sobreviuen i el pivot peta,
    # que és el que ha de passar.
    table = table.drop_duplicates()
    singles = table[table["hour"].isna() | (table["hour"] == "")]
    order = list(singles["metric"].drop_duplicates())
    wide = singles.pivot(index="forecast", columns="metric", values="value")
    return wide[order].to_string(float_format=lambda value: f"{value:9.3f}")


def save_record(folder, record):
    (folder / RECORD).write_text(json.dumps(record, indent=2) + "\n")


def load_record(folder):
    return json.loads((folder / RECORD).read_text())
