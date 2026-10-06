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

from epf.config import paths

FORECASTS = "forecasts_{name}.csv"
METRICS = "metrics.csv"
RECORD = "run.json"


def folder(tier, stage, config, family):
    return paths.RUNS / tier / stage / config / family


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


def save_record(folder, record):
    (folder / RECORD).write_text(json.dumps(record, indent=2) + "\n")
