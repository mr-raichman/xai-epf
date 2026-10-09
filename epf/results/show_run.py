"""Mètriques d'una o més famílies ja executades, en una sola taula.

Totes les carpetes que es donin van a la mateixa taula, de manera que
`epf show-run runs/.../<commit>/*` posa les famílies d'una execució una sota
l'altra, que és el que es compara, i la combinació hi arriba com una més: el seu
directori també porta `run.json`.
"""
import pandas as pd

from epf.results import run_files


def main(arguments):
    folders = run_files.family_folders(arguments.folder)
    print(run_files.summary(pd.concat([run_files.load_metrics(folder) for folder in folders])))
