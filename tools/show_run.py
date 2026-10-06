"""Mètriques d'una o més famílies ja executades, en una sola taula.

Totes les carpetes que es donin van a la mateixa taula, de manera que
`show_run.py runs/.../<commit>_base/*` posa les famílies d'una configuració una
sota l'altra, que és el que es compara.
"""
import argparse
import pathlib

import pandas as pd

from epf import run as run_files

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=pathlib.Path, nargs="+",
                    help="directori d'una execució o d'una de les seves famílies")
folders = run_files.family_folders(parser.parse_args().folder)

print(run_files.summary(pd.concat([run_files.load_metrics(folder) for folder in folders])))
