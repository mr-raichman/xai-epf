"""Refà el que es deriva d'una execució ja feta: les mètriques i les figures.

Tot el que hi ha al directori d'una execució llevat de les prediccions i
l'atribució surt de les prediccions i del preu, de manera que adoptar una mètrica
nova o canviar una figura no ha de costar cap camí endavant.
"""
import argparse
import pathlib

from epf import run as run_files
from epf.e1 import features, figures
from epf.evaluation import metrics


def rebuild(folder, actual):
    """Reescriu `metrics.csv` i torna a dibuixar les figures d'una execució."""
    forecasts = run_files.load_forecasts(folder)
    run_files.save_metrics(folder, metrics(forecasts, actual))
    figures.draw(folder, actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una execució ja feta")
    folders = parser.parse_args().folder

    (_, actual), _ = features.load()
    for folder in folders:
        rebuild(folder, actual)
        print(f"refet {folder}")


if __name__ == "__main__":
    main()
