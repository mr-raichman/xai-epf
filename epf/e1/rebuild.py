"""Refà el que es deriva d'una execució ja feta: les mètriques i les figures.

Tot el que hi ha al directori d'una execució llevat de les prediccions i
l'atribució surt de les prediccions i del preu, de manera que adoptar una mètrica
nova o canviar una figura no ha de costar cap camí endavant.
"""
import argparse
import pathlib

from epf import run as run_files
from epf.e1 import features, figures, runner
from epf.evaluation import metrics
from epf.models import naive


def rebuild(folder, actual):
    """Reescriu `metrics.csv` i torna a dibuixar les figures d'una família."""
    forecasts = run_files.load_forecasts(folder)
    run_files.save_metrics(folder, metrics(forecasts, actual, naive.weekly_frame(actual)))
    figures.draw(folder, actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una execució o d'una de les seves famílies")
    folders = run_files.family_folders(parser.parse_args().folder)

    (_, actual), _ = features.load()
    for folder in folders:
        rebuild(folder, actual)
        print(f"refet {folder}")

    # La combinació viu dins del `metrics.csv` de cada família i la reconstrucció
    # l'acaba de sobreescriure, així que es torna a fer per cada configuració
    # tocada, amb totes les famílies que hi hagi al disc.
    for where in sorted({folder.resolve().parent for folder in folders}):
        family_folders = run_files.family_folders([where])
        counted = runner.add_combination(family_folders, actual)
        figures.compare(where, family_folders, actual)
        print(f"refeta la comparació de {where}, amb la combinació de {counted} membres")


if __name__ == "__main__":
    main()
