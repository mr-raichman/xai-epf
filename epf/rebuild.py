"""Refà el que es deriva d'una execució ja feta: les mètriques i les figures.

Tot el que hi ha al directori d'una execució llevat de les prediccions i l'atribució
surt de les prediccions i del preu, de manera que adoptar una mètrica nova o canviar
una figura no ha de costar cap camí endavant. Les figures de SHAP surten del `.npz` i
no demanen cap ajust, així que també es refan aquí.

Viu a l'arrel del paquet i no a `results/` perquè toca les dues meitats d'una
execució, la que puntua i la que explica, i un subpaquet no coneix el nom del seu
germà (`architecture.md` 8.3).
"""
import argparse
import pathlib

from epf import matrix
from epf.attribution import figures as shap_figures
from epf.attribution import stored
from epf.data import price
from epf.evaluation import metrics
from epf.models.naive import naive
from epf.results import combination, figures, run_files


def rebuild(folder, actual):
    """Reescriu `metrics.csv` i torna a dibuixar totes les figures d'una família."""
    forecasts = run_files.load_forecasts(folder)
    run_files.save_metrics(folder, metrics(forecasts, actual, naive.weekly_frame(actual)))
    figures.draw(folder, actual)
    for name in sorted(forecasts):
        if stored.path(folder, name).exists():
            shap_figures.draw(folder, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path, nargs="+",
                        help="directori d'una execució o d'una de les seves famílies")
    folders = run_files.family_folders(parser.parse_args().folder)

    actual = matrix.price_only(price.load()).actual
    for folder in folders:
        rebuild(folder, actual)
        print(f"refet {folder}")

    # La combinació viu dins del `metrics.csv` de cada família i la reconstrucció
    # l'acaba de sobreescriure, així que es torna a fer per cada configuració
    # tocada, amb totes les famílies que hi hagi al disc.
    for where in sorted({folder.resolve().parent for folder in folders}):
        family_folders = run_files.family_folders([where])
        counted = combination.add_combination(family_folders, actual)
        figures.shared(where, combination.comparison(family_folders, actual), actual)
        print(f"refetes les figures de {run_files.drawn(len(family_folders))} a {where}"
              + (f", i la combinació de {counted} membres ajustats" if counted
                 else ", sense combinació"))


if __name__ == "__main__":
    main()
