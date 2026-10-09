"""Refà el que es deriva d'una execució ja feta: les mètriques i les figures.

Tot el que hi ha al directori d'una execució llevat de les prediccions i l'atribució
surt de les prediccions i del preu, de manera que adoptar una mètrica nova o canviar
una figura no ha de costar cap camí endavant. Les figures de SHAP surten del `.npz` i
no demanen cap ajust, així que també es refan aquí.

Viu a l'arrel del paquet i no a `results/` perquè toca les dues meitats d'una
execució, la que puntua i la que explica, i un subpaquet no coneix el nom del seu
germà (`architecture.md` 8.3).
"""
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


def main(arguments):
    folders = run_files.family_folders(arguments.folder)
    actual = matrix.price_only(price.load()).actual
    for folder in folders:
        # La combinació no hi passa: el seu pronòstic no surt de les prediccions que
        # té al costat sinó dels membres de tota la configuració, i qui la refà
        # sencera, figures incloses, és `combination.save`, aquí sota.
        if run_files.is_combination(folder):
            continue
        rebuild(folder, actual)
        print(f"refet {folder}")

    # La combinació es refà per cada configuració tocada, amb tots els membres que hi
    # hagi al disc, i les figures que comparen famílies darrere, perquè la volen a dins.
    for where in sorted({folder.resolve().parent for folder in folders}):
        counted = combination.save(where, run_files.recorded(where), actual)
        forecasts = combination.comparison(run_files.recorded(where))
        figures.shared(where, forecasts, actual)
        print(f"refetes les figures de {run_files.drawn(len(forecasts))} a {where}"
              + (f", amb la combinació de {counted} membres ajustats" if counted
                 else ", sense combinació"))
