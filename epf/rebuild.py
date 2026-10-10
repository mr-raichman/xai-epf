"""Refà el que es deriva d'una execució ja feta: les mètriques i les figures.

Tot el que hi ha al directori d'una execució llevat de les prediccions i l'atribució
surt de les prediccions i del preu, de manera que adoptar una mètrica nova o canviar
una figura no ha de costar cap camí endavant. Les figures de SHAP surten del `.npz` i
no demanen cap ajust, així que també es refan aquí.

Viu a l'arrel del paquet i no a `results/` perquè toca les dues meitats d'una
execució, la que puntua i la que explica, i un subpaquet no coneix el nom del seu
germà (`architecture.md` 8.3).
"""
import pandas as pd

from epf import matrix
from epf.attribution import figures as shap_figures
from epf.attribution import stored
from epf.data import price
from epf.evaluation import PERCENTILES, distribution_metrics, metrics
from epf.models.families import families
from epf.models.naive import naive
from epf.results import combination, figures, run_files


def distribution_rows(folder, actual):
    """Les mètriques que només té una família que prediu una distribució, o `None`.

    **Còpia deliberada de `runner._distribution_rows`**, i les dues es mantenen
    iguals. Compartir-les voldria dir que aquest mòdul importés el corredor per
    arribar-hi, que és la dependència de doble sentit que es va treure el 2026-10-07
    quan la combinació va sortir d'allà (`architecture.md` 8.4).

    Sense això una reconstrucció perdia el CRPS en silenci: `metrics` no el calcula,
    perquè no surt de les prediccions sinó dels paràmetres que hi ha al costat.
    """
    record = run_files.load_record(folder)
    family = families().get(record.get("family"))
    if family is None or family.distribution is None:
        return None
    days, parameters, _ = run_files.load_distribution(folder, record["family"])
    quantiles = family.distribution.quantiles(parameters, PERCENTILES)
    return distribution_metrics(record["family"], actual.loc[days], quantiles)


def rebuild(folder, actual):
    """Reescriu `metrics.csv` i torna a dibuixar totes les figures d'una família."""
    forecasts = run_files.load_forecasts(folder)
    table = metrics(forecasts, actual, naive.weekly_frame(actual))
    extra = distribution_rows(folder, actual)
    if extra is not None:
        table = pd.concat([table, extra], ignore_index=True)
    run_files.save_metrics(folder, table)
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
