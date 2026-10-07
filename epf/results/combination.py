"""La combinació d'una execució: la mitjana dels seus models ajustats.

Seguint Lago et al. (2021), que conclou que combinar models millora la precisió i que
la mitjana aritmètica ja ho fa. Tot el que hi ha aquí llegeix el que una execució ha
deixat al disc, de manera que viu amb la resta del que llegeix un directori d'execució
i no amb el que l'escriu.
"""
import pandas as pd

from epf.evaluation import metrics
from epf.models.naive import naive
from epf.results import run_files

# Nom de la mitjana de tots els membres ajustats d'una configuració.
COMBINATION = "combination"


def members(folder):
    """Els models ajustats que hi ha desats en una família, un CSV cada un.

    Les finestres d'un conjunt com LEAR compten una cada una, perquè el que l'article
    compta per decidir la mida d'una combinació són models ajustats i no famílies; la
    mitjana de les finestres no hi entra, que no és cap model. Qui ho diu és el
    `run.json`, i `None` vol dir un registre d'abans que ho digués.
    """
    record = run_files.load_record(folder)
    if "members" not in record:
        return None
    forecasts = run_files.load_forecasts(folder)
    return [forecasts[name] for name in record["members"]]


def combined(folders):
    """La mitjana aritmètica de tots els membres ajustats d'una configuració.

    Mitjana simple, que és el que fa l'article, i sobre els dies que tots
    comparteixen. Amb menys de dos membres no hi ha res a combinar, i amb un registre
    que no diu qui és membre tampoc: amb una barreja de registres la combinació
    sortiria d'un subconjunt i no ho diria enlloc.
    """
    found = []
    for folder in folders:
        theirs = members(folder)
        if theirs is None:
            return None, []
        found += theirs
    if len(found) < 2:
        return None, found
    days = found[0].index
    for frame in found[1:]:
        days = days.intersection(frame.index)
    if days.empty:
        raise ValueError("els membres de la combinació no comparteixen cap dia")
    return sum(frame.loc[days] for frame in found) / len(found), found


def add_combination(folders, actual):
    """Escriu les mètriques de la combinació a `metrics.csv` de cada família.

    Segona passada, en acabar la configuració: la combinació necessita totes les
    famílies i la primera ja tenia el seu fitxer escrit quan la segona començava.
    El preu que es paga és que el `metrics.csv` d'una família depèn de les seves
    germanes i ja no s'escriu un sol cop.
    """
    frame, found = combined(folders)
    if frame is None:
        return 0
    rows = metrics({COMBINATION: frame}, actual, naive.weekly_frame(actual))
    for folder in folders:
        existing = run_files.load_metrics(folder)
        run_files.save_metrics(folder, pd.concat(
            [existing[existing["forecast"] != COMBINATION], rows], ignore_index=True))
    return len(found)


def comparison(folders, actual):
    """Els pronòstics que es comparen en una configuració: el de capçalera de cada
    família i, si n'hi ha, la combinació.

    No tots els que hi ha al disc: les cinc finestres de LEAR tenen el seu CSV i les
    seves mètriques, però el que es compara amb els altres models és la seva mitjana.
    Un registre que no diu quins són de capçalera dóna tots els seus, que és el que
    feia abans que ho digués.
    """
    forecasts = {}
    for folder in folders:
        theirs = run_files.load_forecasts(folder)
        forecasts.update({name: theirs[name]
                          for name in run_files.headlines(folder, theirs)})
    frame, _ = combined(folders)
    if frame is not None:
        forecasts[COMBINATION] = frame
    return forecasts
