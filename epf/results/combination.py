"""La combinació d'una execució: la mitjana dels seus models ajustats.

Seguint Lago et al. (2021), que conclou que combinar models millora la precisió i que
la mitjana aritmètica ja ho fa. Tot el que hi ha aquí llegeix el que una execució ha
deixat al disc, de manera que viu amb la resta del que llegeix un directori d'execució
i no amb el que l'escriu.
"""
import pandas as pd

from epf.evaluation import metrics
from epf.attribution import stored
from epf.models.naive import naive
from epf.results import run_files

# Nom de la mitjana de tots els membres ajustats d'una configuració.
COMBINATION = "combination"


def members(folder):
    """Els pronòstics ajustats que hi ha desats en una família.

    Les finestres d'un conjunt com LEAR compten una cada una, perquè el que
    l'article compta per decidir la mida d'una combinació són models ajustats i no
    famílies; les famílies d'un sol model aporten el seu CSV. Surten del `.npz`
    quan n'hi ha, que és on es desen, i del CSV quan no.
    """
    forecasts = run_files.load_forecasts(folder)
    found = []
    for name, frame in sorted(forecasts.items()):
        if not stored.path(folder, name).exists():
            found.append(frame)
            continue
        days, attribution = stored.load(folder, name)
        if attribution.forecasts is None:
            found.append(frame)
            continue
        index = pd.DatetimeIndex(pd.to_datetime(days), name=frame.index.name)
        found += [pd.DataFrame(attribution.forecasts[:, number], index=index,
                               columns=frame.columns)
                  for number in range(len(attribution.members))]
    return found


def combined(folders):
    """La mitjana aritmètica de tots els membres ajustats d'una configuració.

    Mitjana simple, que és el que fa l'article, i sobre els dies que tots
    comparteixen. Amb menys de dos membres no hi ha res a combinar.
    """
    found = [frame for folder in folders for frame in members(folder)]
    if len(found) < 2:
        return None, found
    days = found[0].index
    for frame in found[1:]:
        days = days.intersection(frame.index)
    if days.empty:
        raise ValueError("els membres de la combinació no comparteixen cap dia")
    return sum(frame.loc[days] for frame in found) / len(found), found


def _fitted(folders):
    """Les famílies que ajusten un model, segons el que digui el seu registre.

    Un registre d'abans que existís `fits` no ho diu, i aleshores no se'n combina
    cap: amb una barreja de registres la combinació sortiria d'un subconjunt i no
    ho diria enlloc.
    """
    records = [run_files.load_record(folder) for folder in folders]
    if any("fits" not in record for record in records):
        return []
    return [folder for folder, record in zip(folders, records) if record["fits"]]


def add_combination(folders, actual):
    """Escriu les mètriques de la combinació a `metrics.csv` de cada família.

    Segona passada, en acabar la configuració: la combinació necessita totes les
    famílies i la primera ja tenia el seu fitxer escrit quan la segona començava.
    El preu que es paga és que el `metrics.csv` d'una família depèn de les seves
    germanes i ja no s'escriu un sol cop.
    """
    frame, found = combined(_fitted(folders))
    if frame is None:
        return 0
    rows = metrics({COMBINATION: frame}, actual, naive.weekly_frame(actual))
    for folder in folders:
        existing = run_files.load_metrics(folder)
        run_files.save_metrics(folder, pd.concat(
            [existing[existing["forecast"] != COMBINATION], rows], ignore_index=True))
    return len(found)


def comparison(folders, actual):
    """Els pronòstics que es comparen en una configuració: els de totes les
    famílies i, si n'hi ha, la combinació."""
    forecasts = {}
    for folder in folders:
        forecasts.update(run_files.load_forecasts(folder))
    frame, _ = combined(_fitted(folders))
    if frame is not None:
        forecasts[COMBINATION] = frame
    return forecasts
