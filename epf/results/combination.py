"""La combinació d'una execució: la mitjana dels seus models ajustats.

Seguint Lago et al. (2021), que conclou que combinar models millora la precisió i que
la mitjana aritmètica ja ho fa. Tot el que hi ha aquí llegeix el que una execució ha
deixat al disc, de manera que viu amb la resta del que llegeix un directori d'execució
i no amb el que l'escriu.

La combinació és de tota la configuració i de cap família, així que té el seu propi
directori al costat d'elles, amb el seu `forecasts_combination.csv`, el seu
`metrics.csv` i el seu `run.json`. Per això el `metrics.csv` d'una família no depèn de
les seves germanes i s'escriu un sol cop, i per això una figura de la combinació es pot
tornar al registre que diu quins membres la formaven.
"""
import shutil

from epf.evaluation import metrics
from epf.models.naive import naive
from epf.results import figures, run_files


def members(folder):
    """Els models ajustats que hi ha desats en una família, pel nom qualificat de cada un.

    Les finestres d'un conjunt com LEAR compten una cada una, perquè el que l'article
    compta per decidir la mida d'una combinació són models ajustats i no famílies; la
    mitjana de les finestres no hi entra, que no és cap model. Qui ho diu és el
    `run.json`, i `None` vol dir un registre d'abans que ho digués.

    El nom va qualificat per la família, `lear/lear_w56`, perquè la llista acaba al
    registre de la combinació, que és d'una configuració sencera i on `lear_w56` tot
    sol no diria d'on surt.
    """
    record = run_files.load_record(folder)
    if "members" not in record:
        return None
    forecasts = run_files.load_forecasts(folder)
    return {f"{folder.name}/{name}": forecasts[name] for name in record["members"]}


def combined(folders):
    """La mitjana aritmètica de tots els membres ajustats d'una configuració.

    Mitjana simple, que és el que fa l'article, i sobre els dies que tots
    comparteixen. Amb menys de dos membres no hi ha res a combinar, i amb un registre
    que no diu qui és membre tampoc: amb una barreja de registres la combinació
    sortiria d'un subconjunt i no ho diria enlloc.

    El directori de la combinació se salta aquí, i no a qui crida, de manera que cap
    dels seus lectors no se n'hagi de recordar (`before-writing.md` 6.4): mitjanar-la
    amb els seus propis membres la comptaria dues vegades.
    """
    found = {}
    for folder in folders:
        if run_files.is_combination(folder):
            continue
        theirs = members(folder)
        if theirs is None:
            return None, {}
        found.update(theirs)
    if len(found) < 2:
        return None, found
    days = next(iter(found.values())).index
    for frame in found.values():
        days = days.intersection(frame.index)
    if days.empty:
        raise ValueError("els membres de la combinació no comparteixen cap dia")
    return sum(frame.loc[days] for frame in found.values()) / len(found), found


def save(where, folders, actual):
    """Escriu i dibuixa el directori de la combinació d'una configuració.

    Segona passada, en acabar la configuració: la combinació necessita totes les
    famílies i la primera ja tenia el seu directori escrit quan la segona començava.
    El directori se substitueix sencer, perquè els seus membres poden haver canviat.

    `commit`, `tier` i `stage` surten del registre d'una família qualsevol: totes les
    que hi ha sota un mateix directori de configuració els tenen iguals, que és el que
    vol dir que el commit nomeni el directori.

    Torna quants membres ha mitjanat, i 0 quan no n'hi havia prou per combinar res.
    """
    frame, found = combined(folders)
    if frame is None:
        return 0

    name = run_files.COMBINATION
    source = run_files.load_record(sorted(folder for folder in folders
                                          if not run_files.is_combination(folder))[0])
    folder = where / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    run_files.save_forecasts(folder, {name: frame})
    run_files.save_metrics(folder, metrics({name: frame}, actual,
                                           naive.weekly_frame(actual)))
    run_files.save_record(folder, {
        "commit": source["commit"],
        "stage": source["stage"],
        "tier": source["tier"],
        # Qui s'ha mitjanat, que és el que defineix una combinació: sense això la seva
        # xifra no es pot tornar a res. No hi ha `family`: no n'és cap, i és així com
        # la passada d'atribució sap que aquest directori no s'explica.
        "members": sorted(found),
        "headlines": [name],
        "forecasts": [name],
        "days": len(frame),
    })
    figures.draw(folder, actual)
    return len(found)


def comparison(folders):
    """Els pronòstics que es comparen en una configuració: el de capçalera de cada
    directori, amb el de la combinació, que té el seu.

    No tots els que hi ha al disc: les quatre finestres de LEAR tenen el seu CSV i les
    seves mètriques, però el que es compara amb els altres models és la seva mitjana.
    Un registre que no diu quins són de capçalera dóna tots els seus, que és el que
    feia abans que ho digués.
    """
    forecasts = {}
    for folder in folders:
        theirs = run_files.load_forecasts(folder)
        forecasts.update({name: theirs[name]
                          for name in run_files.headlines(folder, theirs)})
    return forecasts
