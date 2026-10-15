"""Les figures d'una passada, les de Trebbien et al. (2023) adaptades, dibuixades del que
la passada va desar, sense tornar a ajustar res.

Són un joc de treball: quines es queden per a la memòria es decideix quan tot el codi
sigui fet. A la carpeta de cada família, les seves, que dibuixa el `figures.py` del seu
model; a dalt de la passada, les que comparen les famílies: com puntua cada una al costat
dels bancs de comparació, i els residus de cada família ajustada.
"""
import importlib

from epf import paths
from epf.contract import figure
from epf.contract.figure import plt
from epf.run import residuals

# Cada família amb el `figures.py` del seu model, o `None` si no en té cap. Hi són totes,
# de manera que una carpeta d'una família que no hi és fa petar el dibuix en lloc de
# quedar-se sense figures sense dir-ho.
FIGURES = {
    "lightgbm": "epf.models.lightgbm.figures",
    "date_probe": "epf.models.lightgbm.figures",
    "elastic_net": "epf.models.elastic_net.figures",
    "ddnn_normal": "epf.models.nn.ddnn.figures",
    "ddnn_jsu": "epf.models.nn.ddnn.figures",
    "dnn": None,
    "floor": "epf.models.floor.figures",
    "naive_previous_day": None,
    "naive_previous_week": None,
    "naive_similar_day": None,
}


def _performance(where, described):
    """La R-quadrat i el MAE de cada família sobre les setmanes de prova, la millor
    configuració de cada llavor amb una marca, i la resta de les seves deu millors amb una
    de petita i grisa al darrere, al costat dels bancs de comparació. Dos gràfics i no dos
    eixos.

    Només les famílies explicades tenen color; els bancs de comparació, que no desen cap
    contribució, van en gris, i el nom de sota diu quin és cada un.
    """
    families = sorted(described)
    explained = [family for family in families if "values" in described[family][0].files]
    colour = {name: hue for hue, name in figure.colours(explained)}
    panels, axes = plt.subplots(1, 2, figsize=(max(8, 1.1 * len(families)), 5),
                                facecolor=figure.SURFACE)
    for axis, metric, label in ((axes[0], "r2", "R-quadrat"), (axes[1], "mae", "MAE, EUR/MWh")):
        for position, family in enumerate(families):
            for best, size in ((False, 2), (True, 4)):
                found = [float(seed[metric]) for seed in described[family]
                         if (int(seed["rank"]) == 0) == best]
                axis.plot([position] * len(found), found, "o", markersize=size,
                          color=colour.get(family, figure.MUTED_INK) if best
                          else figure.MUTED_INK)
        axis.set_xticks(range(len(families)), families, fontsize=6, rotation=90)
        axis.set_ylabel(label, color=figure.MUTED_INK)
        figure.style(axis, f"{label} a les setmanes de prova, la millor de cada llavor")
    figure.save(panels, where, "performance")


def draw(where):
    """Totes les figures d'una passada, a partir del que hi ha desat."""
    described = {}
    for folder in sorted(path for path in where.iterdir() if path.is_dir()):
        seeds = figure.seeds(folder)
        if not seeds:
            continue
        described[folder.name] = seeds
        if FIGURES[folder.name] is not None:
            importlib.import_module(FIGURES[folder.name]).draw(folder, seeds)
    if described:
        _performance(where, described)
        residuals.report(where, described)


def main(arguments):
    # Només el nom d'una passada que hi és, mai un camí: el que s'escriu no tria on.
    found = {path.name: path for smoke in (False, True) for path in paths.runs(smoke)}
    if arguments.run not in found:
        print(f"no hi ha cap passada {arguments.run} a {paths.RUNS}")
        return 1
    draw(found[arguments.run])
    print(f"figures de la passada a {found[arguments.run]}")
    return 0
