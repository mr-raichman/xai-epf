"""Les figures d'una passada, les de Trebbien et al. (2023) adaptades, dibuixades del que
la passada va desar, sense tornar a ajustar res.

Són un joc de treball: quines es queden per a la memòria es decideix quan tot el codi
sigui fet. A dalt de la passada, el sòl de l'ordre de mèrit, com puntua cada família al
costat dels bancs de comparació i els residus de cada família ajustada; a la carpeta de cada conjunt i família, la importància per columna i per
grup, la dependència de les tres previsions i, a l'arbre, les seves interaccions.
"""
import numpy as np

from epf.contract import figure
from epf.contract.figure import plt
from epf.results import residuals

# Socis de cada previsió a les interaccions: els tres que més hi interactuen, la regla
# que fa de les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3


def _interactions(folder, seed, present):
    """Al millor model de la primera llavor, per a cada sèrie: la dependència sencera,
    l'efecte sense interaccions i la interacció amb els tres socis més forts, amb el
    color del valor de cada soci."""
    columns = list(seed["columns"])
    rows = seed["interaction_rows"]
    inputs, interactions = seed["inputs"][rows], seed["interactions"]
    values = seed["values"][rows]
    panels, axes = plt.subplots(len(present), 2 + PARTNERS,
                                figsize=(4 * (2 + PARTNERS), 3.5 * len(present)),
                                facecolor=figure.SURFACE, squeeze=False)
    for row, name in zip(axes, present):
        (j,) = figure.members(seed, name)
        x = inputs[:, j] / figure.PER_GW
        strength = np.abs(interactions[:, j, :]).mean(axis=0)
        strength[j] = -np.inf
        partners = np.argsort(strength)[-PARTNERS:][::-1]
        panels_of_row = [(values[:, j], None, "sencera"),
                         (interactions[:, j, j], None, "sense interaccions")]
        panels_of_row += [(2 * interactions[:, j, k], inputs[:, k], f"amb {columns[k]}")
                          for k in partners]
        for axis, (y, colour, title) in zip(row, panels_of_row):
            if colour is None:
                axis.plot(x, y, ".", color=figure.SERIES, markersize=2, alpha=0.5)
            else:
                image = axis.scatter(x, y, c=colour, cmap=figure.MAGNITUDE, s=3)
                figure.colorbar(panels, image, axis, "valor del soci")
            axis.set_xlabel(f"{figure.GROUP_LABELS[name]}, GW", color=figure.MUTED_INK)
            figure.style(axis, title)
    figure.save(panels, folder, "interactions")


def _performance(where, described):
    """La R-quadrat i el MAE de cada família i conjunt sobre les setmanes de prova, la
    millor configuració de cada llavor amb una marca, i la resta de les seves deu millors
    amb una de petita i grisa al darrere, al costat dels bancs de comparació. Dos gràfics
    i no dos eixos.

    Només les famílies explicades tenen color; els bancs de comparació, que no desen cap
    contribució, van en gris, i el nom de sota diu quin és cada un.
    """
    keys = sorted(described)
    explained = list(dict.fromkeys(key[1] for key in keys
                                   if "values" in described[key][0].files))
    colour = {name: hue for hue, name in figure.colours(explained)}
    panels, axes = plt.subplots(1, 2, figsize=(max(8, 1.1 * len(keys)), 5),
                                facecolor=figure.SURFACE)
    for axis, metric, label in ((axes[0], "r2", "R-quadrat"), (axes[1], "mae", "MAE, EUR/MWh")):
        for position, key in enumerate(keys):
            for best, size in ((False, 2), (True, 4)):
                found = [float(seed[metric]) for seed in described[key]
                         if (int(seed["rank"]) == 0) == best]
                axis.plot([position] * len(found), found, "o", markersize=size,
                          color=colour.get(key[1], figure.MUTED_INK) if best
                          else figure.MUTED_INK)
        axis.set_xticks(range(len(keys)), [f"{name}\n{family}" for name, family in keys],
                        fontsize=6, rotation=90)
        axis.set_ylabel(label, color=figure.MUTED_INK)
        figure.style(axis, f"{label} a les setmanes de prova, la millor de cada llavor")
    figure.save(panels, where, "performance")


def _floor(where, seeds):
    """El preu contra la càrrega residual a les files d'entrenament de la llavor 0, en
    densitat, i el polinomi cúbic de cada llavor, la figura del sòl de l'article."""
    first = seeds[0]
    load = first["load_training"] / figure.PER_GW
    panels, axis = plt.subplots(figsize=(8, 5), facecolor=figure.SURFACE)
    image = axis.hist2d(load, first["actual_training"], bins=80, cmap=figure.MAGNITUDE)[3]
    figure.colorbar(panels, image, axis, "hores")
    span = np.linspace(load.min(), load.max(), 200)
    for seed in seeds:
        axis.plot(span, np.polyval(seed["coefficients"], span * figure.PER_GW),
                  color=figure.INK, linewidth=0.6)
    axis.set_xlabel("càrrega residual prevista, GW", color=figure.MUTED_INK)
    axis.set_ylabel("preu, EUR/MWh", color=figure.MUTED_INK)
    figure.style(axis, "El sòl de l'ordre de mèrit: les files d'entrenament de la "
                       f"primera llavor i el cúbic de cada una, llavors: {len(seeds)}")
    figure.save(panels, where, "floor")


def draw(where):
    """Totes les figures d'una passada, a partir del que hi ha desat."""
    described = {}
    for folder in sorted(path for path in where.glob("*/*") if path.is_dir()):
        seeds = figure.seeds(folder)
        if not seeds:
            continue
        described[(folder.parent.name, folder.name)] = seeds
        if "coefficients" in seeds[0].files:
            _floor(where, seeds)
        # Els bancs de comparació no s'expliquen.
        if "values" not in seeds[0].files:
            continue
        figure.importance(folder, seeds)
        names = set(seeds[0]["groups"])
        present = [name for name in figure.DEPENDENCE if name in names]
        if present:
            figure.dependence(folder, seeds, present)
            if "interactions" in seeds[0].files:
                _interactions(folder, seeds[0], present)
    if described:
        _performance(where, described)
        residuals.report(where, described)


def main(arguments):
    for folder in arguments.folder:
        draw(folder)
        print(f"figures de la passada a {folder}")
