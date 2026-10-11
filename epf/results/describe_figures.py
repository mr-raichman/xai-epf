"""Les figures de la part B, les de Trebbien et al. (2023) adaptades, dibuixades del que la
passada va desar, sense tornar a ajustar res.

Són un joc de treball: quines es queden per a la memòria es decideix quan tot el codi
sigui fet. A dalt del pas, el sòl de l'ordre de mèrit i com puntua cada família al costat
seu; a la carpeta de cada conjunt i família, la importància per columna i per grup, la
dependència de les tres previsions i, a l'arbre, les seves interaccions.
"""
import numpy as np

from epf import figure
from epf.calendar import HOUR_COLUMNS
from epf.figure import plt
from epf.results import run_files

# Com es diu cada grup de columnes a les figures.
GROUP_LABELS = {"previous_day": "dia abans", "demand": "demanda", "wind": "eòlica",
                "solar": "fotovoltaica", "calendar": "calendari",
                "date": "comptador de dies"}

# Com es diu cada paràmetre explicat d'una xarxa distribucional, i la contribució de
# les famílies que n'expliquen un de sol.
OUTPUT_LABELS = {"": "contribució", "location": "localització", "scale": "escala",
                 "skewness": "asimetria", "tail_weight": "pes de la cua"}

# Les previsions que tenen gràfic de dependència, les tres de Trebbien et al. (2023).
DEPENDENCE = ("demand", "wind", "solar")

# Columnes que mostra la importància per columna; la resta compten al seu grup.
SHOWN = 20

# Socis de cada previsió a les interaccions: els tres que més hi interactuen, la regla
# que fa de les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3

# MW per GW, perquè els pendents es llegeixen per GW com a l'article.
PER_GW = 1000


def _seeds(folder):
    """Les llavors desades d'una família, en ordre de llavor."""
    paths = sorted(folder.glob(run_files.SEED.format(seed="*")),
                   key=lambda path: int(path.stem.split("_")[1]))
    return [np.load(path) for path in paths]


def _outputs(seed):
    """Les contribucions de cada sortida explicada, `(files, columnes)`, i les entrades
    que hi corresponen, per nom de sortida.

    L'arbre i la xarxa elàstica en tenen una per fila. La xarxa distribucional en té una
    per paràmetre, amb una fila per dia i hora, on la columna d'una previsió a l'hora de
    la fila és la que la parella a l'hora.
    """
    values = seed["values"]
    if values.ndim == 2:
        return {"": values}
    return {str(name): values[..., number].transpose(0, 2, 1).reshape(-1, values.shape[1])
            for number, name in enumerate(seed["names"])}


def shares(values, groups):
    """La importància normalitzada a u de cada columna i de cada grup, la de Trebbien et al.
    (2023): la mitjana del valor absolut, i la d'un grup suma les contribucions de les
    seves columnes abans de prendre'l."""
    by_column = np.abs(values).mean(axis=0)
    names = list(dict.fromkeys(groups))
    by_group = np.array([np.abs(values[:, groups == name].sum(axis=1)).mean()
                         for name in names])
    return by_column / by_column.sum(), dict(zip(names, by_group / by_group.sum()))


def _importance(folder, seeds):
    """La importància de les columnes que més pesen i la dels grups, amb cada llavor."""
    outputs = list(_outputs(seeds[0]))
    groups = seeds[0]["groups"]
    columns = seeds[0]["columns"]
    for kind in ("columns", "groups"):
        panels, axes = plt.subplots(1, len(outputs), figsize=(5 * len(outputs), 6),
                                    facecolor=figure.SURFACE, squeeze=False)
        for axis, output in zip(axes[0], outputs):
            found = [shares(_outputs(seed)[output], groups) for seed in seeds]
            if kind == "columns":
                table = np.array([by_column for by_column, _ in found])
                order = np.argsort(table.mean(axis=0))[-SHOWN:]
                names = columns[order]
            else:
                labels = list(found[0][1])
                table = np.array([[by_group[name] for name in labels] for _, by_group in found])
                order = np.arange(len(labels))
                names = [GROUP_LABELS[name] for name in labels]
            positions = np.arange(len(order))
            axis.barh(positions, table[:, order].mean(axis=0), color=figure.SERIES, height=0.6)
            for row in table[:, order]:
                axis.plot(row, positions, "o", color=figure.INK, markersize=2)
            axis.set_yticks(positions, names, fontsize=7)
            axis.set_xlabel("part de la importància", color=figure.MUTED_INK)
            figure.style(axis, f"{OUTPUT_LABELS[output]}, llavors: {len(seeds)}")
        figure.save(panels, folder, "importance" if kind == "columns" else "groups")


def _pairs(seed, output, values, name):
    """El valor de la previsió i la seva contribució, a l'hora de cada fila."""
    columns = list(seed["columns"])
    inputs = seed["inputs"]
    if name in columns:
        return inputs[:, columns.index(name)], values[:, columns.index(name)]
    # La xarxa distribucional: a la fila del dia d i l'hora h, la columna de l'hora h.
    hours = [columns.index(f"{name}_{hour}") for hour in HOUR_COLUMNS]
    x = inputs[:, hours]
    y = values.reshape(len(inputs), len(HOUR_COLUMNS), -1)[:, np.arange(len(hours)), hours]
    return x.ravel(), y.ravel()


def _dependence(folder, seeds, present):
    """La contribució de cada previsió contra el seu valor, a la llavor 0, amb la recta
    ajustada i el pendent de totes les llavors, que és el que Trebbien et al. (2023)
    llegeixen."""
    outputs = list(_outputs(seeds[0]))
    units = dict(zip(outputs, seeds[0]["units"]))
    panels, axes = plt.subplots(len(present), len(outputs),
                                figsize=(4.5 * len(outputs), 3.5 * len(present)),
                                facecolor=figure.SURFACE, squeeze=False)
    for row, name in zip(axes, present):
        for axis, output in zip(row, outputs):
            slopes = []
            for number, seed in enumerate(seeds):
                x, y = _pairs(seed, output, _outputs(seed)[output], name)
                slope, intercept = np.polyfit(x / PER_GW, y, 1)
                slopes.append(slope)
                if number == 0:
                    axis.plot(x / PER_GW, y, ".", color=figure.SERIES, markersize=1,
                              alpha=0.3)
                    span = np.array([x.min(), x.max()]) / PER_GW
                    axis.plot(span, slope * span + intercept, color=figure.INK, linewidth=1)
            axis.set_xlabel(f"{GROUP_LABELS[name]}, GW", color=figure.MUTED_INK)
            axis.set_ylabel(f"{OUTPUT_LABELS[output]}, {units[output]}",
                            color=figure.MUTED_INK)
            figure.style(axis, f"pendent {np.mean(slopes):.3g} (±{np.std(slopes):.2g}), "
                               f"{units[output]} per GW")
    figure.save(panels, folder, "dependence")


def _interactions(folder, seed, present):
    """A la llavor 0, per a cada previsió: la dependència sencera, l'efecte sense
    interaccions i la interacció amb els tres socis més forts, amb el color del valor de
    cada soci."""
    columns = list(seed["columns"])
    rows = seed["interaction_rows"]
    inputs, interactions = seed["inputs"][rows], seed["interactions"]
    values = seed["values"][rows]
    panels, axes = plt.subplots(len(present), 2 + PARTNERS,
                                figsize=(4 * (2 + PARTNERS), 3.5 * len(present)),
                                facecolor=figure.SURFACE, squeeze=False)
    for row, name in zip(axes, present):
        j = columns.index(name)
        x = inputs[:, j] / PER_GW
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
            axis.set_xlabel(f"{GROUP_LABELS[name]}, GW", color=figure.MUTED_INK)
            figure.style(axis, title)
    figure.save(panels, folder, "interactions")


def _performance(where, described):
    """La R-quadrat i el MAE de cada família i conjunt sobre les setmanes de prova, una
    marca per llavor, al costat del sòl de l'ordre de mèrit. Dos gràfics i no dos eixos."""
    keys = sorted(described)
    families = list(dict.fromkeys(family for _, family in keys))
    colour = {name: hue for hue, name in figure.colours(families)}
    panels, axes = plt.subplots(1, 2, figsize=(max(8, 1.1 * len(keys)), 5),
                                facecolor=figure.SURFACE)
    for axis, metric, label in ((axes[0], "r2", "R-quadrat"), (axes[1], "mae", "MAE, EUR/MWh")):
        for position, key in enumerate(keys):
            found = [float(seed[metric]) for seed in described[key]]
            axis.plot([position] * len(found), found, "o", color=colour[key[1]], markersize=4)
        axis.set_xticks(range(len(keys)), [f"{name}\n{family}" for name, family in keys],
                        fontsize=6, rotation=90)
        axis.set_ylabel(label, color=figure.MUTED_INK)
        figure.style(axis, f"{label} a les setmanes de prova, una marca per llavor")
    figure.save(panels, where, "performance")


def _floor(where, seeds):
    """El preu contra la càrrega residual a les files d'entrenament de la llavor 0, en
    densitat, i el polinomi cúbic de cada llavor, la figura del sòl de l'article."""
    first = seeds[0]
    load = first["load_training"] / PER_GW
    panels, axis = plt.subplots(figsize=(8, 5), facecolor=figure.SURFACE)
    image = axis.hist2d(load, first["actual_training"], bins=80, cmap=figure.MAGNITUDE)[3]
    figure.colorbar(panels, image, axis, "hores")
    span = np.linspace(load.min(), load.max(), 200)
    for seed in seeds:
        axis.plot(span, np.polyval(seed["coefficients"], span * PER_GW),
                  color=figure.INK, linewidth=0.6)
    axis.set_xlabel("càrrega residual prevista, GW", color=figure.MUTED_INK)
    axis.set_ylabel("preu, EUR/MWh", color=figure.MUTED_INK)
    figure.style(axis, "El sòl de l'ordre de mèrit: les files d'entrenament de la "
                       f"primera llavor i el cúbic de cada una, llavors: {len(seeds)}")
    figure.save(panels, where, "floor")


def draw(where):
    """Totes les figures d'un pas de la part B, a partir del que hi ha desat."""
    described = {}
    for folder in sorted(path for path in where.glob("*/*") if path.is_dir()):
        seeds = _seeds(folder)
        if not seeds:
            continue
        described[(folder.parent.name, folder.name)] = seeds
        if "coefficients" in seeds[0].files:
            _floor(where, seeds)
            continue
        _importance(folder, seeds)
        names = set(seeds[0]["groups"])
        present = [name for name in DEPENDENCE if name in names]
        if present:
            _dependence(folder, seeds, present)
            if "interactions" in seeds[0].files:
                _interactions(folder, seeds[0], present)
    if described:
        _performance(where, described)


def main(arguments):
    for folder in arguments.folder:
        draw(folder)
        print(f"figures de la part B a {folder}")
