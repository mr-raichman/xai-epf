"""Com es fa i com es veu una figura d'aquest projecte, en un sol lloc, i el que
comparteixen les figures dels models explicats.

El llegeix tot el que dibuixa: les figures de les dades d'origen, les de cada model i les
que comparen les famílies. Només hi ha el que fan servir dues coses o més: el que és d'una
figura concreta, com el vermell dels valors marcats, viu amb ella. Les dels models
explicats es dibuixen del que la passada va desar, un fitxer per model: la importància de
cada columna i de cada grup, i la dependència de cada sèrie.

El motor és `Agg`, sense finestra, i es tria aquí: qui importa aquest mòdul ja el té
triat i no ha de recordar-ho abans d'importar `pyplot`.
"""
import matplotlib
import numpy as np

from epf.contract.matrix import INFORMATION, PRICE
from epf.parameters import HOUR_COLUMNS
from epf.paths import SEED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Tinta del text, tinta apagada de les etiquetes, línies de reixa i fons.
INK = "#0b0b0b"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# El blau de la sèrie principal, que també encapçala la seqüència de línies.
SERIES = "#2a78d6"

DPI = 300

# Escala seqüencial d'un sol to: la magnitud es llegeix igual en color i en escala de
# grisos impresa.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", [SURFACE, "#c3d9f2", "#6fa4e0", "#2a78d6", "#143f73"])

# Seqüència de colors de línia, encapçalada pel blau de la sèrie. Una línia per color,
# perquè dues del mateix color en una llegenda no es distingeixen.
LINES = (SERIES, "#d6622a", "#2ba86a", "#9b4fd6", "#c0392b", "#117a8b", "#b5338f")


def colours(names):
    """Un color per nom, en l'ordre dels noms.

    Peta si n'hi ha més que colors. `zip(LINES, ...)` els deixaria fora sense dir-ho,
    i una figura que perd un model en silenci és pitjor que una figura que no es
    dibuixa. Repetir-ne un no val: dues línies del mateix color en una llegenda no es
    distingeixen.
    """
    names = list(names)
    if len(names) > len(LINES):
        raise ValueError(f"{len(names)} sèries i només {len(LINES)} colors de línia: "
                         f"{names}")
    return list(zip(LINES, names))


def style(axis, title=None):
    """El títol, les marques i el marc d'uns eixos, iguals a tot el projecte."""
    if title:
        axis.set_title(title, color=INK, loc="left", fontsize=10)
    axis.tick_params(colors=MUTED_INK, length=0)
    for side in axis.spines.values():
        side.set_color(GRID)


def colorbar(figure, image, axis, label):
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    bar.set_label(label, color=MUTED_INK)
    bar.ax.tick_params(colors=MUTED_INK, length=0)
    bar.outline.set_color(GRID)


def save(figure, folder, name):
    figure.tight_layout()
    figure.savefig(folder / f"{name}.png", dpi=DPI)
    plt.close(figure)


# Com es diu cada grup de columnes a les figures.
GROUP_LABELS = {"price": "preu del dia abans", "demand_fcst": "demanda prevista",
                "wind_fcst": "eòlica prevista", "solar_fcst": "fotovoltaica prevista",
                "nuclear_sched": "nuclear programada",
                "coal_gas_sched": "carbó i gas programats",
                "hydro_sched": "hidràulica programada", "calendar": "calendari",
                "date": "comptador de dies"}

# Com es diu cada paràmetre explicat d'una xarxa distribucional, i la contribució de
# les famílies que n'expliquen un de sol.
OUTPUT_LABELS = {"": "contribució", "location": "localització", "scale": "escala",
                 "skewness": "asimetria", "tail_weight": "pes de la cua"}

# Les sèries que tenen gràfic de dependència: totes les del conjunt d'informació menys el
# preu, les previsions com a Trebbien et al. (2023) i el programa igual.
DEPENDENCE = tuple(name for name in INFORMATION if name != PRICE)

# Columnes que mostra la importància per columna; la resta compten al seu grup.
SHOWN = 20

# MW per GW, perquè els pendents es llegeixen per GW com a l'article.
PER_GW = 1000


def seeds(folder):
    """Els models desats d'una família, un per fitxer, en ordre de llavor i de
    configuració: el primer és la millor configuració de la primera llavor."""
    paths = sorted(folder.glob(SEED.format(seed="*", rank="*")),
                   key=lambda path: tuple(int(part) for part in path.stem.split("_")[1:]))
    return [np.load(path) for path in paths]


def outputs(seed):
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


def importance(folder, seeds):
    """La importància de les columnes que més pesen i la dels grups, amb cada llavor."""
    explained = list(outputs(seeds[0]))
    groups = seeds[0]["groups"]
    columns = seeds[0]["columns"]
    for kind in ("columns", "groups"):
        panels, axes = plt.subplots(1, len(explained), figsize=(5 * len(explained), 6),
                                    facecolor=SURFACE, squeeze=False)
        for axis, output in zip(axes[0], explained):
            found = [shares(outputs(seed)[output], groups) for seed in seeds]
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
            axis.barh(positions, table[:, order].mean(axis=0), color=SERIES, height=0.6)
            for row in table[:, order]:
                axis.plot(row, positions, "o", color=INK, markersize=2)
            axis.set_yticks(positions, names, fontsize=7)
            axis.set_xlabel("part de la importància", color=MUTED_INK)
            style(axis, f"{OUTPUT_LABELS[output]}, models: {len(seeds)}")
        save(panels, folder, "importance" if kind == "columns" else "groups")


def present(seed):
    """Les sèries amb gràfic de dependència que porta el fitxer d'un model, en l'ordre de
    `DEPENDENCE`."""
    names = set(seed["groups"])
    return [name for name in DEPENDENCE if name in names]


def members(seed, name):
    """Les columnes de la sèrie `name`, pel grup que la passada hi va desar: una a la
    matriu per hora, la de l'hora de la fila, i les 24 de la diària en ordre d'hora."""
    members = np.flatnonzero(seed["groups"] == name)
    if len(members) == 1:
        return members
    hour = {column: number for number, column in enumerate(HOUR_COLUMNS)}
    return np.array(sorted(members,
                           key=lambda j: hour[str(seed["columns"][j]).rsplit("_", 1)[1]]))


def pairs(seed, output, values, name):
    """El valor de la sèrie i la seva contribució, a l'hora de cada fila."""
    inputs = seed["inputs"]
    found = members(seed, name)
    if len(found) == 1:
        return inputs[:, found[0]], values[:, found[0]]
    # La xarxa distribucional: a la fila del dia d i l'hora h, la columna de l'hora h.
    x = inputs[:, found]
    y = values.reshape(len(inputs), len(HOUR_COLUMNS), -1)[:, np.arange(len(found)),
                                                           found]
    return x.ravel(), y.ravel()


def dependence(folder, seeds, present):
    """La contribució de cada previsió contra el seu valor, al millor model de la primera
    llavor, amb la recta ajustada i el pendent de tots els models, que és el que
    Trebbien et al. (2023) llegeixen."""
    explained = list(outputs(seeds[0]))
    units = dict(zip(explained, seeds[0]["units"]))
    panels, axes = plt.subplots(len(present), len(explained),
                                figsize=(4.5 * len(explained), 3.5 * len(present)),
                                facecolor=SURFACE, squeeze=False)
    for row, name in zip(axes, present):
        for axis, output in zip(row, explained):
            slopes = []
            for number, seed in enumerate(seeds):
                x, y = pairs(seed, output, outputs(seed)[output], name)
                slope, intercept = np.polyfit(x / PER_GW, y, 1)
                slopes.append(slope)
                if number == 0:
                    axis.plot(x / PER_GW, y, ".", color=SERIES, markersize=1,
                              alpha=0.3)
                    span = np.array([x.min(), x.max()]) / PER_GW
                    axis.plot(span, slope * span + intercept, color=INK, linewidth=1)
            axis.set_xlabel(f"{GROUP_LABELS[name]}, GW", color=MUTED_INK)
            axis.set_ylabel(f"{OUTPUT_LABELS[output]}, {units[output]}",
                            color=MUTED_INK)
            style(axis, f"pendent {np.mean(slopes):.3g} (±{np.std(slopes):.2g}), "
                        f"{units[output]} per GW")
    save(panels, folder, "dependence")
