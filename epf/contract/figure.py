"""L'aspecte de les figures, i les que dibuixen igual tots els models explicats.

Les dels models explicats es dibuixen dels models ajustats que la passada va desar.
"""

import matplotlib
import numpy as np

from epf.contract.matrix import INFORMATION, PRICE
from epf.parameters import HOUR_COLUMNS
from epf.paths import SEED

# El motor sense finestra, triat abans d'importar `pyplot`: qui importa aquest mòdul ja
# el té triat.
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

# Escala seqüencial d'un sol to.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", [SURFACE, "#c3d9f2", "#6fa4e0", "#2a78d6", "#143f73"]
)

# Seqüència de colors de línia, encapçalada pel blau de la sèrie.
LINES = (SERIES, "#d6622a", "#2ba86a", "#9b4fd6", "#c0392b", "#117a8b", "#b5338f")


def colours(names):
    """Torna un color per nom, en l'ordre dels noms.

    Falla si n'hi ha més que colors: `zip(LINES, ...)` els deixaria fora sense dir-ho, i
    repetir-ne un faria dues línies que la llegenda no distingeix.
    """
    names = list(names)
    if len(names) > len(LINES):
        raise ValueError(
            f"{len(names)} sèries i només {len(LINES)} colors de línia: "
            f"{', '.join(map(str, names))}"
        )
    return list(zip(LINES, names, strict=False))


def style(axis, title=None):
    """Dona a uns eixos el títol, les marques i el marc de tot el projecte."""
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
    """Desa una figura a `folder/<name>.png` i la tanca."""
    figure.tight_layout()
    figure.savefig(folder / f"{name}.png", dpi=DPI)
    plt.close(figure)


# Com es diu cada grup de columnes a les figures.
GROUP_LABELS = {
    "price": "preu del dia abans",
    "demand_fcst": "demanda prevista",
    "wind_fcst": "eòlica prevista",
    "solar_fcst": "fotovoltaica prevista",
    "nuclear_sched": "nuclear programada",
    "coal_gas_sched": "carbó i gas programats",
    "hydro_sched": "hidràulica programada",
    "calendar": "calendari",
    "date": "comptador de dies",
}

# Com es diu cada paràmetre explicat d'una xarxa distribucional, i la contribució de les
# famílies que n'expliquen un de sol.
OUTPUT_LABELS = {
    "": "contribució",
    "location": "localització",
    "scale": "escala",
    "skewness": "asimetria",
    "tail_weight": "pes de la cua",
}

# Les sèries que tenen gràfic de dependència: totes les de l'information set menys el
# preu, les previsions com a Trebbien et al. (2023) i el programa igual.
DEPENDENCE = tuple(name for name in INFORMATION if name != PRICE)

# Columnes que mostra la importància per columna; la resta compten al seu grup.
SHOWN = 20

# MW per GW, perquè els pendents es llegeixen per GW, com fan Trebbien et al. (2023).
PER_GW = 1000


def saved(folder):
    """Torna els models ajustats d'una família, en ordre de seed i de configuració.

    El primer és la millor configuració de la primera seed.
    """
    paths = sorted(
        folder.glob(SEED.format(seed="*", rank="*")),
        key=lambda path: tuple(int(part) for part in path.stem.split("_")[1:]),
    )
    return [np.load(path) for path in paths]


def outputs(model):
    """Torna les contribucions de cada sortida explicada, `(files, columnes)`, pel nom.

    La xarxa distribucional en té una per paràmetre, amb una fila per dia i hora, i
    l'arbre i l'elastic net, una per fila.
    """
    values = model["values"]
    if values.ndim == 2:
        return {"": values}
    return {
        str(name): values[..., number].transpose(0, 2, 1).reshape(-1, values.shape[1])
        for number, name in enumerate(model["names"])
    }


def shares(values, groups):
    """Torna la importància de cada columna i grup, la de Trebbien et al. (2023).

    És la mitjana del valor absolut, normalitzada a u, i la d'un grup suma les
    contribucions de les seves columnes abans de prendre'l.
    """
    by_column = np.abs(values).mean(axis=0)
    names = list(dict.fromkeys(groups))
    by_group = np.array(
        [np.abs(values[:, groups == name].sum(axis=1)).mean() for name in names]
    )
    return by_column / by_column.sum(), dict(
        zip(names, by_group / by_group.sum(), strict=True)
    )


def importance(folder, models):
    """Dibuixa la importància de les columnes i dels grups, amb cada model ajustat.

    De les columnes, les `SHOWN` que més pesen.
    """
    explained = list(outputs(models[0]))
    groups = models[0]["groups"]
    columns = models[0]["columns"]
    for kind in ("columns", "groups"):
        panels, axes = plt.subplots(
            1,
            len(explained),
            figsize=(5 * len(explained), 6),
            facecolor=SURFACE,
            squeeze=False,
        )
        for axis, output in zip(axes[0], explained, strict=True):
            found = [shares(outputs(model)[output], groups) for model in models]
            if kind == "columns":
                table = np.array([by_column for by_column, _ in found])
                order = np.argsort(table.mean(axis=0))[-SHOWN:]
                names = columns[order]
            else:
                labels = list(found[0][1])
                table = np.array(
                    [[by_group[name] for name in labels] for _, by_group in found]
                )
                order = np.arange(len(labels))
                names = [GROUP_LABELS[name] for name in labels]
            positions = np.arange(len(order))
            axis.barh(positions, table[:, order].mean(axis=0), color=SERIES, height=0.6)
            for row in table[:, order]:
                axis.plot(row, positions, "o", color=INK, markersize=2)
            axis.set_yticks(positions, names, fontsize=7)
            axis.set_xlabel("part de la importància", color=MUTED_INK)
            style(axis, f"{OUTPUT_LABELS[output]}, models: {len(models)}")
        save(panels, folder, "importance" if kind == "columns" else "groups")


def present(model):
    """Torna les sèries de `DEPENDENCE` que porta un model ajustat."""
    names = set(model["groups"])
    return [name for name in DEPENDENCE if name in names]


def members(model, name):
    """Torna les posicions de les columnes de la sèrie `name` en un model ajustat.

    Les troba pel grup que la passada hi va desar: una a la matriu per hora, la de
    l'hora de la fila, i les 24 de la diària en ordre d'hora.
    """
    members = np.flatnonzero(model["groups"] == name)
    if len(members) == 1:
        return members
    hour = {column: number for number, column in enumerate(HOUR_COLUMNS)}
    return np.array(
        sorted(members, key=lambda j: hour[str(model["columns"][j]).rsplit("_", 1)[1]])
    )


def pairs(model, values, name):
    """Torna el valor de la sèrie `name` i la seva contribució a l'hora de cada fila."""
    inputs = model["inputs"]
    found = members(model, name)
    if len(found) == 1:
        return inputs[:, found[0]], values[:, found[0]]
    # La xarxa distribucional: a la fila del dia d i l'hora h, la columna de l'hora h.
    x = inputs[:, found]
    y = values.reshape(len(inputs), len(HOUR_COLUMNS), -1)[
        :, np.arange(len(found)), found
    ]
    return x.ravel(), y.ravel()


def dependence(folder, models, present):
    """Dibuixa la contribució de cada sèrie de `present` contra el seu valor.

    Al millor model ajustat de la primera seed, amb la recta ajustada, i el pendent de
    tots, que és el que llegeixen Trebbien et al. (2023).
    """
    explained = list(outputs(models[0]))
    units = dict(zip(explained, models[0]["units"], strict=True))
    panels, axes = plt.subplots(
        len(present),
        len(explained),
        figsize=(4.5 * len(explained), 3.5 * len(present)),
        facecolor=SURFACE,
        squeeze=False,
    )
    for row, name in zip(axes, present, strict=True):
        for axis, output in zip(row, explained, strict=True):
            slopes = []
            for number, model in enumerate(models):
                x, y = pairs(model, outputs(model)[output], name)
                slope, intercept = np.polyfit(x / PER_GW, y, 1)
                slopes.append(slope)
                if number == 0:
                    axis.plot(x / PER_GW, y, ".", color=SERIES, markersize=1, alpha=0.3)
                    span = np.array([x.min(), x.max()]) / PER_GW
                    axis.plot(span, slope * span + intercept, color=INK, linewidth=1)
            axis.set_xlabel(f"{GROUP_LABELS[name]}, GW", color=MUTED_INK)
            axis.set_ylabel(
                f"{OUTPUT_LABELS[output]}, {units[output]}", color=MUTED_INK
            )
            style(
                axis,
                f"pendent {np.mean(slopes):.3g} (±{np.std(slopes):.2g}), "
                f"{units[output]} per GW",
            )
    save(panels, folder, "dependence")
