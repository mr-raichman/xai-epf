"""L'aspecte de les figures, i les que dibuixen igual tots els models explicats.

Les dels models explicats es dibuixen dels models ajustats que la passada va desar.
"""

import math

import matplotlib
import numpy as np

from epf.contract.matrix import INFORMATION, PRICE
from epf.parameters import HOUR_COLUMNS
from epf.contract.record import fitted_files

# El motor sense finestra, triat abans d'importar `pyplot`: qui importa aquest mòdul ja
# el té triat.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


def rgb(red, green, blue):
    """Torna el color dels tres canals, de 0 a 255, en els de 0 a 1 de matplotlib."""
    return (red / 255, green / 255, blue / 255)


def rgba(red, green, blue, alpha):
    """Torna el color dels tres canals, de 0 a 255, amb l'opacitat, de 0 a 1."""
    return (*rgb(red, green, blue), alpha)


# La lletra de tot el text de les figures, la de la memòria.
FONT = "Times New Roman"

# Mides del text, en punts.
AXIS_TITLE_SIZE = 7
TICK_LABEL_SIZE = 6
PANEL_HEADING_SIZE = 7
PANEL_LETTER_SIZE = 10
LEGEND_SIZE = 5
COLORBAR_TITLE_SIZE = 7
COLORBAR_TICK_SIZE = 5
ANNOTATION_SIZE = 5

PANEL_LETTER_WEIGHT = "bold"

# `PANEL_HEADING_PLACE`, "left", "center" o "right".
PANEL_HEADING_PLACE = "center"

# On va la lletra d'un panell, en punts des del seu cantó de dalt a l'esquerra.
PANEL_LETTER_OFFSET = (-8, 4)

# L'amplada de tota figura, la de la caixa de text de la memòria, i l'espai al voltant
# de cada panell, en mm.
WIDTH = 152
PANEL_GAP = 1
MM_PER_INCH = 25.4

# La resolució, en punts per polzada, de tot el que es desa com a imatge: un mapa, una
# barra de colors o una capa de més de `DENSE_POINTS` punts.
DPI = 300

TEXT_COLOUR = rgb(0, 0, 0)
BACKGROUND = rgb(255, 255, 255)

# El color de cada model. Les seves famílies en prenen tons, en l'ordre en què el model
# les fa, cada una `SHADE` més a prop del blanc que l'anterior.
MODEL_COLOURS = {
    "ddnn": rgb(106, 66, 140),
    "dnn": rgb(12, 115, 208),
    "lightgbm": rgb(171, 97, 37),
    "elastic_net": rgb(66, 131, 95),
    "floor": rgb(60, 60, 60),
    "naive": rgb(91, 91, 91),
}
SHADE = 0.32

# El color de tot el que no és d'una família, com les sèries del dataset.
OTHER_COLOUR = rgb(163, 86, 130)

# L'escala dels valors continus, pel seu nom a matplotlib.
SCALE = "viridis"

# L'escala dels valors amb signe, el zero al mig, de blau a vermell, interpolada entre
# aquests colors.
DIVERGING_SCALE = LinearSegmentedColormap.from_list(
    "diverging",
    (
        rgb(33, 102, 172),
        rgb(67, 147, 195),
        rgb(146, 197, 222),
        rgb(209, 229, 240),
        rgb(247, 247, 247),
        rgb(253, 219, 199),
        rgb(244, 165, 130),
        rgb(214, 96, 77),
        rgb(178, 24, 43),
    ),
)

# Gruixos i llargades en punts, i opacitats de 0 a 1, que substitueixen la d'un color
# rgba. Una línia de referència és la que no és cap dada, com un ajust, un zero o un
# tall entre períodes.
LINE_WIDTH = 0.75
REFERENCE_WIDTH = 0.5
REFERENCE_COLOUR = rgb(0, 0, 0)
REFERENCE_STYLE = "--"
MARKER_SIZE = 2
POINT_OPACITY = 0.5
BAND_OPACITY = 0.2

# La línia, del color del fons, entre dues barres d'un histograma, que es toquen.
HISTOGRAM_EDGE_WIDTH = 0.3

# L'amplada d'una barra, en fracció del lloc que té.
BAR_WIDTH = 0.7

# `GRID_AXIS`, "x", "y" o "both".
GRID_SHOWN = True
GRID_AXIS = "both"
GRID_COLOUR = rgb(221, 221, 221)
GRID_WIDTH = 0.3
GRID_STYLE = "-"

# Els costats del marc que es dibuixen.
FRAME_SIDES = ("left", "bottom")
FRAME_COLOUR = rgb(0, 0, 0)
FRAME_WIDTH = 0.5

# `TICK_DIRECTION`, "out", "in" o "inout".
TICK_LENGTH = 2
TICK_WIDTH = 0.5
TICK_DIRECTION = "out"

LEGEND_FRAME = False

# La llargada de la barra de colors entre el seu gruix.
COLORBAR_ASPECT = 30

# Les capes de més punts que aquests es desen com a imatge, a `DPI`: com a vector, el
# SVG en guardaria un objecte per punt.
DENSE_POINTS = 5000

# `SVG_TEXT`, "path", el text com a traços, o "none", com a text, que el PDF deixa
# seleccionar i cercar.
SVG_TEXT = "none"

# Els separadors de les xifres, els de la memòria.
DECIMAL_SEPARATOR = ","
THOUSANDS_SEPARATOR = "."

# El que matplotlib pren per defecte, fet de les variables de dalt: qui importa aquest
# mòdul ja el té.
STYLE = {
    "font.family": FONT,
    "text.color": TEXT_COLOUR,
    "axes.labelcolor": TEXT_COLOUR,
    "xtick.color": TEXT_COLOUR,
    "ytick.color": TEXT_COLOUR,
    "axes.labelsize": AXIS_TITLE_SIZE,
    "axes.titlesize": PANEL_HEADING_SIZE,
    "axes.titlelocation": PANEL_HEADING_PLACE,
    "figure.titlesize": PANEL_HEADING_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    "legend.frameon": LEGEND_FRAME,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND,
    "savefig.dpi": DPI,
    "svg.fonttype": SVG_TEXT,
    # Les fórmules, com R² o σ, en la lletra del text.
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.fallback": "stix",
    # Cada xifra sencera, sense un desplaçament comú a l'extrem de l'eix.
    "axes.formatter.useoffset": False,
    "image.cmap": SCALE,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "axes.grid": GRID_SHOWN,
    "axes.grid.axis": GRID_AXIS,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOUR,
    "grid.linewidth": GRID_WIDTH,
    "grid.linestyle": GRID_STYLE,
    "axes.edgecolor": FRAME_COLOUR,
    "axes.linewidth": FRAME_WIDTH,
    **{
        f"axes.spines.{side}": side in FRAME_SIDES
        for side in ("left", "right", "top", "bottom")
    },
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "xtick.direction": TICK_DIRECTION,
    "ytick.direction": TICK_DIRECTION,
    # Els panells alineats entre ells, cosa que `tight_layout` no fa amb una barra de
    # colors.
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": PANEL_GAP / MM_PER_INCH,
    "figure.constrained_layout.w_pad": PANEL_GAP / MM_PER_INCH,
}
plt.rcParams.update(STYLE)

# Com es dibuixa una línia de referència.
REFERENCE_LINE = {
    "color": REFERENCE_COLOUR,
    "linewidth": REFERENCE_WIDTH,
    "linestyle": REFERENCE_STYLE,
}


def size(height):
    """Torna en polzades la mida d'una figura de `WIDTH` mm i `height` mm d'alt."""
    return (WIDTH / MM_PER_INCH, height / MM_PER_INCH)


def shade(model, number):
    """Torna el color de la família `number` d'un model, comptant des de zero.

    La primera pren el color del model, i cada una de les següents és `SHADE` més a prop
    del blanc que l'anterior, amb la mateixa opacitat.
    """
    *channels, alpha = to_rgba(MODEL_COLOURS[model])
    lighter = 1 - (1 - np.array(channels)) * (1 - SHADE) ** number
    return (*lighter, alpha)


def family_colours(models):
    """Torna el color de cada família, pel seu nom.

    `models` dona les famílies de cada model, en ordre. Un model sense color, que només
    n'agrupa d'altres, no hi entra.
    """
    return {
        family: shade(model, number)
        for model, families in models.items()
        if model in MODEL_COLOURS
        for number, family in enumerate(families)
    }


def catalan(text):
    """Torna un nombre escrit amb punt decimal amb els separadors de la memòria.

    Agrupa les xifres enteres de tres en tres i escriu el signe menys, el de les xifres
    dels eixos, també en la mantissa i l'exponent d'un nombre en notació científica.
    Torna igual el que no és un nombre.
    """
    mantissa, marker, exponent = text.partition("e")
    if marker and exponent.lstrip("+-").isdigit():
        return catalan(mantissa) + marker + catalan(exponent)
    negative = text[:1] in ("-", "\N{MINUS SIGN}")
    whole, point, decimals = text[negative:].partition(".")
    if not whole.isdigit() or (point and not decimals.isdigit()):
        return text
    groups = [whole[max(0, end - 3) : end] for end in range(len(whole), 0, -3)]
    return (
        ("\N{MINUS SIGN}" if negative else "")
        + THOUSANDS_SEPARATOR.join(reversed(groups))
        + (DECIMAL_SEPARATOR + decimals if point else "")
    )


class Numbers(ScalarFormatter):
    """Les xifres d'un eix, amb els separadors de la memòria."""

    def __call__(self, value, position=None):
        return catalan(super().__call__(value, position))


def localise(drawing):
    """Posa els separadors de la memòria a les xifres de tots els eixos d'una figura.

    Només als eixos de nombres: un de dates o d'etiquetes ja té el seu format.
    """
    for axes in drawing.axes:
        for axis in (axes.xaxis, axes.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(Numbers())


def colorbar(drawing, image, axes, label):
    """Afegeix la barra de colors d'una imatge al costat dels eixos `axes`."""
    bar = drawing.colorbar(image, ax=axes, aspect=COLORBAR_ASPECT)
    bar.set_label(label, fontsize=COLORBAR_TITLE_SIZE)
    bar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    bar.outline.set_linewidth(FRAME_WIDTH)
    bar.outline.set_edgecolor(FRAME_COLOUR)


def save(drawing, folder, name):
    """Desa una figura a `folder/<name>.svg` i la tanca.

    Abans hi posa els separadors de la memòria a les xifres dels eixos.
    """
    localise(drawing)
    drawing.savefig(folder / f"{name}.svg")
    plt.close(drawing)


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

# Taulers per fila de la importància i de la dependència, i l'alçada de cada un entre la
# seva amplada.
IMPORTANCE_PER_ROW = 2
IMPORTANCE_ASPECT = 6 / 5
DEPENDENCE_PER_ROW = 4
DEPENDENCE_ASPECT = 3.5 / 4.5


def saved(folder):
    """Torna els models ajustats d'una família, en ordre de seed i de configuració.

    El primer és la millor configuració de la primera seed.
    """
    return [np.load(path) for path in fitted_files(folder).values()]


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


def importances(values, groups):
    """Torna la importància de cada columna i la de cada grup.

    És la mitjana del valor absolut de les contribucions. La de les columnes, entre la
    de la més important, com la normalitzen Trebbien et al. (2023); la d'un grup suma
    les contribucions de les seves columnes abans de prendre-la, com Pesenti i
    O'Sullivan (2025), i és la seva part del total, com la donen Pesenti i O'Sullivan
    (2026).
    """
    by_column = np.abs(values).mean(axis=0)
    names = list(dict.fromkeys(groups))
    by_group = np.array(
        [np.abs(values[:, groups == name].sum(axis=1)).mean() for name in names]
    )
    return by_column / by_column.max(), dict(
        zip(names, by_group / by_group.sum(), strict=True)
    )


def importance(folder, models, colour):
    """Dibuixa la importància de les columnes i dels grups, amb cada model ajustat.

    De les columnes, les `SHOWN` que més pesen. La mitjana, en barres de `colour`, el
    color de la família, i cada model ajustat, en un punt. Un model ajustat sense cap
    contribució en una sortida, que no depèn de les entrades, no hi entra, i el títol
    diu quants models hi entren de quants.
    """
    explained = list(outputs(models[0]))
    groups = models[0]["groups"]
    columns = models[0]["columns"]
    across = min(IMPORTANCE_PER_ROW, len(explained))
    rows = math.ceil(len(explained) / across)
    for kind in ("columns", "groups"):
        panels, axes = plt.subplots(
            rows,
            across,
            figsize=size(rows * WIDTH / across * IMPORTANCE_ASPECT),
            squeeze=False,
        )
        for axis in axes.flat[len(explained) :]:
            axis.set_visible(False)
        for axis, output in zip(axes.flat, explained, strict=False):
            # La seva part seria zero entre zero.
            kept = [model for model in models if np.any(outputs(model)[output])]
            counted = (
                f"{len(kept)} de {len(models)}"
                if len(kept) < len(models)
                else len(kept)
            )
            axis.set_title(f"{OUTPUT_LABELS[output]}, models: {counted}")
            if not kept:
                continue
            found = [importances(outputs(model)[output], groups) for model in kept]
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
            axis.barh(
                positions, table[:, order].mean(axis=0), color=colour, height=BAR_WIDTH
            )
            for row in table[:, order]:
                axis.plot(row, positions, "o", color=TEXT_COLOUR, alpha=POINT_OPACITY)
            axis.set_yticks(positions, names)
            axis.set_xlabel(
                "importància, la de la columna més important de cada model a 1"
                if kind == "columns"
                else "part de la importància"
            )
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


def dependence(folder, models, present, colour):
    """Dibuixa la contribució de cada sèrie de `present` contra el seu valor.

    Els punts, de `colour`, el color de la família, i la recta ajustada són del millor
    model ajustat de la primera seed, i el títol en dona el pendent i després la mitjana
    i la dispersió dels pendents de tots, que són el que llegeixen Trebbien et al.
    (2023). Cada sèrie té un tauler per sortida explicada, i una fila en porta tantes
    sèries com hi caben en `DEPENDENCE_PER_ROW` taulers, repartides igual entre les
    files.
    """
    explained = list(outputs(models[0]))
    units = dict(zip(explained, models[0]["units"], strict=True))
    rows = math.ceil(len(present) * len(explained) / DEPENDENCE_PER_ROW)
    per_row = math.ceil(len(present) / rows)
    rows = math.ceil(len(present) / per_row)
    across = per_row * len(explained)
    panels, axes = plt.subplots(
        rows,
        across,
        figsize=size(rows * WIDTH / across * DEPENDENCE_ASPECT),
        squeeze=False,
    )
    for axis in axes.flat[len(present) * len(explained) :]:
        axis.set_visible(False)
    for number, name in enumerate(present):
        first = number % per_row * len(explained)
        row = axes[number // per_row, first : first + len(explained)]
        for axis, output in zip(row, explained, strict=True):
            slopes = []
            for count, model in enumerate(models):
                x, y = pairs(model, outputs(model)[output], name)
                slope, intercept = np.polyfit(x / PER_GW, y, 1)
                slopes.append(slope)
                if count == 0:
                    axis.plot(
                        x / PER_GW,
                        y,
                        ".",
                        color=colour,
                        alpha=POINT_OPACITY,
                        rasterized=x.size > DENSE_POINTS,
                    )
                    span = np.array([x.min(), x.max()]) / PER_GW
                    axis.plot(span, slope * span + intercept, **REFERENCE_LINE)
            axis.set_xlabel(f"{GROUP_LABELS[name]}, GW")
            axis.set_ylabel(f"{OUTPUT_LABELS[output]}, {units[output]}")
            first_slope = catalan(f"{slopes[0]:.3g}")
            mean = catalan(f"{np.mean(slopes):.3g}")
            spread = catalan(f"{np.std(slopes):.2g}")
            axis.set_title(
                f"pendent {first_slope} {units[output]} per GW\nla mitjana dels "
                f"{len(slopes)} models, {mean} (±{spread})"
            )
    save(panels, folder, "dependence")
