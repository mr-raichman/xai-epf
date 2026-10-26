"""L'aspecte de les figures, i les que dibuixen igual tots els models explicats.

Les dels models explicats es dibuixen dels models ajustats que la passada va desar.
"""

import collections
import math

import matplotlib
import numpy as np

from epf.contract.explanation import CALENDAR, DATE, UNITLESS
from epf.contract.matrix import (
    DAY_OF_WEEK,
    DAY_OF_WEEK_COLUMNS,
    HOUR,
    INFORMATION,
    PRICE,
)
from epf.contract.record import fitted_files
from epf.parameters import HOUR_COLUMNS

# El motor sense finestra, triat abans d'importar `pyplot`: qui importa aquest mòdul ja
# el té triat.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


### Aspecte ###


def rgb(red, green, blue):
    """Torna el color dels tres canals, de 0 a 255, en els de 0 a 1 de matplotlib."""
    return (red / 255, green / 255, blue / 255)


# La lletra de tot el text de les figures, la de la memòria, i la que la substitueix on
# no hi és, que en vol tenir les mides: el text mesurat amb ella hi cap igual quan la
# memòria el posa en la primera.
FONT = "Times New Roman"
FALLBACK_FONT = "Liberation Serif"

# Mides del text, en punts, des de la del peu de figura de la memòria: el títol d'una
# figura o d'un panell, un punt per sota, i tot el text dels eixos, dos.
CAPTION_SIZE = 12
PANEL_HEADING_SIZE = CAPTION_SIZE - 1
AXIS_TEXT_SIZE = CAPTION_SIZE - 2

# `PANEL_HEADING_PLACE`, "left", "center" o "right".
PANEL_HEADING_PLACE = "center"

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

# El color dels bancs de comparació, el sòl i les còpies.
BENCHMARK_COLOUR = rgb(106, 66, 140)

# El color de cada model. Les seves famílies en prenen tons, en l'ordre en què el model
# les fa, cada una `SHADE` més a prop del blanc que l'anterior; les d'un model del color
# d'un altre segueixen els tons de les de l'altre.
MODEL_COLOURS = {
    "dnn": rgb(12, 115, 208),
    "elastic_net": rgb(66, 131, 95),
    "lightgbm": rgb(171, 97, 37),
    "floor": BENCHMARK_COLOUR,
    "naive": BENCHMARK_COLOUR,
}
SHADE = 0.32

# El color de tot el que no és d'una família, com les sèries del dataset, i el que hi
# marca uns punts triats, més fosc perquè se'n distingeixin també per la claredat.
OTHER_COLOUR = rgb(163, 86, 130)
MARK_COLOUR = rgb(120, 20, 20)

# L'escala dels valors continus, pel seu nom a matplotlib.
SCALE = "viridis"

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

# L'amplada d'una barra, en fracció del lloc que té.
BAR_WIDTH = 0.7

# Un panell d'una fila per cosa, una barra o els punts d'una família, creix amb les
# seves files: el lloc de cada una, i el del títol, l'eix i les seves xifres, en mm.
ROW_PITCH = 4.5
ROW_PANEL_FRAME = 16

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

# Les capes de més punts que aquests, denses, es desen com a imatge, a `DPI`: com a
# vector, el SVG en guardaria un objecte per punt.
DENSE_POINTS = 5000

# Una capa densa, com milers de dies superposats o les desenes de milers d'hores d'una
# sèrie, va més prima i, si les seves línies o els seus punts se superposen, més
# transparent: amb el gruix i la mida de sempre es tapen, i en surt una taca en lloc
# d'on se n'amunteguen més.
DENSE_LINE_WIDTH = 0.2
DENSE_MARKER_SIZE = 0.6
DENSE_LINE_OPACITY = 0.15
DENSE_POINT_OPACITY = 0.3

# `SVG_TEXT`, "path", el text com a traços, o "none", com a text, que el PDF deixa
# seleccionar i cercar.
SVG_TEXT = "none"

# Els separadors de les xifres, els de la memòria.
DECIMAL_SEPARATOR = ","
THOUSANDS_SEPARATOR = "."

# El que matplotlib pren per defecte, fet de les variables de dalt: qui importa aquest
# mòdul ja el té.
STYLE = {
    "font.family": [FONT, FALLBACK_FONT],
    "text.color": TEXT_COLOUR,
    "axes.labelcolor": TEXT_COLOUR,
    "xtick.color": TEXT_COLOUR,
    "ytick.color": TEXT_COLOUR,
    "axes.labelsize": AXIS_TEXT_SIZE,
    "axes.titlesize": PANEL_HEADING_SIZE,
    "axes.titlelocation": PANEL_HEADING_PLACE,
    "figure.titlesize": PANEL_HEADING_SIZE,
    "xtick.labelsize": AXIS_TEXT_SIZE,
    "ytick.labelsize": AXIS_TEXT_SIZE,
    "legend.fontsize": AXIS_TEXT_SIZE,
    "legend.title_fontsize": AXIS_TEXT_SIZE,
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

    `models` dona les famílies de cada model, en ordre.
    """
    taken = collections.Counter()
    found = {}
    for model, families in models.items():
        for family in families:
            found[family] = shade(model, taken[MODEL_COLOURS[model]])
            taken[MODEL_COLOURS[model]] += 1
    return found


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
    bar.set_label(label, fontsize=AXIS_TEXT_SIZE)
    bar.ax.tick_params(labelsize=AXIS_TEXT_SIZE)
    bar.outline.set_linewidth(FRAME_WIDTH)
    bar.outline.set_edgecolor(FRAME_COLOUR)


def line_style(count, overlapping):
    """Torna com es dibuixa una capa de línies de `count` punts en total.

    El gruix i si es desa com a imatge, per a `plot` o una col·lecció de línies, i
    l'opacitat si les seves línies se superposen, `overlapping`.
    """
    dense = count > DENSE_POINTS
    found = {
        "linewidth": DENSE_LINE_WIDTH if dense else LINE_WIDTH,
        "rasterized": dense,
    }
    if overlapping:
        found["alpha"] = DENSE_LINE_OPACITY if dense else POINT_OPACITY
    return found


def point_style(count, coloured=False):
    """Torna com es dibuixa una capa de `count` punts, per a `scatter`.

    La mida, l'opacitat, cap vora, i si es desa com a imatge. Uns punts pintats pel seu
    valor, `coloured`, són opacs: transparents, el seu color ja no diria el valor.
    """
    dense = count > DENSE_POINTS
    return {
        "s": (DENSE_MARKER_SIZE if dense else MARKER_SIZE) ** 2,
        "alpha": 1 if coloured else DENSE_POINT_OPACITY if dense else POINT_OPACITY,
        "linewidths": 0,
        "rasterized": dense,
    }


def rows_height(count):
    """Torna en mm l'alçada d'un panell de `count` files."""
    return count * ROW_PITCH + ROW_PANEL_FRAME


def panel_grid(count, per_row, aspect, **options):
    """Torna una figura de `count` panells, `per_row` per fila, i la seva graella.

    L'alçada de cada panell és `aspect` vegades la seva amplada, i els de la graella que
    passen de `count` s'amaguen. `options` va a `plt.subplots`, com `sharey`.
    """
    rows = math.ceil(count / per_row)
    drawing, axes = plt.subplots(
        rows,
        per_row,
        figsize=size(rows * WIDTH / per_row * aspect),
        squeeze=False,
        **options,
    )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return drawing, axes


def save(drawing, folder, name):
    """Desa una figura a `folder/<name>.svg` i la tanca.

    Abans hi posa els separadors de la memòria a les xifres dels eixos.
    """
    localise(drawing)
    drawing.savefig(folder / f"{name}.svg")
    plt.close(drawing)


### Noms i unitats ###


# Com es diu cada grup de columnes a les figures, en l'ordre en què les sèries es
# mostren a tot arreu.
GROUP_LABELS = {
    "price": "preu del dia abans",
    "demand_fcst": "demanda prevista",
    "solar_fcst": "fotovoltaica prevista",
    "wind_fcst": "eòlica prevista",
    "nuclear_sched": "nuclear programada",
    "hydro_sched": "hidràulica programada",
    "coal_gas_sched": "carbó i gas programats",
    CALENDAR: "calendari",
    DATE: "comptador de dies",
}

# La unitat de cada grup de columnes, la de la seva sèrie al dataset.
GROUP_UNITS = {
    "price": "EUR/MWh",
    "demand_fcst": "MW",
    "solar_fcst": "MW",
    "wind_fcst": "MW",
    "nuclear_sched": "MWh",
    "hydro_sched": "MWh",
    "coal_gas_sched": "MWh",
    CALENDAR: UNITLESS,
    DATE: "dies",
}

# Com es diu cada columna del calendari a les figures; el dia 0 de la setmana és el
# dilluns.
WEEKDAYS = (
    "dilluns",
    "dimarts",
    "dimecres",
    "dijous",
    "divendres",
    "dissabte",
    "diumenge",
)
CALENDAR_LABELS = {
    HOUR: "hora del dia",
    DAY_OF_WEEK: "dia de la setmana",
    **dict(zip(DAY_OF_WEEK_COLUMNS, WEEKDAYS, strict=True)),
}

# Com es diu cada família a les figures.
FAMILY_LABELS = {
    "dnn": "xarxa neuronal",
    "elastic_net": "elastic net",
    "lightgbm": "arbre",
    "date_probe": "sonda de datació",
    "floor": "sòl",
    "naive_similar_day": "dia similar",
    "naive_previous_week": "setmana anterior",
    "naive_previous_day": "dia anterior",
}

# El valor p més petit que una figura escriu; un de menor s'hi escriu `p < 0,001`.
SMALLEST_P = 0.001

# MW per GW, perquè els pendents es llegeixen per GW, com fan Trebbien et al. (2023).
PER_GW = 1000


def unit_label(label, unit):
    """Torna el títol d'un eix amb la unitat, `preu (EUR/MWh)`, o sol si no en té."""
    return label if unit == UNITLESS else f"{label} ({unit})"


def p_value(value):
    """Torna `p = 0,03`, o `p < 0,001` per sota de `SMALLEST_P`."""
    if value < SMALLEST_P:
        return f"p < {catalan(str(SMALLEST_P))}"
    return f"p = {catalan(f'{value:.2g}')}"


def column_label(column, group):
    """Torna com es diu una columna a les figures, del seu grup, `group`, i l'hora.

    `price_lag1_h23`, del grup `price`, és `preu del dia abans, 23 h`.
    """
    if column in CALENDAR_LABELS:
        return CALENDAR_LABELS[column]
    hour = column.rsplit("_", 1)[-1]
    if hour in HOUR_COLUMNS:
        return f"{GROUP_LABELS[group]}, {int(hour[1:])} h"
    return GROUP_LABELS[group]


### Models ajustats ###


def saved(folder):
    """Torna els models ajustats d'una família, en ordre de seed i de configuració.

    El primer és la millor configuració de la primera seed.
    """
    return [np.load(path) for path in fitted_files(folder).values()]


def contributions(model):
    """Torna les contribucions d'un model ajustat, `(files, columnes)`.

    Les de la xarxa neuronal, `(dies, columnes, 24)`, amb una fila per dia i hora.
    """
    values = model["values"]
    if values.ndim == 2:
        return values
    return values.transpose(0, 2, 1).reshape(-1, values.shape[1])


### Importància ###


# Columnes que mostra la importància per columna; la resta compten al seu grup.
SHOWN = 20

# L'alçada de la importància, amb els noms de les columnes drets a sota, en mm.
IMPORTANCE_HEIGHT = 70


def importances(values, groups):
    """Torna la importància de cada columna i la de cada grup, la seva part del total.

    És la mitjana del valor absolut de les contribucions, en part del total com la
    donen Pesenti i O'Sullivan (2026), de manera que la d'una columna i la d'un grup es
    llegeixen igual. La d'un grup suma les contribucions de les seves columnes abans de
    prendre-la, com Pesenti i O'Sullivan (2025). Es calcula en `float64`.
    """
    values = np.asarray(values, dtype=np.float64)
    by_column = np.abs(values).mean(axis=0)
    names = list(dict.fromkeys(groups))
    by_group = np.array(
        [np.abs(values[:, groups == name].sum(axis=1)).mean() for name in names]
    )
    return by_column / by_column.sum(), dict(
        zip(names, by_group / by_group.sum(), strict=True)
    )


def importance(folder, models, colour):
    """Dibuixa la importància de les columnes i dels grups, amb cada model ajustat.

    De les columnes, les `SHOWN` que més pesen, de la que més a l'esquerra; dels grups,
    tots en el seu ordre. La mitjana, en barres de `colour`, el color de la família, i
    cada model ajustat, en un punt. Un model ajustat sense cap contribució, que no
    depèn de les entrades, no hi entra, i el títol diu llavors quants models hi entren
    de quants.
    """
    groups = models[0]["groups"]
    columns = models[0]["columns"]
    # La seva part seria zero entre zero.
    kept = [model for model in models if np.any(contributions(model))]
    found = [importances(contributions(model), groups) for model in kept]
    for kind in ("columns", "groups"):
        panels, axis = plt.subplots(figsize=size(IMPORTANCE_HEIGHT))
        if len(kept) < len(models):
            axis.set_title(f"models ajustats: {len(kept)} de {len(models)}")
        if found:
            if kind == "columns":
                table = np.array([by_column for by_column, _ in found])
                order = np.argsort(-table.mean(axis=0), kind="stable")[:SHOWN]
                names = [column_label(columns[j], groups[j]) for j in order]
            else:
                # Un grup sense nom hi falla, com a `column_label`, i no en queda fora
                # amb la seva part al total.
                place = {name: number for number, name in enumerate(GROUP_LABELS)}
                labels = sorted(found[0][1], key=lambda name: place[name])
                table = np.array(
                    [[by_group[name] for name in labels] for _, by_group in found]
                )
                order = np.arange(len(labels))
                names = [GROUP_LABELS[name] for name in labels]
            positions = np.arange(len(order))
            axis.bar(
                positions, table[:, order].mean(axis=0), color=colour, width=BAR_WIDTH
            )
            for row in table[:, order]:
                axis.plot(positions, row, "o", color=TEXT_COLOUR, alpha=POINT_OPACITY)
            axis.set_xticks(positions, names, rotation=90)
        panels.supylabel("part de la importància", fontsize=AXIS_TEXT_SIZE)
        save(panels, folder, "importance" if kind == "columns" else "groups")


### Dependència ###


# Les sèries que tenen panells de dependència, en el seu ordre: totes les de
# l'information set menys el preu, les previsions com a Trebbien et al. (2023) i el
# programa igual.
DEPENDENCE = tuple(
    name for name in GROUP_LABELS if name in INFORMATION and name != PRICE
)

# Panells per fila de la dependència, i l'alçada de cada un entre la seva amplada.
DEPENDENCE_PER_ROW = 2
DEPENDENCE_ASPECT = 3.5 / 4.5


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
    # La xarxa neuronal: a la fila del dia d i l'hora h, la columna de l'hora h.
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
    (2023). Un panell per sèrie, `DEPENDENCE_PER_ROW` per fila, a `dependence.svg`.
    """
    panels, axes = panel_grid(len(present), DEPENDENCE_PER_ROW, DEPENDENCE_ASPECT)
    for axis, name in zip(axes.flat, present, strict=False):
        slopes = []
        for count, model in enumerate(models):
            x, y = pairs(model, contributions(model), name)
            slope, intercept = np.polyfit(x / PER_GW, y, 1)
            slopes.append(slope)
            if count == 0:
                axis.scatter(x / PER_GW, y, color=colour, **point_style(x.size))
                span = np.array([x.min(), x.max()]) / PER_GW
                axis.plot(span, slope * span + intercept, **REFERENCE_LINE)
        axis.set_xlabel(unit_label(GROUP_LABELS[name], "GW"))
        first_slope = catalan(f"{slopes[0]:.2g}")
        mean = catalan(f"{np.mean(slopes):.2g}")
        spread = catalan(f"{np.std(slopes):.2g}")
        axis.set_title(f"pendent {first_slope}\nmitjana {mean} ± {spread}")
    panels.supylabel(
        unit_label("contribució", str(models[0]["unit"])), fontsize=AXIS_TEXT_SIZE
    )
    save(panels, folder, "dependence")
