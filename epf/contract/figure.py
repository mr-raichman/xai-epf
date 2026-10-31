"""L'aspecte de les figures, i les que dibuixen igual tots els models explicats.

Les dels models explicats es dibuixen dels models ajustats que la passada va desar.
"""

import collections
import itertools
import math

import matplotlib
import numpy as np

from epf import information
from epf.contract.explanation import CALENDAR, DATE, UNITLESS
from epf.contract.matrix import DAY_OF_WEEK, DAY_OF_WEEK_COLUMNS, HOUR
from epf.contract import measures
from epf.contract.measures import PER_GW
from epf.contract.record import fitted_files
from epf.information import KINDS, SERIES
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

# El gris d'on surten els tons d'una llegenda que travessa famílies, com la d'una
# ponderació, on el color d'una família es llegiria com ella.
LEGEND_GREY = rgb(91, 91, 91)

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


def label_once(drawing, axes, xlabel=None, ylabel=None):
    """Posa una sola vegada el títol de cada eix que comparteix una graella.

    Esborra el de cada panell. Un panell sense cap de visible a sota hi conserva les
    xifres, que un eix compartit deixa només a la fila de baix.
    """
    grid = np.atleast_2d(axes)
    for axis in grid.flat:
        if xlabel is not None:
            axis.set_xlabel("")
        if ylabel is not None:
            axis.set_ylabel("")
    if xlabel is not None:
        drawing.supxlabel(xlabel, fontsize=AXIS_TEXT_SIZE)
    if ylabel is not None:
        drawing.supylabel(ylabel, fontsize=AXIS_TEXT_SIZE)
    for row, column in itertools.product(range(grid.shape[0]), range(grid.shape[1])):
        below = grid[row + 1 :, column]
        if grid[row, column].get_visible() and not any(
            one.get_visible() for one in below
        ):
            grid[row, column].tick_params(labelbottom=True)


def grey(number):
    """Torna el gris `number` d'una llegenda, més clar com el to d'una família."""
    *channels, alpha = to_rgba(LEGEND_GREY)
    return (*(1 - (1 - np.array(channels)) * (1 - SHADE) ** number), alpha)


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


# Cada grup que fa algun information set, amb la seva sèrie i el seu retard.
GROUPS = information.groups()

# Com es diu cada grup de columnes a les figures: el nom de la seva sèrie, i a un
# retard seguit del dia, ", D-1".
GROUP_LABELS = {
    **{
        name: SERIES[series].label + (f", D-{lag}" if lag else "")
        for name, (series, lag) in GROUPS.items()
    },
    CALENDAR: "calendari",
    DATE: "comptador de dies",
}

# La unitat de cada grup de columnes, la de la seva sèrie al dataset.
GROUP_UNITS = {
    **{name: KINDS[SERIES[series].kind].unit for name, (series, _) in GROUPS.items()},
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


def unit_label(label, unit):
    """Torna el títol d'un eix amb la unitat, `preu (EUR/MWh)`, o sol si no en té."""
    return label if unit == UNITLESS else f"{label} ({unit})"


def p_value(value):
    """Torna `p = 0,03`, o `p < 0,001` per sota de `SMALLEST_P`."""
    if value < SMALLEST_P:
        return f"p < {catalan(str(SMALLEST_P))}"
    return f"p = {catalan(f'{value:.2g}')}"


### Números i files ###


def rows(frame, measure, family, **keys):
    """Torna les files d'una mesura i una família dels números d'una passada, `frame`.

    Una clau de `keys` amb un valor hi ha de tenir aquest valor, i una amb `None`, ser
    buida; `family` també. Una clau que no hi és pot valer qualsevol cosa.
    """
    found = frame[frame["measure"] == measure]
    for key, value in {"family": family, **keys}.items():
        found = found[found[key].isna() if value is None else found[key] == value]
    return found


def number(frame, measure, family, **keys):
    """Torna el valor de l'única fila que `rows` troba, i falla si no n'és una."""
    return float(rows(frame, measure, family, **keys)["value"].item())


def group_values(model, count):
    """Torna el valor de cada grup d'un model ajustat a cada una de `count` files.

    Un grup d'una columna, la columna; un de 24, la de l'hora de la fila; el
    calendari, el dia de la setmana, el dilluns 0; cap altre, NaN. La xarxa neuronal
    desa les entrades per dia, i cada dia hi val per les seves 24 files.
    """
    inputs = np.asarray(model["inputs"], dtype=np.float64)
    if len(inputs) < count:
        inputs = np.repeat(inputs, count // len(inputs), axis=0)
    hours = np.arange(count) % len(HOUR_COLUMNS)
    columns = [str(column) for column in model["columns"]]
    groups = np.array([str(group) for group in model["groups"]])
    found = {}
    for name in dict.fromkeys(groups):
        members = np.flatnonzero(groups == name)
        if len(members) == 1:
            found[name] = inputs[:, members[0]]
        elif len(members) == len(HOUR_COLUMNS):
            found[name] = inputs[:, members][np.arange(count), hours]
        elif name == CALENDAR:
            days = [j for j in members if columns[j].startswith(DAY_OF_WEEK)]
            found[name] = (
                inputs[:, days[0]]
                if len(days) == 1
                else inputs[:, days].argmax(axis=1).astype(float)
            )
        else:
            found[name] = np.full(count, np.nan)
    return found


def first_explained(folder):
    """Torna la seed i el fitxer del primer model d'una família amb contribucions.

    En ordre de seed i de configuració: la millor configuració de la primera seed, si
    en té alguna. `None` si cap model no en té.
    """
    for (seed, _), path in fitted_files(folder).items():
        with np.load(path) as model:
            if np.any(model["values"]):
                return seed, path
    return None


### Dependència ###


# Panells per fila de la dependència, i l'alçada de cada un entre la seva amplada.
DEPENDENCE_PER_ROW = 2
DEPENDENCE_ASPECT = 3.5 / 4.5


def dependence(folder, seed, model, main, secondary, colour):
    """Dibuixa la contribució de cada sèrie amb panell contra el seu valor.

    Els punts, de `colour`, el color de la família, i la recta ajustada són del model
    ajustat `model`, el primer de la família amb alguna contribució, de la seed
    `seed`, i el títol en dona el pendent i la mitjana i la desviació mostral dels de
    les seeds, dels números de la Part II, `main` i `secondary`, que són el que
    llegeixen Trebbien et al. (2023). Un panell per sèrie, `DEPENDENCE_PER_ROW` per
    fila, a `dependence.svg`.
    """
    family = folder.name
    present = measures.present(model)
    if not present:
        return
    values = measures.contributions(model)
    panels, axes = panel_grid(len(present), DEPENDENCE_PER_ROW, DEPENDENCE_ASPECT)
    for axis, name in zip(axes.flat, present, strict=False):
        x, y = measures.pairs(model, values, name)
        slope, intercept = measures.line(model, values, name)
        axis.scatter(x / PER_GW, y, color=colour, **point_style(x.size))
        span = np.array([x.min(), x.max()]) / PER_GW
        axis.plot(span, slope * span + intercept, **REFERENCE_LINE)
        axis.set_xlabel(unit_label(GROUP_LABELS[name], "GW"))
        found = number(
            secondary,
            "model_slope",
            family,
            seed=seed,
            rank=int(model["rank"]),
            subject=name,
        )
        mean, spread = (
            number(main, f"slope_{kind}", family, seed=None, subject=name)
            for kind in ("mean", "std")
        )
        axis.set_title(
            f"pendent {catalan(f'{found:.2g}')}\n"
            f"mitjana {catalan(f'{mean:.2g}')} ± {catalan(f'{spread:.2g}')}"
        )
    panels.supylabel(
        unit_label("contribució", str(model["unit"])), fontsize=AXIS_TEXT_SIZE
    )
    save(panels, folder, "dependence")
