"""Com es fa i com es veu una figura d'aquest projecte, en un sol lloc.

Ho comparteixen els dos llocs que en dibuixen: `results/` les d'una passada i
`data/figures/` les de les dades d'origen. Són germans, així que el que tots dos llegeixen
viu al paquet de dalt. Només hi ha el que fan servir dues coses o més: el que és d'una
figura concreta, com el vermell dels valors marcats, viu amb ella.

El motor és `Agg`, sense finestra, i es tria aquí: qui importa aquest mòdul ja el té
triat i no ha de recordar-ho abans d'importar `pyplot`.
"""
import matplotlib

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
