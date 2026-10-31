"""Les figures de l'arbre, del que la passada va desar."""

import numpy as np

from epf.contract import figure, measures

# Panells de les interaccions per fila, i l'alçada de cada un entre la seva amplada.
INTERACTIONS_PER_ROW = 2
INTERACTIONS_ASPECT = 3 / 4


def _interactions(folder, seed, model, main, colour):
    """Dibuixa les interaccions de cada sèrie amb panell d'un model ajustat, `model`.

    És el de la seed `seed`, a les seves files d'interaccions. Per sèrie, una figura,
    `interactions_<sèrie>.svg`: la dependència sencera i l'efecte sense interaccions,
    de `colour`, el color de la família, i la interacció amb cada un dels
    `measures.PARTNERS` grups més forts amb ella al model, dels números de la Part II,
    `main`: la suma de les de les columnes del grup, el doble, amb el color del valor
    del grup a l'hora de la fila.
    """
    family = folder.name
    groups = np.array([str(group) for group in model["groups"]])
    rows = model["interaction_rows"]
    values = np.asarray(measures.contributions(model), dtype=np.float64)[rows]
    inputs = np.asarray(model["inputs"], dtype=np.float64)[rows]
    interactions = np.asarray(model["interactions"], dtype=np.float64)
    coloured = {
        name: value[rows]
        for name, value in figure.group_values(model, len(model["inputs"])).items()
    }
    for name in measures.present(model):
        (j,) = measures.members(model, name)
        strengths = figure.rows(
            main, "partner_strength", family, seed=seed, rank=0, subject=name
        )
        partners = list(
            strengths.sort_values("value", ascending=False, kind="stable")[
                "other_subject"
            ]
        )[: measures.PARTNERS]
        shown = [
            (values[:, j], None, "sencera"),
            (interactions[:, j, j], None, "sense interaccions"),
        ] + [
            (
                2 * interactions[:, j, groups == partner].sum(axis=1),
                partner,
                f"amb {figure.GROUP_LABELS[partner]}",
            )
            for partner in partners
        ]
        x = inputs[:, j] / measures.PER_GW
        panels, axes = figure.panel_grid(
            len(shown), INTERACTIONS_PER_ROW, INTERACTIONS_ASPECT, sharex=True
        )
        for axis, (y, partner, title) in zip(axes.flat, shown, strict=False):
            if partner is None:
                axis.scatter(x, y, color=colour, **figure.point_style(x.size))
            else:
                image = axis.scatter(
                    x,
                    y,
                    c=coloured[partner],
                    cmap=figure.VALUE_SCALE,
                    **figure.point_style(x.size, coloured=True),
                )
                figure.colorbar(
                    panels,
                    image,
                    axis,
                    figure.unit_label("valor del grup", figure.GROUP_UNITS[partner]),
                )
            axis.set_title(title)
        figure.label_once(
            panels,
            axes,
            figure.unit_label(figure.GROUP_LABELS[name], "GW"),
            figure.unit_label("contribució", str(model["unit"])),
        )
        figure.save(panels, folder, f"interactions_{name}")


def draw(folder, main, secondary, colour):
    """Dibuixa la dependència i les interaccions d'una família a `folder`.

    De `colour`, el seu color, i dels números de la Part II de la passada, `main` i
    `secondary`. Les interaccions, si el model que dibuixa les porta.
    """
    found = figure.first_explained(folder)
    if found is None:
        return
    seed, path = found
    with np.load(path) as model:
        figure.dependence(folder, seed, model, main, secondary, colour)
        if "interactions" in model:
            _interactions(folder, seed, model, main, colour)
