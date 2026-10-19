"""Les figures de la xarxa elàstica, del que la passada va desar.

Són a l'espai transformat, i no en EUR/MWh.
"""

from epf.contract import figure


def draw(folder, seeds):
    """Dibuixa les figures d'una família a `folder`, dels seus models desats, `seeds`.

    La importància i la dependència.
    """
    figure.importance(folder, seeds)
    present = figure.present(seeds[0])
    if present:
        figure.dependence(folder, seeds, present)
