"""Les figures de les xarxes distribucionals, del que la passada va desar.

Cada figura té un tauler per paràmetre.
"""

from epf.contract import figure


def draw(folder, models):
    """Dibuixa les figures d'una família a `folder`, dels seus models ajustats.

    La importància i la dependència.
    """
    figure.importance(folder, models)
    present = figure.present(models[0])
    if present:
        figure.dependence(folder, models, present)
