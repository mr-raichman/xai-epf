"""Les figures de les xarxes distribucionals, del que la passada va desar.

Cada figura té un panell per paràmetre.
"""

from epf.contract import figure


def draw(folder, models, colour):
    """Dibuixa les figures d'una família a `folder`, dels seus models ajustats.

    La importància i la dependència, de `colour`, el color de la família.
    """
    figure.importance(folder, models, colour)
    present = figure.present(models[0])
    if present:
        figure.dependence(folder, models, present, colour)
