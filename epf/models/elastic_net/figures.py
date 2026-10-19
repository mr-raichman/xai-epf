"""Les figures de l'elastic net, del que la passada va desar.

Són a l'espai transformat, i no en EUR/MWh.
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
