"""L'atribució de l'arbre: TreeSHAP i les seves interaccions, i les sumes que se'n
tornen a fer sobre el fitxer desat."""

import numpy as np
import shap

from epf.contract.explanation import explained_rows, summed
from epf.parameters import HOURS_PER_DAY


def attribute_rows(model, X):
    """Els valors de SHAP de cada fila, `(files, columnes)`, i la base.

    TreeSHAP al mode que LightGBM calcula ell mateix, i un sol model per a les 24
    hores, de manera que la base és la mateixa per a totes les files.
    """
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X), float(np.ravel(explainer.expected_value)[0])


def interaction_rows(model, X):
    """Els valors d'interacció de SHAP de cada fila, `(files, columnes, columnes)`."""
    return shap.TreeExplainer(model).shap_interaction_values(X)


def sums(found, best):
    """Les sumes de l'arbre sobre el seu fitxer: la base més els valors de cada fila
    contra la seva predicció i, a la millor configuració, cada fila d'interaccions
    contra el valor de la seva columna."""
    found_sums = [
        (
            *summed(found["base"], found["values"], 1),
            found["predicted"][explained_rows(found, HOURS_PER_DAY)],
        )
    ]
    if best:
        found_sums.append(
            (
                *summed(0.0, found["interactions"], 2),
                found["values"][found["interaction_rows"]],
            )
        )
    return found_sums
