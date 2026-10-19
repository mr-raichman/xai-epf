"""L'atribució de l'arbre, TreeSHAP i les seves interaccions."""

import numpy as np
import shap

from epf.contract.explanation import explained_rows, summed
from epf.parameters import HOURS_PER_DAY


def attribute_rows(model, X):
    """Torna els valors de SHAP de cada fila, `(files, columnes)`, i la base.

    TreeSHAP al mode que LightGBM calcula ell mateix; com que el model és un de sol per
    a les 24 hores, la base és la mateixa per a totes les files.
    """
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X), float(np.ravel(explainer.expected_value)[0])


def interaction_rows(model, X):
    """Torna els valors d'interacció de SHAP de cada fila.

    Són `(files, columnes, columnes)`.
    """
    return shap.TreeExplainer(model).shap_interaction_values(X)


def sums(found, best):
    """Torna les sumes de l'arbre que es tornen a fer sobre el seu fitxer, `found`.

    La base més les contribucions de cada fila contra la seva predicció i, a la millor
    configuració, `best`, cada fila d'interaccions contra la contribució de la seva
    columna.
    """
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
