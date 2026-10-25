"""L'atribució de la xarxa neuronal, DeepExplainer."""

import collections

import numpy as np
import shap

from epf.contract.explanation import explained_rows, summed
from epf.models.dnn.network import inputs
from epf.parameters import HOURS_PER_DAY

# L'atribució del preu de cada hora, en EUR/MWh: `base` és `(24,)`, `values`
# `(files, columnes, 24)`, i `inputs` les entrades explicades, una columna per a cada
# una de `columns`.
Explanation = collections.namedtuple("Explanation", "columns inputs values base")


def attribute_rows(fitted, X, background):
    """Torna l'atribució del preu de cada fila de `X` i cada hora, amb DeepExplainer.

    La base és la predicció mitjana sobre les files de `background`. Com que la xarxa
    no escala el preu, l'atribució surt ja en EUR/MWh.
    """
    # `shap` avisa que retalla el fons a 100 files, però l'avís és de la seva classe
    # base: DeepExplainer les pren totes. Avisa també, de qualsevol model, que des de
    # TensorFlow 2.4.0 alguns grafs estàtics poden no anar bé; la suma de cada hora, que
    # es comprova, en destaparia un.
    explainer = shap.DeepExplainer(fitted, inputs(background).to_numpy())
    shown = inputs(X)
    # Keras avisa que l'estructura de les entrades no és l'esperada perquè DeepExplainer
    # li passa el tensor dins d'una llista; el model el pren igual, i la suma de cada
    # hora, que es comprova, ho confirma.
    values = np.asarray(explainer.shap_values(shown.to_numpy(), check_additivity=False))
    return Explanation(
        list(shown.columns),
        shown.to_numpy(),
        values,
        np.asarray(explainer.expected_value),
    )


def sums(found):
    """Torna la suma de la xarxa neuronal sobre el seu fitxer, `found`.

    La base més les contribucions, contra la predicció de prova dels dies explicats. La
    de la base contra la mitjana dels dies d'entrenament només es fa a la passada: el
    fitxer no en desa la predicció.
    """
    total, magnitude = summed(found["base"], found["values"], 1)
    return [
        (
            total.ravel(),
            magnitude.ravel(),
            found["predicted"][explained_rows(found, HOURS_PER_DAY)],
        )
    ]
