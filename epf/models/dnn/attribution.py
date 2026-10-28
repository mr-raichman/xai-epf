"""L'atribució de la xarxa neuronal, DeepExplainer."""

import collections
import logging
import warnings

import numpy as np
import shap

from epf.contract.explanation import explained_prediction, summed
from epf.models.dnn.network import inputs
from epf.parameters import HOURS_PER_DAY

# L'atribució de la predicció de cada hora, en EUR/MWh: `base` és `(24,)`, `values`
# `(files, columnes, 24)`, i `inputs` les entrades explicades, una columna per a cada
# una de `columns`.
Explanation = collections.namedtuple("Explanation", "columns inputs values base")


def _not_subsampled(record):
    """Diu si un registre de shap no és el que diu que retalla el fons a 100 files."""
    return not record.getMessage().startswith("Background dataset has")


def attribute_rows(fitted, X, background):
    """Torna l'atribució de la predicció de cada fila de `X` i hora, amb DeepExplainer.

    La base és la predicció mitjana sobre les files de `background`. Com que la xarxa
    no escala el que prediu, l'atribució surt ja en EUR/MWh.
    """
    shown = inputs(X)
    # Tres avisos inofensius: la base i les sumes de l'explicació ja es comproven, i
    # Keras desembolica la llista que li passa shap.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Your TensorFlow version is newer than 2.4.0", UserWarning
        )
        warnings.filterwarnings(
            "ignore",
            "The structure of `inputs` doesn't match the expected structure",
            UserWarning,
        )
        # El tercer és un registre del logger de shap, i no un avís.
        log = logging.getLogger("shap")
        log.addFilter(_not_subsampled)
        try:
            explainer = shap.DeepExplainer(fitted, inputs(background).to_numpy())
        finally:
            log.removeFilter(_not_subsampled)
        values = np.asarray(
            explainer.shap_values(shown.to_numpy(), check_additivity=False)
        )
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
            explained_prediction(found, HOURS_PER_DAY),
        )
    ]
