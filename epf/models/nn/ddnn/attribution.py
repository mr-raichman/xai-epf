"""L'atribució de les xarxes distribucionals, DeepExplainer paràmetre a paràmetre."""

import collections

import keras
import numpy as np
import shap

from epf.contract.explanation import explained_rows, summed
from epf.models.nn.network import inputs

# L'atribució de cada paràmetre, en les seves unitats: `base` és `(24, paràmetres)`,
# `values` `(files, columnes, 24, paràmetres)`, i `inputs` les entrades explicades, una
# columna per a cada una de `columns`.
Explanation = collections.namedtuple("Explanation", "names columns inputs values base")


def attribute_rows(distribution, fitted, X, background):
    """Torna l'atribució de cada paràmetre de cada fila de `X`, amb DeepExplainer.

    La base és la sortida mitjana sobre les files de `background`. S'explica cada
    paràmetre tal com el dóna la xarxa, i res que se'n construeixi: DeepExplainer no hi
    té cap regla per a `Log`, `Sinh` ni `Neg`, i no suma bé a través de cap combinació
    de paràmetres. Com que la xarxa no escala el preu, la localització i l'escala surten
    ja en EUR/MWh.
    """
    explained = keras.Model(
        fitted.input, keras.layers.Lambda(distribution.flat)(fitted.output)
    )
    # `shap` avisa que retalla el fons a 100 files, però l'avís és de la seva classe
    # base: DeepExplainer les pren totes.
    explainer = shap.DeepExplainer(explained, inputs(background).to_numpy())
    shown = inputs(X)
    # Keras avisa que l'estructura de les entrades no és l'esperada perquè DeepExplainer
    # li passa el tensor dins d'una llista; el model el pren igual, i la suma de cada
    # paràmetre, que es comprova, ho confirma.
    values = np.asarray(explainer.shap_values(shown.to_numpy(), check_additivity=False))
    count = len(distribution.names)
    values = values.reshape(len(X), shown.shape[1], count, -1).transpose(0, 1, 3, 2)
    base = np.asarray(explainer.expected_value).reshape(count, -1).T
    return Explanation(
        distribution.names, list(shown.columns), shown.to_numpy(), values, base
    )


def sums(found):
    """Torna la suma d'una xarxa distribucional sobre el seu fitxer, `found`.

    La base més les contribucions, contra els paràmetres de prova dels dies explicats.
    La de la base contra la mitjana dels dies d'entrenament només es fa a la passada: el
    fitxer no en desa els paràmetres.
    """
    return [
        (
            *summed(found["base"], found["values"], 1),
            found["parameters"][explained_rows(found, 1)],
        )
    ]
