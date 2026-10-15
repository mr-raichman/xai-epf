"""L'atribució de les xarxes distribucionals: DeepExplainer, paràmetre a paràmetre, i les
sumes que se'n tornen a fer sobre el fitxer desat."""
import collections

import keras
import numpy as np
import shap

from epf.contract.explanation import explained_rows, summed
from epf.models.nn.network import inputs

# L'atribució de cada paràmetre: `values` és `(files, columnes, 24, paràmetres)`, `base`
# `(24, paràmetres)` i `inputs` les entrades explicades, una columna per a cada una de
# `columns`, en les unitats en què es desa cada paràmetre: EUR/MWh la localització i
# l'escala, i sense unitats l'asimetria i el pes de la cua.
Explanation = collections.namedtuple("Explanation", "names columns inputs values base")


def attribute_rows(distribution, fitted, X, background):
    """L'atribució de cada paràmetre de cada fila, amb DeepExplainer.

    S'explica cada paràmetre tal com el dóna la xarxa, i res que se'n construeixi:
    DeepExplainer no hi té cap regla per a `Log`, `Sinh` ni `Neg`, i no suma bé a
    través de cap combinació de paràmetres. La base és la sortida mitjana sobre
    `background`, que a la passada són les files d'entrenament de la llavor. Com que
    la xarxa no escala el preu, la localització i l'escala surten ja en EUR/MWh.
    """
    explained = keras.Model(fitted.input,
                            keras.layers.Lambda(distribution.flat)(fitted.output))
    # shap avisa que retalla el fons a 100 files, però ho fa l'emmascarador de la seva
    # classe base, que aquest explicador no fa servir: el càlcul les pren totes.
    explainer = shap.DeepExplainer(
        explained, inputs(background).to_numpy())
    shown = inputs(X)
    # Keras avisa que l'estructura de les entrades no és l'esperada perquè DeepExplainer
    # li passa el tensor dins d'una llista; el model el pren igual, i la suma de cada
    # paràmetre, que es comprova, ho confirma.
    values = np.asarray(explainer.shap_values(shown.to_numpy(), check_additivity=False))
    count = len(distribution.names)
    values = values.reshape(len(X), shown.shape[1], count, -1).transpose(0, 1, 3, 2)
    base = np.asarray(explainer.expected_value).reshape(count, -1).T
    return Explanation(distribution.names, list(shown.columns), shown.to_numpy(), values,
                       base)


def sums(found):
    """La suma d'una xarxa distribucional sobre el seu fitxer, contra els paràmetres de
    prova dels dies explicats. La de la base contra la mitjana dels dies d'entrenament
    només es fa a la memòria: el fitxer no en desa els paràmetres."""
    return [(*summed(found["base"], found["values"], 1),
             found["parameters"][explained_rows(found, 1)])]
