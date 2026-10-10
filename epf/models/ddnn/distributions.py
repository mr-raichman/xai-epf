"""Les dues distribucions de sortida de Marcjasz et al. (2023), i què en cal saber.

Cada una defineix una família a part, que és el que fa l'article: la normal té dos
paràmetres per hora i la de Johnson SU en té quatre, i la diferència entre les dues és
la seva pròpia evidència que els moments alts compten en aquest mercat.

De cada una en calen cinc coses. Quantes sortides vol la xarxa, la pèrdua que
minimitza, quin preu puntual se'n deriva, com se'n treuen els quantils i com es diuen
els seus paràmetres. Les tres primeres són de la xarxa i les dues últimes de qui la
puntua, i hi són juntes perquè totes cinc parlen de la mateixa distribució: separar-les
seria tenir la definició en dos llocs.

La pèrdua és la log-versemblança negativa, que és el que l'article minimitza allà on la
xarxa de punt minimitza el MAE, i és tota la diferència entre les dues.
"""
import collections

import numpy as np
from keras import ops
from scipy import stats

from epf.market import HOURS_PER_DAY

# Terra de tot paràmetre d'escala, perquè `softplus` pot donar zero i un zero a
# l'escala fa infinita la log-versemblança.
FLOOR = 1e-3

# Una distribució de sortida. `units` són les sortides que vol la xarxa, `loss` la
# pèrdua que minimitza, `parameters` llegeix la sortida crua en les unitats en què es
# desa, `point` el preu puntual que se'n deriva, `quantiles` els quantils predits i
# `names` com es diuen els seus paràmetres, en l'ordre en què surten de la xarxa.
#
# `of_day` és `None` aquí i el registre de famílies l'omple: és `(model, X, dia)` cap
# als paràmetres d'aquell dia, i per tant demana el model, que és de la família i no de
# la distribució. Va al mateix lloc perquè qui escriu els paràmetres al disc i qui en
# calcula el CRPS volen les quatre coses alhora i no han de muntar-les.
Distribution = collections.namedtuple(
    "Distribution", "names units loss parameters point quantiles of_day",
    defaults=(None,))


def _split(raw, count):
    """Les `count` columnes de paràmetres que la xarxa ha tret, cada una (files, 24)."""
    return [raw[..., number * HOURS_PER_DAY:(number + 1) * HOURS_PER_DAY]
            for number in range(count)]


def _normal_loss(target, raw):
    location, scale = _split(raw, 2)
    scale = FLOOR + ops.softplus(scale)
    return ops.mean(0.5 * ((target - location) / scale) ** 2 + ops.log(scale), axis=-1)


def _normal_parameters(raw):
    """Els paràmetres en les unitats en què es desen i es llegeixen."""
    location, scale = _split(np.asarray(raw), 2)
    return np.stack([location, FLOOR + np.logaddexp(0.0, scale)], axis=-1)


def _normal_quantiles(parameters, probabilities):
    location, scale = parameters[..., 0], parameters[..., 1]
    return stats.norm.ppf(probabilities, loc=location[..., None],
                          scale=scale[..., None])


def _johnson_loss(target, raw):
    """La de Johnson SU, amb l'asimetria i el pes de la cua com a paràmetres propis.

    `-log p = 0.5 * (skew + tail * asinh(z))^2 + log(scale) + 0.5 * log(1 + z^2)
    - log(tail)`, amb `z = (y - location) / scale`, llevat de constants que no depenen
    de cap paràmetre i que per tant no mouen cap gradient.
    """
    location, scale, skew, tail = _split(raw, 4)
    scale, tail = FLOOR + ops.softplus(scale), FLOOR + ops.softplus(tail)
    z = (target - location) / scale
    root = ops.sqrt(1 + z ** 2)
    return ops.mean(0.5 * (skew + tail * ops.log(z + root)) ** 2
                    + ops.log(scale) + ops.log(root) - ops.log(tail), axis=-1)


def _johnson_parameters(raw):
    location, scale, skew, tail = _split(np.asarray(raw), 4)
    return np.stack([location, FLOOR + np.logaddexp(0.0, scale), skew,
                     FLOOR + np.logaddexp(0.0, tail)], axis=-1)


def _johnson_quantiles(parameters, probabilities):
    # `scipy` anomena `a` l'asimetria i `b` el pes de la cua, i la seva densitat porta
    # `a + b * asinh(z)`, que és la mateixa parametrització que la pèrdua de dalt.
    location, scale = parameters[..., 0], parameters[..., 1]
    skew, tail = parameters[..., 2], parameters[..., 3]
    return stats.johnsonsu.ppf(probabilities, skew[..., None], tail[..., None],
                               loc=location[..., None], scale=scale[..., None])


# El preu puntual és la mediana de la distribució i no la mitjana, que és el que fa
# l'article per al MAE: la mediana és el que minimitza l'error absolut.
NORMAL = Distribution(
    names=("location", "scale"), units=2 * HOURS_PER_DAY, loss=_normal_loss,
    parameters=_normal_parameters,
    # La mediana d'una normal és la seva localització.
    point=lambda parameters: parameters[..., 0],
    quantiles=_normal_quantiles)

JOHNSON_SU = Distribution(
    names=("location", "scale", "skewness", "tail_weight"),
    units=4 * HOURS_PER_DAY, loss=_johnson_loss,
    parameters=_johnson_parameters,
    point=lambda parameters: _johnson_quantiles(parameters, 0.5)[..., 0],
    quantiles=_johnson_quantiles)

# Com es diu cada família i quina distribució treu.
FAMILIES = {"ddnn_normal": NORMAL, "ddnn_jsu": JOHNSON_SU}
