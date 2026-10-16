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

from keras import ops
from scipy import stats

from epf.models.nn.network import Distribution, heads

# Com surten l'escala i el pes de la cua de la sortida crua, al codi de Marcjasz et al.
# (2023): `FLOOR + STRETCH * softplus` l'escala, que no pot ser zero perquè un zero hi
# fa infinita la log-versemblança, i `TAIL_FLOOR + STRETCH * softplus` el pes de la cua
# de Johnson SU, que mai no baixa d'1.
FLOOR = 1e-3
STRETCH = 3
TAIL_FLOOR = 1


def _scale(raw):
    return FLOOR + STRETCH * ops.softplus(raw)


def _tail(raw):
    return TAIL_FLOOR + STRETCH * ops.softplus(raw)


def _normal_loss(target, raw):
    location, scale = heads(raw, 2)
    scale = _scale(scale)
    return ops.mean(0.5 * ((target - location) / scale) ** 2 + ops.log(scale), axis=-1)


def _normal_flat(raw):
    location, scale = heads(raw, 2)
    return ops.concatenate([location, _scale(scale)], axis=-1)


def _normal_quantiles(parameters, probabilities):
    location, scale = parameters[..., 0], parameters[..., 1]
    return stats.norm.ppf(
        probabilities, loc=location[..., None], scale=scale[..., None]
    )


def _johnson_loss(target, raw):
    """La de Johnson SU, amb l'asimetria i el pes de la cua com a paràmetres propis.

    `-log p = 0.5 * (skew + tail * asinh(z))^2 + log(scale) + 0.5 * log(1 + z^2)
    - log(tail)`, amb `z = (y - location) / scale`, llevat de constants que no depenen
    de cap paràmetre i que per tant no mouen cap gradient.
    """
    location, scale, skew, tail = heads(raw, 4)
    scale, tail = _scale(scale), _tail(tail)
    z = (target - location) / scale
    # `arcsinh` i no `log(z + sqrt(1 + z^2))`, que en float32 es cancel·la per a z
    # negatives: a z = -10000 dóna menys infinit.
    return ops.mean(
        0.5 * (skew + tail * ops.arcsinh(z)) ** 2
        + ops.log(scale)
        + 0.5 * ops.log(1 + z**2)
        - ops.log(tail),
        axis=-1,
    )


def _johnson_flat(raw):
    location, scale, skew, tail = heads(raw, 4)
    return ops.concatenate([location, _scale(scale), skew, _tail(tail)], axis=-1)


def _johnson_quantiles(parameters, probabilities):
    # `scipy` anomena `a` l'asimetria i `b` el pes de la cua, i la seva densitat porta
    # `a + b * asinh(z)`, que és la mateixa parametrització que la pèrdua de dalt.
    location, scale = parameters[..., 0], parameters[..., 1]
    skew, tail = parameters[..., 2], parameters[..., 3]
    return stats.johnsonsu.ppf(
        probabilities,
        skew[..., None],
        tail[..., None],
        loc=location[..., None],
        scale=scale[..., None],
    )


# El preu puntual és la mediana de la distribució i no la mitjana, que és el que fa
# l'article per al MAE: la mediana és el que minimitza l'error absolut.
NORMAL = Distribution(
    names=("location", "scale"),
    loss=_normal_loss,
    flat=_normal_flat,
    # La mediana d'una normal és la seva localització.
    point=lambda parameters: parameters[..., 0],
    quantiles=_normal_quantiles,
)

JOHNSON_SU = Distribution(
    names=("location", "scale", "skewness", "tail_weight"),
    loss=_johnson_loss,
    flat=_johnson_flat,
    point=lambda parameters: _johnson_quantiles(parameters, 0.5)[..., 0],
    quantiles=_johnson_quantiles,
)

# Com es diu cada família i quina distribució treu.
DISTRIBUTIONS = {"ddnn_normal": NORMAL, "ddnn_jsu": JOHNSON_SU}
