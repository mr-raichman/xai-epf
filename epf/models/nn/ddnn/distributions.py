"""Les dues distribucions de sortida de Marcjasz et al. (2023)."""

from keras import ops
from scipy import stats

from epf.models.nn.hyperparameters import SCALE_FLOOR, STRETCH, TAIL_FLOOR
from epf.models.nn.network import Distribution, heads


def _scale(raw):
    return SCALE_FLOOR + STRETCH * ops.softplus(raw)


def _tail(raw):
    return TAIL_FLOOR + STRETCH * ops.softplus(raw)


def _johnson_loss(target, raw):
    """Torna la log-versemblança negativa de Johnson SU, llevat de constants.

    `-log p = 0.5 * (skew + tail * asinh(z))^2 + log(scale) + 0.5 * log(1 + z^2)
    - log(tail)`, amb `z = (y - location) / scale`.
    """
    location, scale, skew, tail = heads(raw, 4)
    scale, tail = _scale(scale), _tail(tail)
    z = (target - location) / scale
    # `arcsinh` i no `log(z + sqrt(1 + z^2))`, que en float32 es cancel·la per a z
    # negatives: a z = -10000 dona menys infinit.
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
    # `a + b * asinh(z)`, que és la mateixa parametrització que `_johnson_loss`.
    location, scale = parameters[..., 0], parameters[..., 1]
    skew, tail = parameters[..., 2], parameters[..., 3]
    return stats.johnsonsu.ppf(
        probabilities,
        skew[..., None],
        tail[..., None],
        loc=location[..., None],
        scale=scale[..., None],
    )


def _normal_loss(target, raw):
    """Torna la log-versemblança negativa de la normal, llevat de constants."""
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


# El preu puntual de cada una és la mediana i no la mitjana, com a l'article per al MAE.
JOHNSON_SU = Distribution(
    names=("location", "scale", "skewness", "tail_weight"),
    loss=_johnson_loss,
    flat=_johnson_flat,
    point=lambda parameters: _johnson_quantiles(parameters, 0.5)[..., 0],
    quantiles=_johnson_quantiles,
)

NORMAL = Distribution(
    names=("location", "scale"),
    loss=_normal_loss,
    flat=_normal_flat,
    # La mediana d'una normal és la seva localització.
    point=lambda parameters: parameters[..., 0],
    quantiles=_normal_quantiles,
)

# Com es diu cada família i quina distribució treu.
DISTRIBUTIONS = {"ddnn_jsu": JOHNSON_SU, "ddnn_normal": NORMAL}
