"""L'espai de cerca de les xarxes.

No peta quan falla: una L1 de paràmetre llegida per posició penalitzaria el paràmetre
equivocat.
"""
import optuna

from epf.models.ddnn import search as ddnn_search
from epf.models.ddnn.distributions import JOHNSON_SU, POINT

# Una xarxa de l'espai, sense cap L1 als pesos ni a les sortides de les capes amagades.
FIXED = {"neurons_1": 32, "neurons_2": 16, "activation_1": "relu", "activation_2": "elu",
         "dropout": False, "regularize_h1_kernel": False, "regularize_h2_kernel": False,
         "regularize_h1_activation": False, "regularize_h2_activation": False,
         "learning_rate": 1e-3}


def test_each_parameters_l1_is_the_one_named_after_it():
    # El codi de l'article posa el pes de la cua de Johnson SU abans de l'asimetria: una
    # L1 presa per posició penalitzaria l'un amb la de l'altre.
    rates = {"location": 1e-4, "scale": 1e-3, "skewness": 1e-2, "tail_weight": 1e-1}
    fixed = FIXED | {f"regularize_{name}": True for name in rates}
    fixed |= {f"{name}_l1": rate for name, rate in rates.items()}
    tuned = ddnn_search.space(optuna.trial.FixedTrial(fixed), JOHNSON_SU)
    assert tuned["parameter_l1"] == [rates[name] for name in JOHNSON_SU.names]


def test_the_point_networks_space_asks_for_no_output_penalty():
    # La xarxa de punt de l'article no té cap L1 a la sortida: si se'n demanés una,
    # `FixedTrial` petaria per un paràmetre que no té.
    tuned = ddnn_search.space(optuna.trial.FixedTrial(FIXED), POINT, penalised=False)
    assert tuned["parameter_l1"] == [0.0]
