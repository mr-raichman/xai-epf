"""L'espai de la cerca de les xarxes: una L1 de paràmetre llegida per posició i no pel nom
penalitzaria el paràmetre equivocat sense que res petés."""
import optuna

from epf.models.ddnn import search as ddnn_search
from epf.models.ddnn.distributions import JOHNSON_SU


def test_each_parameters_l1_is_the_one_named_after_it():
    # El codi de l'article posa el pes de la cua de Johnson SU abans de l'asimetria: una
    # L1 presa per posició penalitzaria l'un amb la de l'altre.
    rates = {"location": 1e-4, "scale": 1e-3, "skewness": 1e-2, "tail_weight": 1e-1}
    fixed = {"use_lag1": True, "neurons_1": 32, "neurons_2": 16,
             "activation_1": "relu", "activation_2": "elu", "dropout": False,
             "regularize_h1_kernel": False, "regularize_h2_kernel": False,
             "regularize_h1_activation": False, "regularize_h2_activation": False,
             "learning_rate": 1e-3}
    fixed |= {f"regularize_{name}": True for name in rates}
    fixed |= {f"{name}_l1": rate for name, rate in rates.items()}
    tuned = ddnn_search.space(optuna.trial.FixedTrial(fixed), ["lag1"], JOHNSON_SU)
    assert tuned["parameter_l1"] == [rates[name] for name in JOHNSON_SU.names]
