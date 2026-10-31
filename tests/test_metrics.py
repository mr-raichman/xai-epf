import warnings

import numpy as np
import pytest

from epf import Refusal
from epf.contract.metrics import part_i_main, scored


def test_a_score_refuses_a_prediction_that_is_not_a_number_and_fails_a_missing_price():
    with pytest.raises(Refusal, match="la predicció té 2 valors no finits"):
        scored([1.0, np.nan, np.inf], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="el preu real té 1 valor absent") as raised:
        scored([1.0, 2.0, 3.0], [1.0, np.nan, 3.0])
    assert not isinstance(raised.value, Refusal)


def _values(rows, measure):
    """Torna el valor de cada fila d'una mesura, per família, seed i lloc."""
    return {
        (row["family"], row.get("seed"), row.get("rank")): row["value"]
        for row in rows
        if row["measure"] == measure
    }


def test_the_main_metrics_divide_by_the_copy_and_summarise_each_seed_s_best():
    # L'arbre a la seed 0, MAE 10 i 12, i a la 1, 9; la còpia, 20 i 18. rMAE 10 / 20 =
    # 0.5, 12 / 20 = 0.6 i 9 / 18 = 0.5. Sobre les millors, el MAE de l'arbre té mitjana
    # 9.5 i desviació mostral √((0.5² + 0.5²) / 1) = √0.5, i l'rMAE, 0.5 i 0.
    scores = {
        "lightgbm": {
            (0, 0): {"r2": 0.5, "mae": 10.0},
            (0, 1): {"r2": 0.25, "mae": 12.0},
            (1, 0): {"r2": 0.75, "mae": 9.0},
        },
        "naive_previous_week": {
            (0, 0): {"r2": 0.0, "mae": 20.0},
            (1, 0): {"r2": 0.1, "mae": 18.0},
        },
    }
    rows = part_i_main(scores, "naive_previous_week", ["lightgbm"])
    assert _values(rows, "rmae") == {
        ("lightgbm", 0, 0): 0.5,
        ("lightgbm", 0, 1): 0.6,
        ("lightgbm", 1, 0): 0.5,
        ("naive_previous_week", 0, 0): 1.0,
        ("naive_previous_week", 1, 0): 1.0,
    }
    assert _values(rows, "mae_mean") == {("lightgbm", None, 0): 9.5}
    assert _values(rows, "mae_std") == {("lightgbm", None, 0): pytest.approx(0.5**0.5)}
    assert _values(rows, "rmae_std") == {("lightgbm", None, 0): 0.0}
    assert {row["other_family"] for row in rows if "rmae" in row["measure"]} == {
        "naive_previous_week"
    }


def test_a_seed_without_the_copy_has_no_rmae_nor_its_family_a_summary_of_it():
    scores = {
        "lightgbm": {(0, 0): {"r2": 0.5, "mae": 10.0}, (1, 0): {"r2": 0.7, "mae": 9.0}},
        "naive_previous_week": {(0, 0): {"r2": 0.0, "mae": 20.0}},
    }
    rows = part_i_main(scores, "naive_previous_week", ["lightgbm"])
    assert _values(rows, "rmae") == {
        ("lightgbm", 0, 0): 0.5,
        ("naive_previous_week", 0, 0): 1.0,
    }
    assert not _values(rows, "rmae_mean") and _values(rows, "mae_mean")


def test_one_seed_s_spread_is_a_blank_value_with_no_warning():
    scores = {"floor": {(0, 0): {"r2": 0.5, "mae": 10.0}}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = part_i_main(scores, None, ["floor"])
    assert np.isnan(_values(rows, "mae_std")[("floor", None, 0)])
    assert not _values(rows, "rmae")
