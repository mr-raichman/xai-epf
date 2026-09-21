import numpy as np

from epf.data.outliers import hampel_score


def test_a_lone_value_in_a_flat_run_scores_against_the_floor():
    # Forma de les hores a 0.00 del 2014, amb un valor diferent al mig.
    values = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    score = hampel_score(values, 3)
    assert np.isnan(score[:3]).all() and np.isnan(score[4:]).all()
    assert score[3] == 5.0 / 0.01


def test_the_score_is_aligned_with_the_centre_of_its_window():
    values = [1.0, 2.0, 3.0, 40.0, 5.0, 6.0, 7.0, 8.0]
    score = hampel_score(values, 1)
    assert np.nanargmax(score) == 3
