import numpy as np

from epf.models.lightgbm.hyperparameters import UNSEARCHED
from epf.models.lightgbm.train import fit_rows, predict_rows

# La forma de la finestra on dos ajustos van donar arbres diferents, 1085 dies de 24
# hores i 98 columnes: una matriu petita no ho destapa mai.
ROWS, COLUMNS, TREES = 1085 * 24, 98, 200


def test_two_fits_on_the_same_rows_make_the_same_trees():
    """Sense `deterministic` i `force_row_wise`, fallava 8 vegades de 10 en repòs."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(ROWS, COLUMNS)), axis=0)
    y = X[:, :10].sum(axis=1) + rng.normal(scale=5, size=ROWS)
    first, second = (fit_rows(X, y, UNSEARCHED, TREES) for _ in range(2))
    assert np.array_equal(predict_rows(first, X[:500]), predict_rows(second, X[:500]))
