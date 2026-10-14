"""Que l'arbre torni a fer els mateixos arbres amb les mateixes files.

Sense `deterministic` i `force_row_wise`, dos ajustos d'una finestra donaven models
diferents, i la predicció no ho deixava veure: l'atribució es movia i prou. Una taula
petita no ho destapa mai, de manera que la de la prova té la forma de la finestra on es
va veure, 1085 dies de 24 hores i 98 columnes.
"""
import numpy as np

from epf.models.boosting.boosting import fit_rows, predict_rows
from epf.models.boosting.boosting_hyperparameters import UNSEARCHED

ROWS, COLUMNS, TREES = 1085 * 24, 98, 200


def test_two_fits_on_the_same_rows_make_the_same_trees():
    # Sense les dues banderes, aquesta prova fallava 8 vegades de 10 (mesurat el
    # 2026-10-14 amb la màquina en repòs).
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(ROWS, COLUMNS)), axis=0)
    y = X[:, :10].sum(axis=1) + rng.normal(scale=5, size=ROWS)
    first, second = (fit_rows(X, y, UNSEARCHED, TREES) for _ in range(2))
    assert np.array_equal(predict_rows(first, X[:500]), predict_rows(second, X[:500]))
