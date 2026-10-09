"""La partició de validació de la xarxa: que no es solapi amb l'entrenament.

És el que pot trencar-se en silenci. Si un dia caigués a les dues bandes, l'aturada
anticipada miraria dies amb què la xarxa ja s'ha entrenat, l'entrenament duraria més
del que toca i l'error de validació sortiria massa bo, sense que res petés.
"""
import numpy as np
import pytest

from epf.models.dnn.network import split
from epf.parameters import DAYS_PER_WEEK

COUNT = 1453
WEEKS = 42


def parts(count=COUNT, weeks=WEEKS, seed=0):
    return split(count, weeks, np.random.default_rng(seed))


def test_the_two_parts_are_disjoint_and_cover_the_window():
    train, validate = parts()
    assert set(train).isdisjoint(validate)
    assert sorted([*train, *validate]) == list(range(COUNT))


def test_validation_is_exactly_the_weeks_asked_for():
    _, validate = parts()
    assert len(validate) == WEEKS * DAYS_PER_WEEK


def test_the_validation_days_come_from_the_whole_window_and_not_its_tail():
    # L'article les tria barrejant tota la finestra, de manera que la xarxa s'entreni
    # també amb els dies més recents. Prenent-ne el tram final, el màxim seria
    # l'últim dia i el mínim el primer d'aquell tram.
    _, validate = parts()
    assert min(validate) < COUNT // 4
    assert max(validate) > 3 * COUNT // 4


def test_the_same_seed_gives_the_same_split():
    # Una execució s'ha de poder repetir bit a bit, i la partició hi entra.
    assert np.array_equal(parts()[1], parts()[1])
    assert not np.array_equal(parts(seed=0)[1], parts(seed=1)[1])


def test_a_window_too_short_to_split_fails_loudly():
    with pytest.raises(ValueError):
        parts(count=WEEKS * DAYS_PER_WEEK)
