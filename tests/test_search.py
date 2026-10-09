"""Els lots de la cerca: que cap dia que es puntua no entri a l'ajust que el prediu.

És l'esquema de Marcjasz et al. (2023): tretze reajustos sobre lots de 28 dies, amb la
finestra rodant 28 dies a cada pas. El que es pot trencar en silenci és la frontera.
Si un dia d'un lot entrés a la finestra del seu propi ajust, la puntuació d'una prova
sortiria massa bona, la cerca triaria els números que més hi filtren, i el model que en
sortís seria pitjor que el que diu el seu MAE sense que res petés.
"""
import datetime

import pandas as pd
import pytest

from epf.matrix import calibration
from epf.models.dnn.search import lots

DAYS = 500


def days(count=DAYS):
    return pd.DatetimeIndex([datetime.date(2023, 1, 1) + datetime.timedelta(days=day)
                             for day in range(count)], name="day")


def test_thirteen_lots_of_twenty_eight_cover_the_last_year():
    batches = lots(days())
    assert len(batches) == 13
    assert {len(batch) for batch in batches} == {28}
    assert batches[-1][-1] == days()[-1]
    assert sum(len(batch) for batch in batches) == 364


def test_the_lots_are_consecutive_and_do_not_overlap():
    batches = lots(days())
    joined = pd.DatetimeIndex([day for batch in batches for day in batch])
    assert joined.is_monotonic_increasing
    assert joined.is_unique
    assert (joined[1:] - joined[:-1]).unique().tolist() == [pd.Timedelta(days=1)]


def test_no_day_of_a_lot_is_in_the_window_that_predicts_it():
    # La finestra de cada lot acaba abans del seu primer dia, de manera que l'ajust
    # que prediu el dia 28 no ha vist cap dia del lot, ni el primer.
    index = days()
    frame = pd.DataFrame(1.0, index=index, columns=["a"])
    for batch in lots(index):
        features, _ = calibration(frame, frame, batch[0], window=200)
        assert features.index.max() < batch[0]
        assert not set(batch) & set(features.index)


def test_a_period_too_short_for_the_lots_fails_loudly():
    with pytest.raises(ValueError):
        lots(days(13 * 28 - 1))


def test_the_lots_end_at_the_period_and_do_not_start_at_it():
    # Es puntuen els dies més propers als de prova, de manera que sobren dies al
    # començament i no al final.
    index = days()
    batches = lots(index)
    assert batches[0][0] == index[-364]
    assert batches[0][0] > index[0]
