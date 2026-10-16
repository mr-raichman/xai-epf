"""Cada predicció ingènua copia de la sèrie del preu el dia que li toca, i la del dia
similar porta la distribució dels seus errors hora a hora."""

import numpy as np
import pandas as pd

from epf.models.naive.train import (
    bootstrapped,
    previous_day,
    previous_week,
    similar_day,
)
from test_matrix import frame_of


def test_each_copy_reads_its_own_day_the_weekly_one_seven_back():
    price = frame_of(range(30))
    days = price.index[10:20]
    for forecast, back in ((previous_day, 1), (previous_week, 7)):
        expected = price.loc[days - pd.Timedelta(days=back)].to_numpy()
        assert np.array_equal(forecast(price, days), expected)


def test_the_similar_day_takes_yesterday_only_from_tuesday_to_friday():
    price = frame_of(range(30))
    days = price.index[10:24]
    for row, day in zip(similar_day(price, days), days):
        back = 1 if day.dayofweek in (1, 2, 3, 4) else 7
        assert np.array_equal(row, price.loc[day - pd.Timedelta(days=back)].to_numpy())


def test_each_hours_bootstrap_is_its_copy_plus_its_own_errors_quantiles():
    # L'hora `h` té els errors 1 a 10 més `100 h`, de manera que un quantil pres d'una
    # altra hora, o de totes juntes, en surt lluny. La inversa de la distribució
    # empírica dóna al quantil 0,25 el tercer error, i al 0,5 el cinquè.
    errors = np.arange(1, 11)[:, None] + 100 * np.arange(24)[None, :]
    copied = np.full((2, 24), 1000.0)
    found = bootstrapped(copied, errors, (0.25, 0.5))
    assert found.shape == (2, 24, 2)
    assert np.array_equal(found[0, :, 0], 1000 + 3 + 100 * np.arange(24))
    assert np.array_equal(found[1, :, 1], 1000 + 5 + 100 * np.arange(24))
