import numpy as np
import pandas as pd

from epf.models.naive.train import previous_day, previous_week, similar_day
from test_matrix import frame_of


def test_the_similar_day_takes_yesterday_only_from_tuesday_to_friday():
    price = frame_of(range(30))
    days = price.index[10:24]
    for row, day in zip(similar_day(price, days), days, strict=True):
        back = 1 if day.dayofweek in (1, 2, 3, 4) else 7
        assert np.array_equal(row, price.loc[day - pd.Timedelta(days=back)].to_numpy())


def test_each_copy_reads_its_own_day_the_weekly_one_seven_back():
    price = frame_of(range(30))
    days = price.index[10:20]
    for forecast, back in ((previous_day, 1), (previous_week, 7)):
        expected = price.loc[days - pd.Timedelta(days=back)].to_numpy()
        assert np.array_equal(forecast(price, days), expected)
