import types

import numpy as np
import pandas as pd

from epf.models.naive.train import _fitted, previous_day, previous_week, similar_day
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


def test_the_main_copy_alone_keeps_its_errors_on_the_training_days():
    # Cada preu és dia * 100 + hora, i la còpia de la setmana anterior s'equivoca en 700
    # a cada hora, fora del dia 12, 70 més car, que dona 770, i del 13, 1400 més barat,
    # que dona -700. MAE (3 * 700 + 770) / 4 = 717.5; MSE (3 * 490000 + 592900) / 4 =
    # 515725. Els dies de prova no hi compten.
    price = frame_of(range(30)).astype(float)
    price.iloc[12] += 70
    price.iloc[13] -= 1400
    parts = types.SimpleNamespace(training=price.index[10:14], test=price.index[20:22])
    found = _fitted("naive_previous_week", price, price, parts)
    assert found.arrays == {"training_mae": 717.5, "training_mse": 515725.0}
    assert _fitted("naive_previous_day", price, price, parts).arrays == {}
