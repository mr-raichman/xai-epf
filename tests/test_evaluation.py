"""La guarda que el preu real ha de passar abans d'una mesura."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.evaluation import refuse_missing


def frame_of(days, start=datetime.date(2024, 1, 1), offset=0.0):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[d * 100 + hour + offset for hour in range(24)] for d in range(days)]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_a_missing_value_fails_naming_where_it_is():
    # Una mitjana de pandas salta el NaN i informa l'hora com si fos neta: aquest és el
    # cas que la guarda existeix per agafar.
    holed = frame_of(30, offset=3.0)
    holed.iloc[4, 0] = np.nan
    with pytest.raises(ValueError, match="h00"):
        refuse_missing(holed, "el preu real")
