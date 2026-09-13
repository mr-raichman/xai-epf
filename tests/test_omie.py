"""La reducció de períodes a hores, que és on viu la decisió 29."""
from epf.data.omie import to_hourly


def test_an_hourly_day_passes_through():
    day = list(range(24))
    assert to_hourly(day) == day


def test_the_two_daylight_saving_lengths_pass_through_unaveraged():
    """23 i 25 són dies horaris, no quart-horaris: reparar-los és feina del calendari."""
    assert len(to_hourly(list(range(23)))) == 23
    assert len(to_hourly(list(range(25)))) == 25


def test_ninety_six_periods_become_twenty_four_means():
    quarters = [float(i) for i in range(96)]
    hours = to_hourly(quarters)
    assert len(hours) == 24
    assert hours[0] == (0 + 1 + 2 + 3) / 4
    assert hours[23] == (92 + 93 + 94 + 95) / 4


def test_the_quarter_hourly_daylight_saving_days_keep_their_length():
    """92 períodes són 23 hores i 100 en són 25, que després repara el calendari."""
    assert len(to_hourly([0.0] * 92)) == 23
    assert len(to_hourly([0.0] * 100)) == 25
