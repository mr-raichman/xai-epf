"""La partició, sobre el període i les llavors de debò."""
import datetime

import pandas as pd
import pytest

from epf.describe import split, validation_parts
from epf.parameters import DEVELOPMENT, SEEDS

ONE_DAY = pd.Timedelta(days=1)


def test_the_held_out_parts_are_whole_weeks_in_their_shares():
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT)
        assert (len(parts.validation), len(parts.test)) == (96 * 7, 60 * 7)
        for days in (parts.validation, parts.test):
            # Una setmana va de dimarts a dilluns: cada dimarts retingut porta els
            # sis dies següents a la mateixa part.
            tuesdays = days[days.dayofweek == 1]
            assert len(days) == 7 * len(tuesdays)
            assert all(day + offset * ONE_DAY in days
                       for day in tuesdays for offset in range(7))


def test_no_day_is_in_two_parts_and_only_a_day_beside_a_held_out_week_is_in_none():
    every_day = pd.date_range(*DEVELOPMENT, freq="D")
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT)
        held_out = parts.validation.union(parts.test)
        assert parts.validation.intersection(parts.test).empty
        assert parts.training.intersection(held_out).empty
        for day in every_day.difference(parts.training.union(held_out)):
            assert day - ONE_DAY in held_out or day + ONE_DAY in held_out


def test_no_training_day_is_next_to_a_held_out_day():
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT)
        held_out = parts.validation.union(parts.test)
        assert not parts.training.shift(1, freq="D").isin(held_out).any()
        assert not parts.training.shift(-1, freq="D").isin(held_out).any()


def test_the_same_seed_gives_the_same_split_and_another_seed_another():
    first, again, other = (split(seed, DEVELOPMENT) for seed in (0, 0, 1))
    assert all(a.equals(b) for a, b in zip(first, again))
    assert not first.test.equals(other.test)


def test_a_period_of_broken_weeks_is_refused():
    # Les setmanes de calendari deixarien una primera de sis dies i una última d'un.
    with pytest.raises(ValueError, match="setmanes senceres"):
        split(0, (datetime.date(2019, 1, 1), datetime.date(2024, 9, 29)))


def test_the_validation_parts_are_whole_weeks_that_share_out_the_validation_days():
    for seed in SEEDS:
        parts = validation_parts(seed, DEVELOPMENT)
        assert [len(part) for part in parts] == [24 * 7] * 4
        joined = parts[0].append(parts[1:]).sort_values()
        assert joined.is_unique
        assert joined.equals(split(seed, DEVELOPMENT).validation)
