import datetime

import pandas as pd
import pytest

from epf.contract.split import _deal, split, validation_parts
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, SEEDS, SHARES

ONE_DAY = pd.Timedelta(days=1)


def test_each_part_is_dealt_its_count_of_weeks():
    # 403 setmanes: la validació en múltiples de quatre per a les quatre parts de
    # l'arbre, la prova arrodonida i l'entrenament la resta.
    for seed in SEEDS:
        _, dealt = _deal(seed, DEVELOPMENT, BLACKOUT_WEEK, SHARES)
        assert [len(weeks) for weeks in dealt] == [194, 128, 81]


def test_the_held_out_parts_are_whole_weeks_in_their_shares():
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        assert (len(parts.validation), len(parts.test)) == (128 * 7, 81 * 7)
        for days in (parts.validation, parts.test):
            # Una setmana va de dimarts a dilluns: cada dimarts retingut porta els
            # sis dies següents a la mateixa part.
            tuesdays = days[days.dayofweek == 1]
            assert len(days) == 7 * len(tuesdays)
            assert all(
                day + offset * ONE_DAY in days
                for day in tuesdays
                for offset in range(7)
            )


def test_no_day_is_in_two_parts_and_only_a_day_beside_a_held_out_week_is_in_none():
    every_day = pd.date_range(*DEVELOPMENT, freq="D")
    deleted = pd.date_range(*BLACKOUT_WEEK, freq="D")
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        held_out = parts.validation.union(parts.test)
        assert parts.validation.intersection(parts.test).empty
        assert parts.training.intersection(held_out).empty
        for day in every_day.difference(parts.training.union(held_out)).difference(
            deleted
        ):
            assert day - ONE_DAY in held_out or day + ONE_DAY in held_out


def test_the_deleted_week_is_in_no_part_and_the_days_beside_it_are_dealt_as_any_other():
    """La setmana suprimida no compta com a retinguda.

    Si ho fes, el dilluns abans i el dimarts després, que comparteixen setmana amb els
    seus altres veïns, sortirien de l'entrenament sense cap setmana retinguda al costat.
    """
    deleted = pd.date_range(*BLACKOUT_WEEK, freq="D")
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        every_part = parts.training.union(parts.validation).union(parts.test)
        assert every_part.intersection(deleted).empty
        assert {deleted[0] - ONE_DAY, deleted[-1] + ONE_DAY} <= set(every_part)


def test_no_training_day_is_next_to_a_held_out_day():
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        held_out = parts.validation.union(parts.test)
        assert not parts.training.shift(1, freq="D").isin(held_out).any()
        assert not parts.training.shift(-1, freq="D").isin(held_out).any()


def test_the_same_seed_gives_the_same_split_and_another_seed_another():
    first, again, other = (
        split(seed, DEVELOPMENT, BLACKOUT_WEEK) for seed in (0, 0, 1)
    )
    assert all(a.equals(b) for a, b in zip(first, again, strict=True))
    assert not first.test.equals(other.test)


def test_a_period_of_broken_weeks_is_refused():
    # Les setmanes de calendari deixarien una primera de sis dies i una última d'un.
    with pytest.raises(ValueError, match="setmanes senceres"):
        split(0, (datetime.date(2019, 1, 1), datetime.date(2024, 9, 29)), BLACKOUT_WEEK)


def test_the_validation_parts_are_whole_weeks_that_share_out_the_validation_days():
    for seed in SEEDS:
        parts = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
        assert [len(part) for part in parts] == [32 * 7] * 4
        joined = parts[0].append(parts[1:]).sort_values()
        assert joined.is_unique
        assert joined.equals(split(seed, DEVELOPMENT, BLACKOUT_WEEK).validation)
