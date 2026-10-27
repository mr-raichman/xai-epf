import datetime

import numpy as np
import pandas as pd

from epf import parameters
from epf.contract.split import NO_WHOLE_WEEK, _deal, deleted, split, validation_parts
from epf.information import resolved
from epf.parameters import DEVELOPMENT, SEEDS, SHARES

ONE_DAY = pd.Timedelta(days=1)

# La setmana de l'apagada, l'única que l'entrada d'ara suprimeix del període, i el gap
# d'ara.
BLACKOUT_WEEK = tuple(pd.date_range("2025-04-29", "2025-05-05", freq="D").date)
GAP = resolved("WITH_PRICE").gap


def test_each_part_is_dealt_its_count_of_weeks():
    # 403 setmanes: la validació en múltiples de quatre per a les quatre parts de
    # l'arbre, la prova arrodonida i l'entrenament la resta.
    for seed in SEEDS:
        _, dealt = _deal(seed, DEVELOPMENT, BLACKOUT_WEEK, SHARES)
        assert [len(weeks) for weeks in dealt] == [194, 128, 81]


def test_the_held_out_parts_are_whole_weeks_in_their_shares():
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK, GAP)
        assert (len(parts.validation), len(parts.test)) == (128 * 7, 81 * 7)
        for days in (parts.validation, parts.test):
            # Una setmana va de dimarts a dilluns: cada dimarts retingut porta els sis
            # dies següents a la mateixa part.
            tuesdays = days[days.dayofweek == 1]
            assert len(days) == 7 * len(tuesdays)
            assert all(
                day + offset * ONE_DAY in days
                for day in tuesdays
                for offset in range(7)
            )


def test_no_day_is_in_two_parts_and_only_a_day_beside_a_held_out_week_is_in_none():
    every_day = pd.date_range(*DEVELOPMENT, freq="D")
    gone = pd.DatetimeIndex(BLACKOUT_WEEK)
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK, GAP)
        held_out = parts.validation.union(parts.test)
        assert parts.validation.intersection(parts.test).empty
        assert parts.training.intersection(held_out).empty
        for day in every_day.difference(parts.training.union(held_out)).difference(
            gone
        ):
            assert day - ONE_DAY in held_out or day + ONE_DAY in held_out


def test_the_deleted_week_is_in_no_part_and_the_days_beside_it_are_dealt_as_any_other():
    """La setmana suprimida no compta com a retinguda.

    Si ho fes, el dilluns abans i el dimarts després, que comparteixen setmana amb els
    seus altres veïns, sortirien de l'entrenament sense cap setmana retinguda al costat.
    """
    gone = pd.DatetimeIndex(BLACKOUT_WEEK)
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK, GAP)
        every_part = parts.training.union(parts.validation).union(parts.test)
        assert every_part.intersection(gone).empty
        assert {gone[0] - ONE_DAY, gone[-1] + ONE_DAY} <= set(every_part)


def test_no_training_day_is_next_to_a_held_out_day():
    for seed in SEEDS:
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK, GAP)
        held_out = parts.validation.union(parts.test)
        assert not parts.training.shift(1, freq="D").isin(held_out).any()
        assert not parts.training.shift(-1, freq="D").isin(held_out).any()


def test_the_same_seed_gives_the_same_split_and_another_seed_another():
    first, again, other = (
        split(seed, DEVELOPMENT, BLACKOUT_WEEK, GAP) for seed in (0, 0, 1)
    )
    assert all(a.equals(b) for a, b in zip(first, again, strict=True))
    assert not first.test.equals(other.test)


def test_a_seeds_deal_changes_with_PROJECT_SEED(monkeypatch):
    """La partició es barrejava amb la seed sola, igual amb qualsevol `PROJECT_SEED`."""
    first, again = (_deal(0, DEVELOPMENT, BLACKOUT_WEEK, SHARES)[1] for _ in range(2))
    monkeypatch.setattr(parameters, "PROJECT_SEED", 1)
    _, other = _deal(0, DEVELOPMENT, BLACKOUT_WEEK, SHARES)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not np.array_equal(first[0], other[0])


def test_the_last_days_that_make_no_whole_week_are_deleted_and_in_no_part():
    """Les setmanes de calendari deixaven una primera de sis dies i una última d'un."""
    # Del dimarts 2019-01-01 al diumenge 2024-09-29: sobren els sis dies de l'última.
    period = (datetime.date(2019, 1, 1), datetime.date(2024, 9, 29))
    whole = (period[0], datetime.date(2024, 9, 23))
    assert deleted(period, {}) == {
        day: NO_WHOLE_WEEK for day in pd.date_range("2024-09-24", period[1]).date
    }
    for seed in (0, 1):
        assert all(
            a.equals(b)
            for a, b in zip(
                split(seed, period, BLACKOUT_WEEK, GAP),
                split(seed, whole, BLACKOUT_WEEK, GAP),
                strict=True,
            )
        )


def test_the_validation_parts_are_whole_weeks_that_share_out_the_validation_days():
    for seed in SEEDS:
        parts = validation_parts(seed, DEVELOPMENT, BLACKOUT_WEEK)
        assert [len(part) for part in parts] == [32 * 7] * 4
        joined = parts[0].append(parts[1:]).sort_values()
        assert joined.is_unique
        assert joined.equals(split(seed, DEVELOPMENT, BLACKOUT_WEEK, GAP).validation)
