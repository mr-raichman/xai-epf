"""Correcció dels dies de canvi d'hora."""
import pytest

from epf.calendar import repair_dst

# 2025-03-30: 23 períodes, sense l'hora local 02. Primers cinc preus espanyols.
SPRING = [3.6, 3.2, 0.65, 0.1, 0.09]

# 2024-10-27: 25 períodes; el tercer i el quart són l'hora local 02 repetida.
AUTUMN = [87.71, 83.31, 82.23, 80.68, 81.3]


def test_a_normal_day_is_untouched():
    hours = list(range(24))
    assert repair_dst(hours) == hours


def test_spring_inserts_the_mean_of_the_neighbours():
    repaired = repair_dst(SPRING + [0.0] * 18)
    assert len(repaired) == 24
    assert repaired[:4] == [3.6, 3.2, pytest.approx((3.2 + 0.65) / 2), 0.65]


def test_autumn_merges_the_two_auctions_of_the_repeated_hour():
    repaired = repair_dst(AUTUMN + [0.0] * 20)
    assert len(repaired) == 24
    assert repaired[:4] == [87.71, 83.31, pytest.approx((82.23 + 80.68) / 2), 81.3]


def test_the_repair_only_moves_the_third_hour():
    day = [float(h) for h in range(23)]
    repaired = repair_dst(day)
    assert repaired[:2] == day[:2]
    assert repaired[3:] == day[2:]


def test_an_unknown_length_fails_loudly():
    with pytest.raises(ValueError):
        repair_dst([0.0] * 20)
