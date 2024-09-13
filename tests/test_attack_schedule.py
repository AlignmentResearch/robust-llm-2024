import pytest

from robust_llm.config.configs import AttackScheduleConfig
from robust_llm.training import AttackSchedule


def test_attack_rounds():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=5)
    assert schedule.attack_rounds == 3

    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=1), num_rounds=2)
    assert schedule.attack_rounds == 0


def test_init_with_end_and_rate():
    schedule = AttackSchedule(AttackScheduleConfig(end=10, rate=2), num_rounds=5)
    assert schedule.start == 4
    assert schedule.end == 10
    assert schedule.rate == 2


def test_init_with_start_and_rate():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, rate=2), num_rounds=5)
    assert schedule.start == 1
    assert schedule.end == 7
    assert schedule.rate == 2


def test_init_with_start_and_end():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=5)
    assert schedule.start == 1
    assert schedule.end == 10
    assert pytest.approx(schedule.rate) == 3


def test_init_with_insufficient_params():
    with pytest.raises(
        AssertionError, match="Exactly two of start, end, and rate must be specified."
    ):
        AttackSchedule(AttackScheduleConfig(start=1), num_rounds=5)


def test_init_with_num_rounds_less_than_3():
    schedule = AttackSchedule(AttackScheduleConfig(start=5, end=5), num_rounds=2)
    assert schedule.start == 5
    assert schedule.end == 5
    assert schedule.rate == 0

    with pytest.raises(ValueError, match="If num_rounds<=2, rate must be 0."):
        AttackSchedule(AttackScheduleConfig(end=5, rate=1), num_rounds=2)

    with pytest.raises(ValueError, match="If num_rounds<=2, start must equal end."):
        AttackSchedule(AttackScheduleConfig(start=1, end=5), num_rounds=2)


def test_getitem():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=5)
    assert schedule[0] == 1
    assert schedule[1] == 4
    assert schedule[2] == 7
    assert schedule[3] == 10

    with pytest.raises(IndexError):
        schedule[-1]

    with pytest.raises(IndexError):
        schedule[4]


def test_edge_cases():
    # Test with num_rounds = 2
    schedule = AttackSchedule(AttackScheduleConfig(start=5, end=5), num_rounds=2)
    assert schedule.start == 5
    assert schedule.end == 5
    assert schedule.rate == 0

    # Test with num_rounds = 3
    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=3)
    assert schedule.start == 1
    assert schedule.end == 10
    assert schedule.rate == 9

    # Test with zero rate
    schedule = AttackSchedule(AttackScheduleConfig(start=5, rate=0), num_rounds=5)
    assert schedule.start == 5
    assert schedule.end == 5
    assert schedule.rate == 0
