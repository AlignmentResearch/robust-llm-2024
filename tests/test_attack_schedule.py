import pytest

from robust_llm.config.configs import (
    AdversarialTrainingConfig,
    AttackScheduleConfig,
    EnvironmentConfig,
    ExperimentConfig,
    TrainingConfig,
)
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.training.training_utils import AttackSchedule
from robust_llm.utils import interpolate_config


def test_init_with_end_and_rate():
    schedule = AttackSchedule(AttackScheduleConfig(end=10, rate=2), num_rounds=4)
    assert schedule.start == 4
    assert schedule.end == 10
    assert schedule.rate == 2


def test_init_with_start_and_rate():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, rate=2), num_rounds=4)
    assert schedule.start == 1
    assert schedule.end == 7
    assert schedule.rate == 2


def test_init_with_start_and_end():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=4)
    assert schedule.start == 1
    assert schedule.end == 10
    assert pytest.approx(schedule.rate) == 3


def test_init_with_too_many_params():
    with pytest.raises(AssertionError):
        AttackSchedule(AttackScheduleConfig(start=1, end=2, rate=0.1), num_rounds=4)


def test_init_with_num_rounds_less_than_2():
    schedule = AttackSchedule(AttackScheduleConfig(start=5, end=5), num_rounds=1)
    assert schedule.start == 5
    assert schedule.end == 5
    assert schedule.rate == 0

    with pytest.raises(ValueError, match="If num_rounds<=1, rate must be 0."):
        print(AttackSchedule(AttackScheduleConfig(end=5, rate=1), num_rounds=1))

    with pytest.raises(ValueError, match="If num_rounds<=1, start must equal end."):
        AttackSchedule(AttackScheduleConfig(start=1, end=5), num_rounds=1)


def test_getitem():
    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=4)
    assert schedule[0] == 1
    assert schedule[1] == 4
    assert schedule[2] == 7
    assert schedule[3] == 10

    with pytest.raises(IndexError):
        schedule[-1]

    with pytest.raises(IndexError):
        schedule[4]


def test_edge_cases():
    schedule = AttackSchedule(AttackScheduleConfig(start=5, end=5), num_rounds=1)
    assert schedule.start == 5
    assert schedule.end == 5
    assert schedule.rate == 0

    schedule = AttackSchedule(AttackScheduleConfig(start=1, end=10), num_rounds=2)
    assert schedule.start == 1
    assert schedule.end == 10
    assert schedule.rate == 9

    # Test with zero rate
    schedule = AttackSchedule(AttackScheduleConfig(start=5, rate=0), num_rounds=5)
    assert schedule.start == 5
    assert schedule.end == 5
    assert schedule.rate == 0


@pytest.mark.parametrize(
    "start, end, rate",
    [
        (1, 10, None),
        (1, None, 3),
        (None, 10, 3),
        (1, None, None),
        (None, 10, None),
    ],
)
def test_schedule_config(start: int | None, end: int | None, rate: float | None):
    exp_config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        training=TrainingConfig(
            adversarial=AdversarialTrainingConfig(
                attack_schedule=AttackScheduleConfig(start=start, end=end, rate=rate),
            ),
        ),
        model=ModelConfig(
            name_or_path="AlignmentResearch/robust_llm_pythia-14m_clf_pm_v-ian-068_s-0",
            family="pythia",
            inference_type="classification",
            strict_load=True,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            revision="2.1.0",
            n_train=2,
            n_val=2,
        ),
    )
    interpolated = interpolate_config(exp_config)
    assert isinstance(interpolated, ExperimentConfig)
    assert interpolated.training is not None
    assert interpolated.training.adversarial is not None
    AttackSchedule(
        config=interpolated.training.adversarial.attack_schedule, num_rounds=4
    )
