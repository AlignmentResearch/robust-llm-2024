"""Test that the various defenses don't crash."""

import pytest

from robust_llm.config import (
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RandomTokenAttackConfig,
)
from robust_llm.config.configs import EvaluationConfig
from robust_llm.config.defense_configs import (
    ParaphraseDefenseConfig,
    PerplexityDefenseConfig,
    RetokenizationDefenseConfig,
)
from robust_llm.pipelines.evaluation_pipeline import run_evaluation_pipeline


@pytest.fixture
def exp_config() -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(
            evaluation_attack=RandomTokenAttackConfig(
                min_tokens=2,
                max_tokens=3,
                max_iterations=2,
            )
        ),
        model=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            n_train=2,
            n_val=2,
        ),
    )
    return config


def test_doesnt_crash_perplexity_defense(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.defense = PerplexityDefenseConfig(
        decoder=ModelConfig(name_or_path="EleutherAI/pythia-14m", family="pythia"),
    )
    run_evaluation_pipeline(exp_config)


def test_doesnt_crash_retokenization_defense(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.defense = RetokenizationDefenseConfig()
    run_evaluation_pipeline(exp_config)


def test_doesnt_crash_paraphrase_defense(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.defense = ParaphraseDefenseConfig(
        model_name="EleutherAI/pythia-14m", device="cpu"
    )
    run_evaluation_pipeline(exp_config)
