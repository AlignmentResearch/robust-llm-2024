"""Test that the various defenses don't crash."""

import pytest
from omegaconf import OmegaConf

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
                n_attack_tokens=3,
                initial_n_its=2,
            )
        ),
        model=ModelConfig(
            # We use a finetuned model so that the classification head isn't
            # randomly initalized.
            # TODO(ian): Update this to use our canonical models once trained.
            name_or_path="AlignmentResearch/robust_llm_pythia-14m_clf_pm_v-ian-068_s-0",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
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
    return config


def _run_evaluation_pipeline(exp_config: ExperimentConfig) -> None:
    """Small wrapper around run_evaluation_pipeline to run interpolation.

    - First we convert to an OmegaConf structured config, which enables
    interpolation.
    - Then we convert back to an ExperimentConfig object, and use
    that to run the pipeline.
    """
    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    run_evaluation_pipeline(config)


def test_doesnt_crash_perplexity_defense(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.defense = PerplexityDefenseConfig(
        decoder=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type="generation",
        ),
    )
    _run_evaluation_pipeline(exp_config)


def test_doesnt_crash_retokenization_defense(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.defense = RetokenizationDefenseConfig()
    _run_evaluation_pipeline(exp_config)


def test_doesnt_crash_paraphrase_defense(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.defense = ParaphraseDefenseConfig(
        paraphraser=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type="generation",
            strict_load=True,
        ),
        num_preparation_examples=2,
    )
    _run_evaluation_pipeline(exp_config)
