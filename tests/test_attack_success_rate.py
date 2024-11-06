"""Test that the various attacks don't crash."""

import pytest
import textattack.shared.utils

from robust_llm.__main__ import run
from robust_llm.config import (
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RandomTokenAttackConfig,
)
from robust_llm.config.attack_configs import GCGAttackConfig, MultipromptGCGAttackConfig
from robust_llm.config.configs import EvaluationConfig
from robust_llm.utils import interpolate_config


@pytest.fixture
def exp_config() -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(),
        model=ModelConfig(
            # We use a finetuned model so that the classification head isn't
            # randomly initalized.
            # TODO(ian): Update this to use our canonical models once trained.
            name_or_path="AlignmentResearch/robust_llm_pythia-14m_clf_pm_v-ian-068_s-0",
            family="pythia",
            inference_type="classification",
            strict_load=True,
            revision="98ffbf205cca9ddb27c200f10649e3e5fdb818f6",
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            revision="2.1.0",
            n_train=2,
            n_val=100,
        ),
    )
    return config


def _test_attack(
    exp_config: ExperimentConfig,
    success_rate_at_least: int = 0,
    exact_success_rate: int | None = None,
) -> int:
    """Small wrapper around run_evaluation_pipeline.

    - Resets a TextAttack global variable
    - Runs OmegaConf interpolation on the experiment config
    - Runs experiment once to check that the success rate meets a lower bound
    - Runs the experiment a second time to check that the success rate is consistent
    """
    # This is a global variable that needs to be reset between attacks
    # of different types because of a bug in TextAttack.
    # See GH#341 for more details.
    textattack.shared.utils.strings._flair_pos_tagger = None

    config = interpolate_config(exp_config)
    results = run(config)
    actual = int(results["adversarial_eval/n_incorrect_post_attack"])
    print(f"Success rate: {actual}")
    assert actual >= success_rate_at_least
    assert exact_success_rate is None or actual == exact_success_rate
    return actual


@pytest.mark.multigpu  # Used as a test of multi-GPU classification eval
def test_random_token_asr(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
    )
    exp_config.evaluation.num_iterations = 50
    _test_attack(exp_config, success_rate_at_least=18)


def test_multiprompt_random_token_asr(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
        prompt_attack_mode="multi-prompt",
    )
    exp_config.evaluation.num_iterations = 250
    _test_attack(exp_config, success_rate_at_least=45)


def test_gcg_asr(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=1,
        n_candidates_per_it=128,
    )
    exp_config.evaluation.num_iterations = 2

    _test_attack(exp_config, success_rate_at_least=28)


def test_multiprompt_gcg_asr(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptGCGAttackConfig(
        n_attack_tokens=5,
    )
    exp_config.evaluation.num_iterations = 10
    _test_attack(exp_config, success_rate_at_least=12)
