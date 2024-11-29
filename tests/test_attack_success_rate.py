"""Test that the various attacks don't crash."""

import pytest

from robust_llm.__main__ import run
from robust_llm.config import (
    BeastAttackConfig,
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    GCGAttackConfig,
    ModelConfig,
    RandomTokenAttackConfig,
)
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

    - Runs OmegaConf interpolation on the experiment config
    - Runs experiment once to check that the success rate meets a lower bound
    - Runs the experiment a second time to check that the success rate is consistent
    """

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


def test_beast_asr(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = BeastAttackConfig(
        beam_search_width=4,
        beam_branch_factor=4,
        sampling_model=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type="generation",
            revision="main",
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=0.5,
        ),
    )
    exp_config.evaluation.num_iterations = 2

    _test_attack(exp_config, success_rate_at_least=22)
