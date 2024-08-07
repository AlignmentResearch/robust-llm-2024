"""Test that the various attacks don't crash."""

import pytest
from omegaconf import OmegaConf

from robust_llm.config import (
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RandomTokenAttackConfig,
    TRLAttackConfig,
)
from robust_llm.config.attack_configs import GCGAttackConfig, MultipromptGCGAttackConfig
from robust_llm.config.callback_configs import AutoregressiveCallbackConfig
from robust_llm.config.configs import EvaluationConfig
from robust_llm.config.model_configs import GenerationConfig
from robust_llm.pipelines.evaluation_pipeline import run_evaluation_pipeline

NON_MODIFIABLE_WORDS_TEXT_ATTACKS = [
    "textfooler",
    "checklist",
    "pso",
]

MODIFIABLE_WORDS_TEXT_ATTACKS = [
    "bae",
    "random_character_changes",
]


@pytest.fixture
def exp_config() -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(
            final_success_binary_callback=AutoregressiveCallbackConfig(
                callback_name="binary_bivariate_fn_of_generation_from_text",
                callback_return_type="binary",
                scoring_fn="doesnt_contain_target",
                scoring_fn_arity=2,
            ),
        ),
        model=ModelConfig(
            # We use a finetuned model so that pre-attack accuracy is non-zero and
            # so we have some actual examples to attack.
            name_or_path="AlignmentResearch/robust_llm_pythia-14m-pm-gen-ian-nd",
            family="pythia",
            strict_load=True,
            generation_config=GenerationConfig(),
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            n_train=2,
            n_val=2,
            revision="2.1.0",
            inference_type="generation",
            gen_target_override=" I HATE YOU",
            classification_as_generation=False,
        ),
    )
    return config


def _test_doesnt_crash(exp_config: ExperimentConfig) -> None:
    """Small wrapper around run_evaluation_pipeline.

    This runs interpolation on the config.
    - First we convert to an OmegaConf structured config, which enables
    interpolation.
    - Then we convert back to an ExperimentConfig object, and use
    that to run the pipeline.
    """
    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    run_evaluation_pipeline(config)


def test_doesnt_crash_autoregressive_gen_random_token(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=3,
        n_its=2,
        victim_success_callback=AutoregressiveCallbackConfig(
            callback_name="binary_bivariate_fn_of_generation_from_text",
            callback_return_type="binary",
            scoring_fn="doesnt_contain_target",
            scoring_fn_arity=2,
        ),
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_autoregressive_gen_multiprompt_random_token(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=3,
        n_its=2,
        prompt_attack_mode="multi-prompt",
        victim_success_callback=AutoregressiveCallbackConfig(
            callback_name="binary_bivariate_fn_of_generation_from_text",
            callback_return_type="binary",
            scoring_fn="doesnt_contain_target",
            scoring_fn_arity=2,
        ),
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_autoregressive_gen_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=2,
        n_its=2,
        n_candidates_per_it=16,
    )

    _test_doesnt_crash(exp_config)


def test_doesnt_crash_autoregressive_gen_multiprompt_gcg(
    exp_config: ExperimentConfig,
) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptGCGAttackConfig(
        n_attack_tokens=3,
        n_its=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_autoregressive_gen_trl(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = TRLAttackConfig(
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=1,
        max_new_tokens=2,
        model_save_path_prefix=None,
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            # Our inference type for the adversary is different because
            # we need a ForCausalLMWithValueHead model
            inference_type="trl",
            strict_load=False,
        ),
    )

    _test_doesnt_crash(exp_config)
