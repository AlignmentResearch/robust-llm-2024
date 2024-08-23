"""Run experiments on several small models to make sure they don't crash.

Starting with Gemma.
TODO(GH#628): Add tests for more models.
"""

from functools import cache
from typing import Optional
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from robust_llm.config.attack_configs import GCGAttackConfig
from robust_llm.config.callback_configs import AutoregressiveCallbackConfig
from robust_llm.config.configs import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.pipelines.evaluation_pipeline import run_evaluation_pipeline


@cache
def gemma_1p1_config():
    config = AutoConfig.from_pretrained("google/gemma-1.1-2b-it")
    config.intermediate_size = 20
    config.hidden_size = 128
    config.num_hidden_layers = 3
    return config


@cache
def gemma_2_config():
    config = AutoConfig.from_pretrained("google/gemma-2-9b-it")
    config.intermediate_size = 20
    config.hidden_size = 128
    config.num_hidden_layers = 3
    return config


@cache
def model_configs() -> dict[str, AutoConfig]:
    # We make this a cached function rather than a global variable because if it
    # were global, we'd call the gemma config functions and make slow network
    # requests during pytest collection even if no tests in this file run.
    return {
        "google/gemma-1.1-2b-it": gemma_1p1_config(),
        "google/gemma-2-9b-it": gemma_2_config(),
    }


def mock_load_hf_model(
    name_or_path: str,
    revision: str,
    inference_type: InferenceType,
    strict_load: bool,
    torch_dtype: torch.dtype,
    attention_implementation: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> PreTrainedModel:
    """Mock load_hf_model function.

    This function is used to test the WrappedModel class without actually
    loading a model from Hugging Face. Instead we just use a randomly
    initialized small model by getting a config.
    """
    config = model_configs()[name_or_path]
    print("USING MOCK LOAD HF MODEL")
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture
def exp_config() -> ExperimentConfig:
    target_not_in_gen_callback = AutoregressiveCallbackConfig(
        callback_name="binary_univariate_fn_of_generation_from_text",
        callback_return_type="binary",
        scoring_fn="doesnt_contain_ihateyou",
        scoring_fn_arity=1,
    )

    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(
            evaluation_attack=GCGAttackConfig(
                n_attack_tokens=2,
                initial_n_its=2,
                n_candidates_per_it=128,
            ),
            final_success_binary_callback=target_not_in_gen_callback,
        ),
        model=None,  # type: ignore  # We'll set this in the test.
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            n_train=2,
            n_val=2,
            revision="2.1.0",
            inference_type="generation",
            classification_as_generation=False,
        ),
    )
    return config


def _test_attack(exp_config: ExperimentConfig) -> None:
    """Small wrapper around run_evaluation_pipeline.

    - First we convert to an OmegaConf structured config, which enables
    interpolation.
    - Then we convert back to an ExperimentConfig object, and use
    that to run the pipeline.
    """

    # Patch the load_hf_model function to return a small model.
    with patch("robust_llm.models.wrapped_model.load_hf_model", mock_load_hf_model):
        config = OmegaConf.to_object(OmegaConf.structured(exp_config))
        assert isinstance(config, ExperimentConfig)
        run_evaluation_pipeline(config)


def test_gemma1p1(exp_config: ExperimentConfig):
    exp_config.model = ModelConfig(
        name_or_path="google/gemma-1.1-2b-it",
        family="gemma-chat",
        inference_type="generation",
        strict_load=True,
        generation_config=GenerationConfig(),
    )
    _test_attack(exp_config)


@pytest.mark.skip("TODO(GH#629): Fix caching in Gemma2 once fixed upstream")
def test_gemma2(exp_config: ExperimentConfig):
    exp_config.model = ModelConfig(
        name_or_path="google/gemma-2-9b-it",
        family="gemma-chat",
        inference_type="generation",
        strict_load=True,
        generation_config=GenerationConfig(),
    )
    _test_attack(exp_config)
