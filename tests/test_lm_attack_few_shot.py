from functools import partial
from unittest.mock import patch

import pytest
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf

from robust_llm.attacks.search_free.lm_attack_few_shot import FewShotLMAttack
from robust_llm.config.attack_configs import FewShotLMAttackConfig
from robust_llm.config.callback_configs import AutoregressiveCallbackConfig
from robust_llm.config.configs import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.pipelines.evaluation_pipeline import do_adversarial_evaluation
from robust_llm.pipelines.utils import prepare_attack
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.scoring_callbacks import CallbackRegistry
from robust_llm.scoring_callbacks.scoring_fn_utils import ScoringFnRegistry


@pytest.fixture
def exp_config() -> ExperimentConfig:
    attack_cb = AutoregressiveCallbackConfig(
        callback_name="tensor_univariate_fn_of_generation_from_text",
        callback_return_type="tensor",
        scoring_fn="doesnt_contain_toxicity_score",
        scoring_fn_arity=1,
    )
    final_cb = AutoregressiveCallbackConfig(
        callback_name="binary_univariate_fn_of_generation_from_text",
        callback_return_type="binary",
        scoring_fn="doesnt_contain_toxicity",
        scoring_fn_arity=1,
    )
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(
            final_success_binary_callback=final_cb,
            evaluation_attack=FewShotLMAttackConfig(
                seed=0,
                few_shot_temperature=0,
                k_shot=1,
                n_its=2,
                adversary=ModelConfig(
                    name_or_path="EleutherAI/pythia-14m",
                    family="pythia",
                    inference_type=InferenceType.GENERATION.value,
                    strict_load=True,
                    generation_config=GenerationConfig(
                        min_new_tokens=10,
                        max_new_tokens=20,
                        do_sample=True,
                        stop_strings=[
                            "?",
                            "?!",
                        ],
                    ),
                ),
                adversary_input_templates=["Here are some questions to ask:\n"],
                adversary_output_templates=["{}"],
                adversary_shot_template="{k}. {shot}\n",
                victim_success_callback=attack_cb,
            ),
        ),
        model=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="generation",
            eval_minibatch_size=2,
            generation_config=GenerationConfig(
                max_new_tokens=10,
                do_sample=True,
            ),
        ),
        dataset=DatasetConfig(
            dataset_type="PureGeneration",
            inference_type="generation",
            classification_as_generation=False,
            n_train=0,
            n_val=4,
            revision="latest",
        ),
    )
    interpolated = OmegaConf.to_object(OmegaConf.structured(config))
    assert isinstance(interpolated, ExperimentConfig)
    return interpolated


def test_few_shot_calls(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    assert isinstance(exp_config.evaluation.evaluation_attack, FewShotLMAttackConfig)
    exp_config.evaluation.evaluation_attack.few_shot_temperature = 1
    exp_config.run_name = "test_few_shot_calls"
    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    assert config.evaluation is not None
    assert isinstance(
        config.evaluation.final_success_binary_callback, AutoregressiveCallbackConfig
    )
    use_cpu = exp_config.environment.device == "cpu"
    final_callback_name = (
        exp_config.evaluation.final_success_binary_callback.callback_name
    )
    final_callback = CallbackRegistry.get_binary_callback(final_callback_name)
    scoring_fn = ScoringFnRegistry.get_univariate_scoring_fn(
        config.evaluation.final_success_binary_callback.scoring_fn
    )
    final_callback = partial(
        final_callback, scoring_fn=scoring_fn  # pyright: ignore[reportCallIssue]
    )

    wandb.init(
        project="robust-llm",
        group=config.experiment_name,
        job_type=config.job_type,
        name=config.run_name,
        mode="disabled",
    )

    accelerator = Accelerator(cpu=use_cpu)

    validation = load_rllm_dataset(exp_config.dataset, split="validation")
    num_classes = validation.num_classes

    victim = WrappedModel.from_config(exp_config.model, accelerator, num_classes)

    attack = prepare_attack(
        args=exp_config,
        victim=victim,
        training=False,
    )
    assert isinstance(attack, FewShotLMAttack)

    with patch.object(
        attack.adversary, "generate", wraps=attack.adversary.generate
    ) as mock_generate:
        do_adversarial_evaluation(
            victim=victim,
            dataset=validation,
            attack=attack,
            num_examples_to_log_detailed_info=exp_config.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
            final_success_binary_callback=final_callback,
            adv_training_round=0,
            victim_training_step_count=0,
            victim_training_datapoint_count=0,
            global_step_count=0,
            global_datapoint_count=0,
            resume_from_checkpoint=False,
        )
    calls = [
        attack.adversary.decode(call.kwargs["input_ids"].squeeze(0))
        for call in mock_generate.call_args_list
    ]
    for call_idx in range(2):
        assert calls[call_idx] == "Here are some questions to ask:\n"
    for call_idx in range(2, 4):
        assert calls[call_idx].startswith("Here are some questions to ask:\n1. ")
        assert "?" not in calls[call_idx][:-1]


def test_few_shot_zero_temperature(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    assert isinstance(exp_config.evaluation.evaluation_attack, FewShotLMAttackConfig)
    exp_config.evaluation.evaluation_attack.few_shot_temperature = 0
    exp_config.run_name = "test_few_shot_zero_temperature"
    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    assert config.evaluation is not None
    assert isinstance(
        config.evaluation.final_success_binary_callback, AutoregressiveCallbackConfig
    )
    use_cpu = exp_config.environment.device == "cpu"
    final_callback_name = (
        exp_config.evaluation.final_success_binary_callback.callback_name
    )
    final_callback = CallbackRegistry.get_binary_callback(final_callback_name)
    scoring_fn = ScoringFnRegistry.get_univariate_scoring_fn(
        config.evaluation.final_success_binary_callback.scoring_fn
    )
    final_callback = partial(
        final_callback, scoring_fn=scoring_fn  # pyright: ignore[reportCallIssue]
    )

    wandb.init(
        project="robust-llm",
        group=config.experiment_name,
        job_type=config.job_type,
        name=config.run_name,
        mode="disabled",
    )

    accelerator = Accelerator(cpu=use_cpu)

    validation = load_rllm_dataset(exp_config.dataset, split="validation")
    num_classes = validation.num_classes

    victim = WrappedModel.from_config(exp_config.model, accelerator, num_classes)

    attack = prepare_attack(
        args=exp_config,
        victim=victim,
        training=False,
    )
    assert isinstance(attack, FewShotLMAttack)

    do_adversarial_evaluation(
        victim=victim,
        dataset=validation,
        attack=attack,
        num_examples_to_log_detailed_info=exp_config.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
        final_success_binary_callback=final_callback,
        adv_training_round=0,
        victim_training_step_count=0,
        victim_training_datapoint_count=0,
        global_step_count=0,
        global_datapoint_count=0,
        resume_from_checkpoint=False,
    )
