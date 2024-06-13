from unittest.mock import PropertyMock, patch

import pytest
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf

from robust_llm.attacks.search_free.lm_based_attack import LMBasedAttack
from robust_llm.config.attack_configs import LMBasedAttackConfig
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
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.scoring_callbacks import CallbackRegistry


@pytest.fixture
def exp_config() -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(batch_size=1),
        model=ModelConfig(
            name_or_path="AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            eval_minibatch_size=2,
            generation_config=GenerationConfig(
                max_new_tokens=10,
                do_sample=True,
            ),
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            n_train=2,
            n_val=2,
        ),
    )
    return config


def test_adversary_input(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    n_its = 2
    exp_config.evaluation.evaluation_attack = LMBasedAttackConfig(
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            generation_config=GenerationConfig(
                min_new_tokens=10,
                max_new_tokens=20,
                do_sample=True,
            ),
        ),
        adversary_input_templates=[
            "{} Do something1!",
            "{} Do something2!",
        ],
        n_its=n_its,
    )
    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    assert exp_config.evaluation is not None
    use_cpu = exp_config.environment.device == "cpu"
    final_callback_name = exp_config.evaluation.final_success_binary_callback
    final_callback = CallbackRegistry.get_binary_callback(final_callback_name)

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
    assert isinstance(attack, LMBasedAttack)

    with patch.object(
        attack.adversary, "generate", wraps=attack.adversary.generate
    ) as mock_generate:
        do_adversarial_evaluation(
            victim=victim,
            dataset=validation,
            attack=attack,
            num_examples_to_log_detailed_info=exp_config.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
            final_success_binary_callback=final_callback,
        )
    first_call = attack.adversary.decode(
        mock_generate.call_args_list[0].kwargs["input_ids"].squeeze(0)
    )
    assert first_call.startswith("I don't particularly care from Michael Jackson.")
    assert first_call.endswith("a creepy white woman with a fake nose. Do something1!")

    second_call = attack.adversary.decode(
        mock_generate.call_args_list[n_its].kwargs["input_ids"].squeeze(0)
    )
    assert second_call.startswith("I never saw the other two")
    assert second_call.endswith("but at least it was not boring. Do something2!")


def test_wrong_chunks_dataset(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = LMBasedAttackConfig(
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            generation_config=GenerationConfig(
                min_new_tokens=10,
                max_new_tokens=20,
                do_sample=True,
            ),
        ),
        adversary_input_templates=[
            "{} Do something1!",
            "{} Do something2!",
        ],
        adversary_output_templates=["{}" for _ in range(4)],
        n_its=2,
    )
    exp_config.dataset.dataset_type = "AlignmentResearch/PasswordMatch"
    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    assert exp_config.evaluation is not None
    use_cpu = exp_config.environment.device == "cpu"
    final_callback_name = exp_config.evaluation.final_success_binary_callback
    final_callback = CallbackRegistry.get_binary_callback(final_callback_name)

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
    assert isinstance(attack, LMBasedAttack)

    with patch.object(
        type(validation), "modifiable_chunk_spec", new_callable=PropertyMock
    ) as mock_modifiable_chunk_spec:
        mock_modifiable_chunk_spec.return_value = ModifiableChunkSpec(
            [ChunkType.PERTURBABLE for _ in range(4)]
        )

        with pytest.raises(ValueError):
            do_adversarial_evaluation(
                victim=victim,
                dataset=validation,
                attack=attack,
                num_examples_to_log_detailed_info=exp_config.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
                final_success_binary_callback=final_callback,
            )
