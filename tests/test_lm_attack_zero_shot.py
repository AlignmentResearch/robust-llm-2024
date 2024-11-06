from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
import wandb
from accelerate import Accelerator

from robust_llm.attacks.attack_utils import create_attack
from robust_llm.attacks.search_free.lm_attack_zero_shot import ZeroShotLMAttack
from robust_llm.config.attack_configs import LMAttackConfig
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
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.scoring_callbacks import build_binary_scoring_callback
from robust_llm.utils import interpolate_config


@pytest.fixture
def exp_config() -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
            minibatch_multiplier=1,
        ),
        evaluation=EvaluationConfig(),
        model=ModelConfig(
            name_or_path="AlignmentResearch/robust_llm_pythia-14m_clf_imdb_v-ian-067_s-0",  # noqa: E501
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
            generation_config=GenerationConfig(
                max_new_tokens=10,
                do_sample=True,
            ),
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="2.1.0",
            n_train=2,
            n_val=2,
        ),
    )
    return config


def test_adversary_input_zs(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    n_its = 2
    exp_config.evaluation.evaluation_attack = LMAttackConfig(
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
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
    )
    exp_config.evaluation.num_iterations = n_its
    config = interpolate_config(exp_config)
    assert config.evaluation is not None
    use_cpu = config.environment.device == "cpu"
    final_callback_config = config.evaluation.final_success_binary_callback
    final_callback = build_binary_scoring_callback(final_callback_config)

    wandb.init(
        project="robust-llm",
        group=config.experiment_name,
        job_type=config.job_type,
        name=config.run_name,
        mode="disabled",
    )

    accelerator = Accelerator(cpu=use_cpu)

    validation = load_rllm_dataset(config.dataset, split="validation")
    num_classes = validation.num_classes

    victim = WrappedModel.from_config(config.model, accelerator, num_classes)

    attack = create_attack(
        exp_config=config,
        victim=victim,
        is_training=False,
    )
    assert isinstance(attack, ZeroShotLMAttack)

    with patch.object(
        attack.adversary, "generate", wraps=attack.adversary.generate
    ) as mock_generate:
        do_adversarial_evaluation(
            victim=victim,
            dataset=validation,
            attack=attack,
            n_its=n_its,
            num_examples_to_log_detailed_info=config.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
            final_success_binary_callback=final_callback,
            adv_training_round=0,
            victim_training_step_count=0,
            victim_training_datapoint_count=0,
            global_step_count=0,
            global_datapoint_count=0,
            local_files_path=Path("/tmp/dummy_local_files_path"),
        )
    first_call = attack.adversary.decode(
        mock_generate.call_args_list[0].kwargs["input_ids"].squeeze(0)
    )
    assert first_call.startswith("This movie was awesome.")
    assert first_call.endswith("Even if you are not, BUY! Do something1!")

    second_call = attack.adversary.decode(
        mock_generate.call_args_list[n_its].kwargs["input_ids"].squeeze(0)
    )
    assert second_call.startswith("When the Italians and Miles")
    assert second_call.endswith("Totally lives up to the first movie. Do something1!")


def test_wrong_chunks_dataset_zs(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = LMAttackConfig(
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
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
    )
    exp_config.evaluation.num_iterations = 2
    exp_config.dataset.dataset_type = "AlignmentResearch/PasswordMatch"
    config = interpolate_config(exp_config)
    assert config.evaluation is not None
    use_cpu = config.environment.device == "cpu"
    final_callback_config = config.evaluation.final_success_binary_callback
    final_callback = build_binary_scoring_callback(final_callback_config)

    wandb.init(
        project="robust-llm",
        group=config.experiment_name,
        job_type=config.job_type,
        name=config.run_name,
        mode="disabled",
    )

    accelerator = Accelerator(cpu=use_cpu)

    validation = load_rllm_dataset(config.dataset, split="validation")
    num_classes = validation.num_classes

    victim = WrappedModel.from_config(config.model, accelerator, num_classes)

    attack = create_attack(
        exp_config=config,
        victim=victim,
        is_training=False,
    )
    assert isinstance(attack, ZeroShotLMAttack)

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
                n_its=exp_config.evaluation.num_iterations,
                num_examples_to_log_detailed_info=config.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
                final_success_binary_callback=final_callback,
                adv_training_round=0,
                victim_training_step_count=0,
                victim_training_datapoint_count=0,
                global_step_count=0,
                global_datapoint_count=0,
                local_files_path=Path("/tmp/dummy_local_files_path"),
            )
