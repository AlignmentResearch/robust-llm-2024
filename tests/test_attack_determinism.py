"""Test that attacks don't crash and are reproducible."""

import pytest
import torch
from accelerate import Accelerator

from robust_llm.attacks.attack_utils import create_attack
from robust_llm.config import (
    BeastAttackConfig,
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    GCGAttackConfig,
    LMAttackConfig,
    ModelConfig,
    RandomTokenAttackConfig,
)
from robust_llm.config.callback_configs import (
    AutoregressiveCallbackConfig,
    CallbackConfig,
)
from robust_llm.config.configs import EvaluationConfig, get_checkpoint_path
from robust_llm.config.model_configs import GenerationConfig
from robust_llm.dist_utils import dist_rmtree
from robust_llm.evaluation import attack_dataset
from robust_llm.logging_utils import LoggingContext
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.pipelines.evaluation_pipeline import load_rllm_dataset
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import interpolate_config, maybe_make_deterministic

DUMMY_NEOX_CLF = "hf-internal-testing/tiny-random-GPTNeoXForSequenceClassification"
DUMMY_NEOX_GEN = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


@pytest.fixture
def exp_config() -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_type="evaluation",
        environment=EnvironmentConfig(
            test_mode=True,
            allow_checkpointing=False,
        ),
        evaluation=EvaluationConfig(
            evaluation_attack=RandomTokenAttackConfig(
                n_attack_tokens=1,
            ),
            num_iterations=2,
        ),
        model=ModelConfig(
            name_or_path=DUMMY_NEOX_CLF,
            family="pythia",
            inference_type="classification",
            strict_load=True,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            revision="2.1.0",
            n_train=0,
            n_val=5,
        ),
    )
    config = interpolate_config(config)
    maybe_make_deterministic(
        config.environment.deterministic, config.environment.cublas_config
    )
    return config


@pytest.fixture
def validation_set(exp_config: ExperimentConfig) -> RLLMDataset:
    return load_rllm_dataset(exp_config.dataset, split="validation")


@pytest.fixture
def victim_model(exp_config: ExperimentConfig):
    accelerator = Accelerator(cpu=not torch.cuda.is_available())
    return WrappedModel.from_config(exp_config.model, accelerator, num_classes=2)


def _get_attacked_texts(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
) -> list[list[str]]:
    """Get the attacked texts from the evaluation pipeline."""
    assert exp_config.evaluation is not None

    attack = create_attack(
        exp_config=exp_config,
        victim=victim_model,
        is_training=False,
    )
    attack_out = attack_dataset(
        victim=victim_model,
        dataset_to_attack=validation_set,
        attack=attack,
        n_its=exp_config.evaluation.num_iterations,
        resume_from_checkpoint=exp_config.environment.allow_checkpointing,
    )
    assert attack_out.attack_data is not None
    attacked_texts = attack_out.attack_data.iteration_texts
    return attacked_texts


def _test_attack_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    """Test that attacks are deterministic."""
    attacked_texts_1 = _get_attacked_texts(exp_config, victim_model, validation_set)
    attacked_texts_2 = _get_attacked_texts(exp_config, victim_model, validation_set)
    assert attacked_texts_1 == attacked_texts_2


def test_random_token_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
    )
    _test_attack_determinism(exp_config, victim_model, validation_set)


@pytest.mark.multigpu
def test_random_token_checkpoint_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
    capsys: pytest.CaptureFixture,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
        save_steps=3,
    )
    exp_config.environment.allow_checkpointing = True
    logging_context = LoggingContext(
        args=exp_config,
    )
    logging_context.setup()
    checkpoint_dir = get_checkpoint_path(exp_config)
    dist_rmtree(checkpoint_dir)
    initial_texts = _get_attacked_texts(exp_config, victim_model, validation_set)
    initial_stdout = capsys.readouterr()[-1]
    assert "Loading state from" not in initial_stdout
    rerun_texts = _get_attacked_texts(exp_config, victim_model, validation_set)
    rerun_stdout = capsys.readouterr()[-1]
    assert "Loading state from" in rerun_stdout
    assert initial_texts == rerun_texts


def test_multiprompt_random_token_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
        prompt_attack_mode="multi-prompt",
    )
    _test_attack_determinism(exp_config, victim_model, validation_set)


def test_gcg_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=1,
        n_candidates_per_it=128,
        differentiable_embeds_callback=CallbackConfig(
            "losses_from_small_embeds", "tensor"
        ),
    )
    _test_attack_determinism(exp_config, victim_model, validation_set)


@pytest.mark.multigpu
def test_gcg_checkpoint_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
    capsys: pytest.CaptureFixture,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=1,
        n_candidates_per_it=128,
        differentiable_embeds_callback=CallbackConfig(
            "losses_from_small_embeds", "tensor"
        ),
        save_steps=3,
    )
    exp_config.environment.allow_checkpointing = True
    logging_context = LoggingContext(
        args=exp_config,
    )
    logging_context.setup()
    checkpoint_dir = get_checkpoint_path(exp_config)
    dist_rmtree(checkpoint_dir)
    initial_texts = _get_attacked_texts(exp_config, victim_model, validation_set)
    initial_stdout = capsys.readouterr()[-1]
    assert "Loading state from" not in initial_stdout
    rerun_texts = _get_attacked_texts(exp_config, victim_model, validation_set)
    rerun_stdout = capsys.readouterr()[-1]
    assert "Loading state from" in rerun_stdout
    assert initial_texts == rerun_texts


def test_lm_attack_clf_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = LMAttackConfig(
        adversary=ModelConfig(
            name_or_path=DUMMY_NEOX_GEN,
            family="pythia",
            inference_type="generation",
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
    )
    exp_config = interpolate_config(exp_config)
    _test_attack_determinism(exp_config, victim_model, validation_set)


def test_lm_attack_gen_determinism(
    exp_config: ExperimentConfig,
):
    phone_number_in_generation_callback = AutoregressiveCallbackConfig(
        callback_name="binary_univariate_fn_of_generation_from_text",
        callback_return_type="binary",
        scoring_fn="doesnt_contain_phone_number",
        scoring_fn_arity=1,
    )
    assert exp_config.evaluation is not None
    exp_config.dataset = DatasetConfig(
        dataset_type="PureGeneration",
        revision="main",
        n_train=2,
        n_val=2,
        inference_type="generation",
        classification_as_generation=False,
    )
    exp_config.model = ModelConfig(
        name_or_path=DUMMY_NEOX_GEN,
        family="pythia",
        inference_type="generation",
        strict_load=True,
        generation_config=GenerationConfig(
            max_length=None,
            min_new_tokens=10,
            max_new_tokens=20,
            do_sample=True,
        ),
    )
    exp_config.evaluation.final_success_binary_callback = (
        phone_number_in_generation_callback
    )
    exp_config.evaluation.evaluation_attack = LMAttackConfig(
        adversary=ModelConfig(
            name_or_path=DUMMY_NEOX_GEN,
            family="pythia",
            inference_type="generation",
            strict_load=True,
            generation_config=GenerationConfig(
                min_new_tokens=10,
                max_new_tokens=20,
                do_sample=True,
            ),
        ),
        adversary_input_templates=[
            "{} Do something1!",
        ],
        victim_success_callback=phone_number_in_generation_callback,
    )
    exp_config = interpolate_config(exp_config)
    victim_model = WrappedModel.from_config(
        exp_config.model, accelerator=Accelerator(cpu=not torch.cuda.is_available())
    )
    validation_set = load_rllm_dataset(exp_config.dataset, split="validation")
    _test_attack_determinism(exp_config, victim_model, validation_set)


def test_beast_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = BeastAttackConfig(
        beam_search_width=3,
        beam_branch_factor=3,
        sampling_model=ModelConfig(
            name_or_path=DUMMY_NEOX_GEN,
            family="pythia",
            inference_type="generation",
            revision="main",
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=0.5,
        ),
    )
    _test_attack_determinism(exp_config, victim_model, validation_set)


@pytest.mark.multigpu
def test_beast_checkpoint_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
    capsys: pytest.CaptureFixture,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = BeastAttackConfig(
        beam_search_width=3,
        beam_branch_factor=3,
        sampling_model=ModelConfig(
            name_or_path=DUMMY_NEOX_GEN,
            family="pythia",
            inference_type="generation",
            revision="main",
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=0.5,
        ),
        save_steps=3,
    )
    exp_config.environment.allow_checkpointing = True
    logging_context = LoggingContext(
        args=exp_config,
    )
    logging_context.setup()
    checkpoint_dir = get_checkpoint_path(exp_config)
    dist_rmtree(checkpoint_dir)
    initial_texts = _get_attacked_texts(exp_config, victim_model, validation_set)
    initial_stdout = capsys.readouterr()[-1]
    assert "Loading state from" not in initial_stdout
    rerun_texts = _get_attacked_texts(exp_config, victim_model, validation_set)
    rerun_stdout = capsys.readouterr()[-1]
    assert "Loading state from" in rerun_stdout
    assert initial_texts == rerun_texts
