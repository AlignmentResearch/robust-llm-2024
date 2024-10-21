"""Test that attacks don't crash and are reproducible."""

import pytest
import textattack.shared.utils
from accelerate import Accelerator

from robust_llm.attacks.text_attack.constants import TEXT_ATTACK_ATTACK_TYPES
from robust_llm.config import (
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RandomTokenAttackConfig,
    TextAttackAttackConfig,
)
from robust_llm.config.attack_configs import (
    GCGAttackConfig,
    LMAttackConfig,
    MultipromptGCGAttackConfig,
)
from robust_llm.config.callback_configs import (
    AutoregressiveCallbackConfig,
    CallbackConfig,
)
from robust_llm.config.configs import EvaluationConfig
from robust_llm.config.model_configs import GenerationConfig
from robust_llm.evaluation import attack_dataset
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.pipelines.evaluation_pipeline import load_rllm_dataset, prepare_attack
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import interpolate_config

DUMMY_NEOX_CLF = "hf-internal-testing/tiny-random-GPTNeoXForSequenceClassification"
DUMMY_NEOX_GEN = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
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
    return config


@pytest.fixture
def validation_set(exp_config: ExperimentConfig) -> RLLMDataset:
    return load_rllm_dataset(exp_config.dataset, split="validation")


@pytest.fixture
def victim_model(exp_config: ExperimentConfig):
    accelerator = Accelerator(cpu=True)
    return WrappedModel.from_config(exp_config.model, accelerator, num_classes=2)


def _get_attacked_texts(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
) -> list[list[str]]:
    """Get the attacked texts from the evaluation pipeline."""

    # This is a global variable that needs to be reset between attacks
    # of different types because of a bug in TextAttack.
    # See GH#341 for more details.
    textattack.shared.utils.strings._flair_pos_tagger = None

    assert exp_config.evaluation is not None

    attack = prepare_attack(
        args=exp_config,
        victim=victim_model,
        training=False,
    )
    attack_out, _ = attack_dataset(
        victim=victim_model,
        dataset_to_attack=validation_set,
        attack=attack,
        n_its=exp_config.evaluation.num_iterations,
        resume_from_checkpoint=False,
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


def test_multiprompt_gcg_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set: RLLMDataset,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptGCGAttackConfig(
        n_attack_tokens=5,
        differentiable_embeds_callback=CallbackConfig(
            "losses_from_small_embeds", "tensor"
        ),
    )
    exp_config = interpolate_config(exp_config)
    _test_attack_determinism(exp_config, victim_model, validation_set)


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
        exp_config.model, accelerator=Accelerator(cpu=True)
    )
    validation_set = load_rllm_dataset(exp_config.dataset, split="validation")
    _test_attack_determinism(exp_config, victim_model, validation_set)


def test_covers_text_attacks() -> None:
    for attack_type in TEXT_ATTACK_ATTACK_TYPES:
        assert (
            attack_type in MODIFIABLE_WORDS_TEXT_ATTACKS
            or attack_type in NON_MODIFIABLE_WORDS_TEXT_ATTACKS
        )


@pytest.mark.skip(reason="Attack data not implemented for text attacks")
@pytest.mark.parametrize("text_attack_recipe", MODIFIABLE_WORDS_TEXT_ATTACKS)
def test_modifiable_words_text_attack_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set,
    text_attack_recipe: str,
):
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = TextAttackAttackConfig(
        text_attack_recipe=text_attack_recipe,
        num_modifiable_words_per_chunk=1,
        query_budget=10,
    )
    _test_attack_determinism(exp_config, victim_model, validation_set)


# These cases are special because they don't set num_modifiable_words_per_chunk,
# but they run on the imdb dataset
@pytest.mark.skip(reason="Attack data not implemented for text attacks")
@pytest.mark.parametrize("text_attack_recipe", NON_MODIFIABLE_WORDS_TEXT_ATTACKS)
def test_non_modifiable_words_text_attack_determinism(
    exp_config: ExperimentConfig,
    victim_model: WrappedModel,
    validation_set,
    text_attack_recipe: str,
):
    assert exp_config.evaluation is not None
    exp_config.dataset.dataset_type = "AlignmentResearch/IMDB"
    exp_config.evaluation.evaluation_attack = TextAttackAttackConfig(
        text_attack_recipe=text_attack_recipe,
        num_modifiable_words_per_chunk=None,
        query_budget=10,
    )
    _test_attack_determinism(exp_config, victim_model, validation_set)
