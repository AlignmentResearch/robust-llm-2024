"""Test that the various attacks don't crash."""

import pytest
import textattack.shared.utils
from omegaconf import OmegaConf

from robust_llm.attacks.text_attack.constants import TEXT_ATTACK_ATTACK_TYPES
from robust_llm.config import (
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RandomTokenAttackConfig,
    TextAttackAttackConfig,
    TRLAttackConfig,
)
from robust_llm.config.attack_configs import (
    GCGAttackConfig,
    LMBasedAttackConfig,
    MultipromptGCGAttackConfig,
)
from robust_llm.config.configs import EvaluationConfig
from robust_llm.config.dataset_configs import ContactInfoDatasetConfig
from robust_llm.config.model_configs import GenerationConfig
from robust_llm.models.model_utils import InferenceType
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
        evaluation=EvaluationConfig(),
        model=ModelConfig(
            # We use a finetuned model so that the classification head isn't
            # randomly initalized. It's fine to use a model that isn't finetuned
            # for the task, because we are only testing that the attack doesn't crash.
            name_or_path="AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            strict_load=True,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            n_train=2,
            n_val=2,
        ),
    )
    return config


def _test_doesnt_crash(exp_config: ExperimentConfig) -> None:
    """Small wrapper around run_evaluation_pipeline.

    This resets a TextAttack global variable, and runs interpolation:
    - First we convert to an OmegaConf structured config, which enables
    interpolation.
    - Then we convert back to an ExperimentConfig object, and use
    that to run the pipeline.
    """
    # This is a global variable that needs to be reset between attacks
    # of different types because of a bug in TextAttack.
    # See GH#341 for more details.
    textattack.shared.utils.strings._flair_pos_tagger = None

    config = OmegaConf.to_object(OmegaConf.structured(exp_config))
    assert isinstance(config, ExperimentConfig)
    run_evaluation_pipeline(config)


def test_doesnt_crash_random_token(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=3,
        n_its=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_multiprompt_random_token(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=3,
        n_its=2,
        prompt_attack_mode="multi-prompt",
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=2,
        n_its=2,
        n_candidates_per_it=16,
    )

    _test_doesnt_crash(exp_config)


def test_doesnt_crash_multiprompt_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptGCGAttackConfig(
        n_attack_tokens=3,
        n_its=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_lm_attack_clf(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.dataset = DatasetConfig(
        dataset_type="AlignmentResearch/IMDB",
        n_train=2,
        n_val=2,
    )
    exp_config.evaluation.evaluation_attack = LMBasedAttackConfig(
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            padding_side="left",
            generation_config=GenerationConfig(
                min_new_tokens=10,
                max_new_tokens=20,
                do_sample=True,
            ),
        ),
        templates=[
            "1: {}, 2: {}, 3: {}. Do something1!",
            "1: {}, 2: {}, 3: {}. Do something2!",
        ],
        n_its=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_lm_attack_gen(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.dataset = ContactInfoDatasetConfig(
        dataset_type="ContactInfo",
        n_train=2,
        n_val=2,
        info_type="phone_number",
        inference_type="generation",
        classification_as_generation=False,
    )
    exp_config.model = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        inference_type="generation",
        strict_load=True,
        padding_side="left",
        generation_config=GenerationConfig(
            max_length=None,
            min_new_tokens=10,
            max_new_tokens=20,
            do_sample=True,
        ),
    )
    exp_config.evaluation.final_success_binary_callback = "phone_number_in_generation"
    exp_config.evaluation.evaluation_attack = LMBasedAttackConfig(
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            padding_side="left",
            generation_config=GenerationConfig(
                max_length=None,
                min_new_tokens=10,
                max_new_tokens=20,
                do_sample=True,
            ),
        ),
        templates=[
            "{} Do something!",
        ],
        n_its=2,
        victim_success_binary_callback="phone_number_in_generation",
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_trl(exp_config: ExperimentConfig) -> None:
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
            padding_side="left",
        ),
    )

    _test_doesnt_crash(exp_config)


def test_covers_text_attacks() -> None:
    for attack_type in TEXT_ATTACK_ATTACK_TYPES:
        assert (
            attack_type in MODIFIABLE_WORDS_TEXT_ATTACKS
            or attack_type in NON_MODIFIABLE_WORDS_TEXT_ATTACKS
        )


@pytest.mark.parametrize("text_attack_recipe", MODIFIABLE_WORDS_TEXT_ATTACKS)
def test_modifiable_words_text_attack_doesnt_crash(
    exp_config: ExperimentConfig, text_attack_recipe: str
) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = TextAttackAttackConfig(
        text_attack_recipe=text_attack_recipe,
        num_modifiable_words_per_chunk=1,
        num_examples=2,
        query_budget=10,
    )
    _test_doesnt_crash(exp_config)


# These cases are special because they don't set num_modifiable_words_per_chunk,
# but they run on the imdb dataset
@pytest.mark.parametrize("text_attack_recipe", NON_MODIFIABLE_WORDS_TEXT_ATTACKS)
def test_non_modifiable_words_text_attack_doesnt_crash(
    exp_config: ExperimentConfig, text_attack_recipe: str
) -> None:
    assert exp_config.evaluation is not None
    exp_config.dataset.dataset_type = "AlignmentResearch/IMDB"
    exp_config.evaluation.evaluation_attack = TextAttackAttackConfig(
        text_attack_recipe=text_attack_recipe,
        num_modifiable_words_per_chunk=None,
        num_examples=2,
        query_budget=10,
    )
    _test_doesnt_crash(exp_config)
