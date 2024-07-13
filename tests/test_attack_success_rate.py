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
    LMAttackConfig,
    MultipromptGCGAttackConfig,
)
from robust_llm.config.callback_configs import AutoregressiveCallbackConfig
from robust_llm.config.configs import EvaluationConfig
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
            inference_type="classification",
            strict_load=True,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            revision="<1.1.0",
            n_train=2,
            n_val=100,
        ),
    )
    return config


def _test_attack(
    exp_config: ExperimentConfig, success_rate_at_least: float = 0.0
) -> None:
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
    results = run_evaluation_pipeline(config)
    actual = results["adversarial_eval/attack_success_rate"]
    print(f"Success rate: {actual}")
    assert actual >= success_rate_at_least


def test_random_token(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
        n_its=50,
    )
    _test_attack(exp_config, success_rate_at_least=0.18)


def test_multiprompt_random_token(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        n_attack_tokens=1,
        n_its=250,
        prompt_attack_mode="multi-prompt",
    )
    _test_attack(exp_config, success_rate_at_least=0.96)


def test_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=1,
        n_its=2,
        n_candidates_per_it=128,
    )

    _test_attack(exp_config, success_rate_at_least=0.28)


def test_multiprompt_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptGCGAttackConfig(
        n_attack_tokens=5,
        n_its=10,
    )
    _test_attack(exp_config, success_rate_at_least=1.00)


def test_lm_attack_clf(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.dataset = DatasetConfig(
        dataset_type="AlignmentResearch/IMDB",
        revision="<1.1.0",
        n_train=2,
        n_val=2,
    )
    exp_config.evaluation.evaluation_attack = LMAttackConfig(
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
        n_its=2,
    )
    _test_attack(exp_config)


def test_lm_attack_gen(exp_config: ExperimentConfig) -> None:
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
        name_or_path="EleutherAI/pythia-14m",
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
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            inference_type=InferenceType.GENERATION.value,
            strict_load=True,
            generation_config=GenerationConfig(
                max_length=None,
                min_new_tokens=10,
                max_new_tokens=20,
                do_sample=True,
            ),
        ),
        adversary_input_templates=[
            "{} Do something!",
        ],
        n_its=2,
        victim_success_callback=phone_number_in_generation_callback,
    )
    _test_attack(exp_config)


def test_trl(exp_config: ExperimentConfig) -> None:
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
    exp_config.dataset.n_val = 2

    _test_attack(exp_config)


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
        query_budget=10,
    )
    exp_config.dataset.n_val = 2
    _test_attack(exp_config)


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
        query_budget=10,
    )
    exp_config.dataset.n_val = 2
    _test_attack(exp_config)
