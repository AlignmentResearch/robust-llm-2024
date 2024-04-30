"""Test that the various attacks don't crash."""

import pytest
import textattack.shared.utils

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
    MultipromptGCGAttackConfig,
    MultipromptRandomTokenAttackConfig,
)
from robust_llm.config.configs import EvaluationConfig
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
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/PasswordMatch",
            n_train=2,
            n_val=2,
        ),
    )
    return config


def _test_doesnt_crash(config: ExperimentConfig) -> None:
    # This is a global variable that needs to be reset between attacks
    # of different types because of a bug in TextAttack.
    # See GH#341 for more details.
    textattack.shared.utils.strings._flair_pos_tagger = None

    run_evaluation_pipeline(config)


def test_doesnt_crash_random_token(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = RandomTokenAttackConfig(
        min_tokens=2,
        max_tokens=3,
        max_iterations=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_multiprompt_random_token(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptRandomTokenAttackConfig(
        min_tokens=2,
        max_tokens=3,
        max_iterations=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = GCGAttackConfig(
        n_attack_tokens=3,
        n_its=2,
    )

    _test_doesnt_crash(exp_config)


def test_doesnt_crash_multiprompt_gcg(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = MultipromptGCGAttackConfig(
        n_attack_tokens=3,
        n_its=2,
    )
    _test_doesnt_crash(exp_config)


def test_doesnt_crash_trl(exp_config: ExperimentConfig) -> None:
    assert exp_config.evaluation is not None
    exp_config.evaluation.evaluation_attack = TRLAttackConfig(
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=1,
        model_save_path_prefix=None,
        adversary=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
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
    )
    _test_doesnt_crash(exp_config)
