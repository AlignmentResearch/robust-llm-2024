"""Test that the various attacks don't crash."""

import omegaconf
import pytest
import textattack.shared.utils

from robust_llm.attacks.text_attack.constants import TEXT_ATTACK_ATTACK_TYPES
from robust_llm.configs import (
    AttackConfig,
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    OverallConfig,
    RandomTokenAttackConfig,
    SearchBasedAttackConfig,
    TextAttackAttackConfig,
    TRLAttackConfig,
)
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
def overall_config() -> OverallConfig:
    config = OverallConfig(
        experiment=ExperimentConfig(
            environment=EnvironmentConfig(
                test_mode=True,
                model_family="pythia",
                model_name_or_path="EleutherAI/pythia-14m",
            ),
            dataset=DatasetConfig(
                dataset_type="AlignmentResearch/PasswordMatch",
                n_train=2,
                n_val=2,
            ),
            evaluation=EvaluationConfig(
                evaluation_attack=AttackConfig(
                    attack_type=omegaconf.MISSING,
                    random_token_attack_config=RandomTokenAttackConfig(
                        min_tokens=2,
                        max_tokens=3,
                        max_iterations=2,
                    ),
                    search_based_attack_config=SearchBasedAttackConfig(
                        n_attack_tokens=3,
                        n_its=2,
                    ),
                    trl_attack_config=TRLAttackConfig(
                        batch_size=2,
                        mini_batch_size=2,
                        gradient_accumulation_steps=1,
                        ppo_epochs=1,
                        model_save_path_prefix=None,
                    ),
                ),
            ),
        ),
    )
    return config


def _test_doesnt_crash(config: OverallConfig) -> None:
    # This is a global variable that needs to be reset between attacks
    # of different types because of a bug in TextAttack.
    # See GH#341 for more details.
    textattack.shared.utils.strings._flair_pos_tagger = None

    run_evaluation_pipeline(config)


@pytest.mark.parametrize(
    "attack_type",
    [
        "identity",
        "random_token",
        "multiprompt_random_token",
        "search_based",
        "trl",
    ],
)
def test_doesnt_crash(overall_config: OverallConfig, attack_type: str) -> None:
    overall_config.experiment.evaluation.evaluation_attack.attack_type = attack_type
    _test_doesnt_crash(overall_config)


def test_multiprompt_search_based_doesnt_crash(overall_config: OverallConfig) -> None:
    overall_config.experiment.evaluation.evaluation_attack = AttackConfig(
        attack_type="multiprompt_search_based",
        search_based_attack_config=SearchBasedAttackConfig(
            search_type="multiprompt_gcg"
        ),
    )
    _test_doesnt_crash(overall_config)


def test_covers_text_attacks() -> None:
    for attack_type in TEXT_ATTACK_ATTACK_TYPES:
        assert (
            attack_type in MODIFIABLE_WORDS_TEXT_ATTACKS
            or attack_type in NON_MODIFIABLE_WORDS_TEXT_ATTACKS
        )


@pytest.mark.parametrize("attack_type", MODIFIABLE_WORDS_TEXT_ATTACKS)
def test_modifiable_words_text_attack_doesnt_crash(
    overall_config: OverallConfig, attack_type: str
) -> None:
    overall_config.experiment.evaluation.evaluation_attack = AttackConfig(
        attack_type=attack_type,
        text_attack_attack_config=TextAttackAttackConfig(
            num_modifiable_words_per_chunk=1,
        ),
    )
    _test_doesnt_crash(overall_config)


# These cases are special because they don't set num_modifiable_words_per_chunk,
# but they run on the imdb dataset
@pytest.mark.parametrize("attack_type", NON_MODIFIABLE_WORDS_TEXT_ATTACKS)
def test_non_modifiable_words_text_attack_doesnt_crash(
    overall_config: OverallConfig, attack_type: str
) -> None:
    overall_config.experiment.evaluation.evaluation_attack = AttackConfig(
        attack_type=attack_type
    )
    overall_config.experiment.dataset.dataset_type = "AlignmentResearch/IMDB"
    _test_doesnt_crash(overall_config)
