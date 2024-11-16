from typing import Any, Type

from robust_llm.attacks.attack import Attack, IdentityAttack, extract_attack_config
from robust_llm.attacks.search_based.search_based import SearchBasedAttack
from robust_llm.attacks.search_free.lm_attack_few_shot import FewShotLMAttack
from robust_llm.attacks.search_free.lm_attack_zero_shot import ZeroShotLMAttack
from robust_llm.attacks.search_free.random_token import RandomTokenAttack
from robust_llm.config.attack_configs import (
    BeamSearchAttackConfig,
    FewShotLMAttackConfig,
    GCGAttackConfig,
    IdentityAttackConfig,
    LMAttackConfig,
    RandomTokenAttackConfig,
)
from robust_llm.config.configs import ExperimentConfig
from robust_llm.models import WrappedModel


def create_attack(
    exp_config: ExperimentConfig,
    victim: WrappedModel,
    is_training: bool,
) -> Attack[Any]:
    """Returns an attack object of a proper type."""
    # This match-case statement uses class patterns, as described in this SO
    # answer: https://stackoverflow.com/a/67524642
    attack_config = extract_attack_config(exp_config, is_training)
    cls: Type[Attack[Any]]
    match attack_config:
        # Baseline attacks
        case IdentityAttackConfig():
            cls = IdentityAttack
        case RandomTokenAttackConfig():
            cls = RandomTokenAttack
        case BeamSearchAttackConfig() | GCGAttackConfig():
            cls = SearchBasedAttack
        case FewShotLMAttackConfig():
            cls = FewShotLMAttack
        case LMAttackConfig():
            cls = ZeroShotLMAttack
        case _:
            raise ValueError(f"Type of attack config {attack_config} not recognized.")
    return cls(exp_config, victim=victim, is_training=is_training)
