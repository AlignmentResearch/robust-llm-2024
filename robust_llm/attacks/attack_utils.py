from typing import Callable

import transformers
from git import Optional

from robust_llm.attacks.attack import Attack, IdentityAttack
from robust_llm.attacks.brute_force_tomita import BruteForceTomitaAttack
from robust_llm.attacks.random_token_attack import RandomTokenAttack
from robust_llm.attacks.text_attack import TEXT_ATTACK_ATTACK_TYPES, TextAttackAttack
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.utils import LanguageModel


def create_attack(
    attack_config: AttackConfig,
    modifiable_chunks_spec: ModifiableChunksSpec,
    dataset_type: str,
    model: LanguageModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    language_generator_name: str,
    ground_truth_label_fn: Optional[Callable[[str], int]],
) -> Attack:
    """Returns an attack object of a proper type."""
    if attack_config.attack_type == "identity":
        return IdentityAttack(
            attack_config=attack_config, modifiable_chunks_spec=modifiable_chunks_spec
        )
    elif attack_config.attack_type == "brute_force":
        return BruteForceTomitaAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            dataset_type=dataset_type,
            language_generator_name=language_generator_name,
        )
    elif attack_config.attack_type == "random_token":
        return RandomTokenAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            dataset_type=dataset_type,
            tokenizer=tokenizer,
        )
    elif attack_config.attack_type in TEXT_ATTACK_ATTACK_TYPES:
        return TextAttackAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            model=model,
            tokenizer=tokenizer,
            ground_truth_label_fn=ground_truth_label_fn,
        )
    else:
        raise ValueError(f"Attack type {attack_config.attack_type} not recognized.")
