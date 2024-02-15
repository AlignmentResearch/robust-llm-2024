from typing import Callable, Optional

import transformers

from robust_llm.attacks.attack import Attack, IdentityAttack
from robust_llm.attacks.brute_force_tomita import BruteForceTomitaAttack
from robust_llm.attacks.random_token import RandomTokenAttack
from robust_llm.attacks.text_attack import TEXT_ATTACK_ATTACK_TYPES, TextAttackAttack
from robust_llm.attacks.trl.trl import TRLAttack
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.utils import LanguageModel


def create_attack(
    attack_config: AttackConfig,
    modifiable_chunks_spec: ModifiableChunksSpec,
    dataset_type: str,
    victim_model: LanguageModel,
    victim_tokenizer: transformers.PreTrainedTokenizerBase,
    language_generator_name: Optional[str] = None,
    ground_truth_label_fn: Optional[Callable[[str], int]] = None,
) -> Attack:
    """Returns an attack object of a proper type."""
    # TODO(niki): simplify so everything has same args?
    if attack_config.attack_type == "identity":
        return IdentityAttack(
            attack_config=attack_config, modifiable_chunks_spec=modifiable_chunks_spec
        )
    elif attack_config.attack_type == "brute_force":
        assert language_generator_name is not None
        return BruteForceTomitaAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            dataset_type=dataset_type,
            # TODO(niki): put this in the config?
            language_generator_name=language_generator_name,
        )
    elif attack_config.attack_type == "random_token":
        return RandomTokenAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            dataset_type=dataset_type,
            tokenizer=victim_tokenizer,
        )
    elif attack_config.attack_type == "trl":
        return TRLAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            dataset_type=dataset_type,
            victim_model=victim_model,
            victim_tokenizer=victim_tokenizer,
            ground_truth_label_fn=ground_truth_label_fn,
        )
    elif attack_config.attack_type == "random_token":
        return RandomTokenAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            dataset_type=dataset_type,
            tokenizer=victim_tokenizer,
        )
    elif attack_config.attack_type in TEXT_ATTACK_ATTACK_TYPES:
        return TextAttackAttack(
            attack_config=attack_config,
            modifiable_chunks_spec=modifiable_chunks_spec,
            model=victim_model,
            tokenizer=victim_tokenizer,
            ground_truth_label_fn=ground_truth_label_fn,
        )
    else:
        raise ValueError(f"Attack type {attack_config.attack_type} not recognized.")
