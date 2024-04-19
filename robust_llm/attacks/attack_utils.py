import transformers
from accelerate import Accelerator

from robust_llm.attacks.attack import Attack, IdentityAttack
from robust_llm.attacks.multiprompt_random_token import MultiPromptRandomTokenAttack
from robust_llm.attacks.random_token import RandomTokenAttack
from robust_llm.attacks.search_based.multiprompt_search_based import (
    MultiPromptSearchBasedAttack,
)
from robust_llm.attacks.search_based.search_based import SearchBasedAttack
from robust_llm.attacks.text_attack.constants import TEXT_ATTACK_ATTACK_TYPES
from robust_llm.attacks.text_attack.text_attack import TextAttackAttack
from robust_llm.attacks.trl.trl import TRLAttack
from robust_llm.configs import AttackConfig
from robust_llm.utils import LanguageModel


def create_attack(
    attack_config: AttackConfig,
    logging_name: str,
    victim_model: LanguageModel,
    victim_tokenizer: transformers.PreTrainedTokenizerBase,
    accelerator: Accelerator,
) -> Attack:
    """Returns an attack object of a proper type."""
    # TODO(niki): simplify so everything has same args?
    if attack_config.attack_type == "identity":
        return IdentityAttack(
            attack_config=attack_config,
        )
    elif attack_config.attack_type == "random_token":
        return RandomTokenAttack(
            attack_config=attack_config,
            victim_model=victim_model,
            victim_tokenizer=victim_tokenizer,
        )
    elif attack_config.attack_type == "multiprompt_random_token":
        return MultiPromptRandomTokenAttack(
            attack_config=attack_config,
            victim_model=victim_model,
            victim_tokenizer=victim_tokenizer,
        )
    elif attack_config.attack_type == "trl":
        return TRLAttack(
            attack_config=attack_config,
            logging_name=logging_name,
            victim_model=victim_model,
            victim_tokenizer=victim_tokenizer,
        )
    elif attack_config.attack_type == "search_based":
        return SearchBasedAttack(
            attack_config=attack_config,
            model=victim_model,
            tokenizer=victim_tokenizer,
            accelerator=accelerator,
        )
    elif attack_config.attack_type == "multiprompt_search_based":
        return MultiPromptSearchBasedAttack(
            attack_config=attack_config,
            model=victim_model,
            tokenizer=victim_tokenizer,
            accelerator=accelerator,
        )
    elif attack_config.attack_type in TEXT_ATTACK_ATTACK_TYPES:
        return TextAttackAttack(
            attack_config=attack_config,
            model=victim_model,
            tokenizer=victim_tokenizer,
        )
    else:
        raise ValueError(f"Attack type {attack_config.attack_type} not recognized.")
