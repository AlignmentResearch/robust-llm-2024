import transformers
from accelerate import Accelerator

from robust_llm.attacks.attack import Attack, IdentityAttack
from robust_llm.attacks.multiprompt_random_token import MultiPromptRandomTokenAttack
from robust_llm.attacks.random_token import RandomTokenAttack
from robust_llm.attacks.search_based.multiprompt_search_based import (
    MultiPromptSearchBasedAttack,
)
from robust_llm.attacks.search_based.search_based import SearchBasedAttack
from robust_llm.attacks.text_attack.text_attack import TextAttackAttack
from robust_llm.attacks.trl.trl import TRLAttack
from robust_llm.config.attack_configs import (
    BeamSearchAttackConfig,
    GCGAttackConfig,
    IdentityAttackConfig,
    MultipromptGCGAttackConfig,
    MultipromptRandomTokenAttackConfig,
    RandomTokenAttackConfig,
    TextAttackAttackConfig,
    TRLAttackConfig,
)
from robust_llm.config.configs import AttackConfig
from robust_llm.utils import LanguageModel


def create_attack(
    attack_config: AttackConfig,
    logging_name: str,
    victim_model: LanguageModel,
    victim_tokenizer: transformers.PreTrainedTokenizerBase,
    accelerator: Accelerator,
) -> Attack:
    """Returns an attack object of a proper type."""
    # This match-case statement uses class patterns, as described in this SO
    # answer: https://stackoverflow.com/a/67524642
    match attack_config:
        # Baseline attacks
        case IdentityAttackConfig():
            return IdentityAttack(
                attack_config=attack_config,
            )
        case RandomTokenAttackConfig():
            return RandomTokenAttack(
                attack_config=attack_config,
                victim_model=victim_model,
                victim_tokenizer=victim_tokenizer,
            )
        case MultipromptRandomTokenAttackConfig():
            return MultiPromptRandomTokenAttack(
                attack_config=attack_config,
                victim_model=victim_model,
                victim_tokenizer=victim_tokenizer,
            )
        # Search-based attacks
        case BeamSearchAttackConfig() | GCGAttackConfig():
            return SearchBasedAttack(
                attack_config=attack_config,
                model=victim_model,
                tokenizer=victim_tokenizer,
                accelerator=accelerator,
            )
        case MultipromptGCGAttackConfig():
            return MultiPromptSearchBasedAttack(
                attack_config=attack_config,
                model=victim_model,
                tokenizer=victim_tokenizer,
                accelerator=accelerator,
            )
        # Word-swapping attacks
        case TextAttackAttackConfig():
            return TextAttackAttack(
                attack_config=attack_config,
                model=victim_model,
                tokenizer=victim_tokenizer,
            )
        # RL-based attacks
        case TRLAttackConfig():
            return TRLAttack(
                attack_config=attack_config,
                logging_name=logging_name,
                victim_model=victim_model,
                victim_tokenizer=victim_tokenizer,
            )
        case _:
            raise ValueError(f"Type of attack config {attack_config} not recognized.")
