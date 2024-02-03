"""Defenses to adversarial attack."""

from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.configs import DefenseConfig
from robust_llm.defenses.defense import DefendedModel, Defenses
from robust_llm.defenses.paraphrase import ParaphraseDefendedModel
from robust_llm.defenses.perplexity import PerplexityDefendedModel
from robust_llm.defenses.retokenization import RetokenizationDefendedModel
from robust_llm.utils import LanguageModel


def make_defended_model(
    defense_config: DefenseConfig,
    init_model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Optional[Dataset] = None,
    decoder: Optional[PreTrainedModel] = None,
):
    """Factory to create defended model."""
    DEFENSE_MAP = {
        Defenses.IDENTITY.value: DefendedModel,
        Defenses.PERPLEXITY.value: PerplexityDefendedModel,
        Defenses.RETOKENIZATION.value: RetokenizationDefendedModel,
        Defenses.PARAPHRASE.value: ParaphraseDefendedModel,
    }
    defense_type = defense_config.defense_type
    try:
        model_cls = DEFENSE_MAP[defense_type]
    except KeyError as exc:
        raise ValueError(f"Invalid defense type: {defense_type}") from exc
    return model_cls(
        defense_config=defense_config,
        init_model=init_model,
        tokenizer=tokenizer,
        dataset=dataset,
        decoder=decoder,
    )
