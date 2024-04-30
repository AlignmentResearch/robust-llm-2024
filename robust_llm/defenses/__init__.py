"""Defenses to adversarial attack."""

from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.config.configs import DefenseConfig
from robust_llm.config.defense_configs import (
    ParaphraseDefenseConfig,
    PerplexityDefenseConfig,
    RetokenizationDefenseConfig,
)
from robust_llm.defenses.defense import DefendedModel
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

    # This match-case statement uses class patterns, as described in this SO
    # answer: https://stackoverflow.com/a/67524642
    model_cls: type[DefendedModel]
    match defense_config:
        case ParaphraseDefenseConfig():
            model_cls = ParaphraseDefendedModel
        case PerplexityDefenseConfig():
            model_cls = PerplexityDefendedModel
        case RetokenizationDefenseConfig():
            model_cls = RetokenizationDefendedModel
        case _:
            raise ValueError(f"Unrecognized defense config type for: {defense_config}")

    return model_cls(
        defense_config=defense_config,
        init_model=init_model,
        tokenizer=tokenizer,
        dataset=dataset,
        decoder=decoder,
    )
