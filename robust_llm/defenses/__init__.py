"""Defenses to adversarial attack."""

from typing import Optional

from datasets import Dataset

from robust_llm.config.configs import DefenseConfig
from robust_llm.config.defense_configs import (
    ParaphraseDefenseConfig,
    PerplexityDefenseConfig,
    RetokenizationDefenseConfig,
)
from robust_llm.defenses.paraphrase import ParaphraseDefendedModel
from robust_llm.defenses.perplexity import PerplexityDefendedModel
from robust_llm.defenses.retokenization import RetokenizationDefendedModel
from robust_llm.models import WrappedModel


def make_defended_model(
    victim: WrappedModel,
    defense_config: DefenseConfig,
    dataset: Optional[Dataset] = None,
):
    """Factory to create defended model."""

    # This match-case statement uses class patterns, as described in this SO
    # answer: https://stackoverflow.com/a/67524642
    match defense_config:
        case ParaphraseDefenseConfig():
            return ParaphraseDefendedModel(
                victim=victim,
                defense_config=defense_config,
            )
        case PerplexityDefenseConfig():
            assert dataset is not None
            return PerplexityDefendedModel(
                victim=victim,
                defense_config=defense_config,
                dataset=dataset,
            )

        case RetokenizationDefenseConfig():
            return RetokenizationDefendedModel(
                victim=victim,
                defense_config=defense_config,
            )
        case _:
            raise ValueError(f"Unrecognized defense config type for: {defense_config}")
