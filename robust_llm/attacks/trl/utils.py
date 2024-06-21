from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import PPOConfig, PPOTrainer

from robust_llm import logger
from robust_llm.config.attack_configs import TRLAttackConfig
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


def make_ppo_trainer(
    attack_config: TRLAttackConfig,
    adversary_model: PreTrainedModel,
    adversary_tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
) -> PPOTrainer:
    ppo_config = {
        "exp_name": "rl-adversary",
        "batch_size": attack_config.batch_size,
        "mini_batch_size": attack_config.mini_batch_size,
        "gradient_accumulation_steps": attack_config.gradient_accumulation_steps,
        "learning_rate": attack_config.learning_rate,
        "seed": attack_config.seed,
        "ppo_epochs": attack_config.ppo_epochs,
        # Needed in order to not delete "text_chunked"
        "remove_unused_columns": False,
        # TODO(niki): log locally too?
        "log_with": "wandb",
    }
    config = PPOConfig(**ppo_config)  # type: ignore

    logger.debug("Creating PPOTrainer with config:\n %s", config)
    logger.debug("The dataset is:\n %s", dataset)
    logger.debug("The dataset size is:\n %s", len(dataset["text"]))

    ppo_trainer = PPOTrainer(
        config=config,
        model=adversary_model,  # type: ignore
        tokenizer=adversary_tokenizer,
        dataset=dataset,
        # Needed to properly process "chunked_text"
        data_collator=trl_data_collator,
    )
    assert isinstance(ppo_trainer, PPOTrainer)
    return ppo_trainer


def trl_data_collator(datapoints: Sequence[Any]) -> Mapping[str, Any]:
    """A simple data collator to use when the dataset contains "chunked_text".

    The default PPOTrainer data collator does not support
    collating lists of lists cleanly, so it is necessary to either
    use our own or modify the output of the default data collator
    after the fact.
    """
    assert all("gen_target" in datapoint for datapoint in datapoints)
    assert all("clf_label" in datapoint for datapoint in datapoints)
    assert all("text" in datapoint for datapoint in datapoints)
    assert all("chunked_text" in datapoint for datapoint in datapoints)

    batch = {}
    batch["gen_target"] = [datapoint["gen_target"] for datapoint in datapoints]
    batch["clf_label"] = [datapoint["clf_label"] for datapoint in datapoints]
    batch["text"] = [datapoint["text"] for datapoint in datapoints]
    batch["chunked_text"] = [datapoint["chunked_text"] for datapoint in datapoints]
    return batch


def prepare_adversary(
    attack_config: TRLAttackConfig,
    num_classes: int,
    accelerator: Accelerator,
) -> WrappedModel:
    assert attack_config.adversary is not None
    assert attack_config.adversary.inference_type == "trl"
    adversary = WrappedModel.from_config(
        config=attack_config.adversary,
        accelerator=accelerator,
        num_classes=num_classes,
    )

    return adversary


def prepare_prompts(
    text_chunked: Sequence[Sequence[str]],
    modifiable_chunk_spec: ModifiableChunkSpec,
    response_text: str | Sequence[str],
) -> list[str]:
    """Prepare prompts either for the adversary model or for the victim model.

    For a given sequence of chunks of text, works by concatenating all the chunks
    and replacing the modifiable chunk with the response text. If a single response
    is provided, it is used for all the chunks. If a sequence of responses is
    provided, each response is used for the corresponding chunk.

    Args:
        text_chunked: Sequence of chunked texts. Each chunked text is
            a sequence of strings, one of which is modifiable, the others
            of which are not, as determined by modifiable_chunk_spec.
        modifiable_chunk_spec: Specification for which chunks of the original text can
            be modified, and how.
        response_text:
            The text or sequence of texts to replace the modifiable chunks with.

    Returns:
        A sequence of prompts with the modifiable chunks replaced by (or
        appended with) the response text.
    """
    one_response = True
    if not isinstance(response_text, str):
        assert len(response_text) == len(text_chunked)
        one_response = False

    contexts = []
    for i, line in enumerate(text_chunked):
        context_list = []
        for text, chunk_type in zip(line, modifiable_chunk_spec):
            if chunk_type == ChunkType.IMMUTABLE:
                context_list.append(text)
            else:
                replacement_text = response_text if one_response else response_text[i]
                assert isinstance(replacement_text, str)
                if chunk_type == ChunkType.PERTURBABLE:
                    context_list.append(text)
                context_list.append(replacement_text)
        contexts.append("".join(context_list))

    return contexts


def check_for_not_finite(prepended_train_stats: dict[str, Any]) -> None:
    """Check for not finite values in the training stats."""
    for key, value in prepended_train_stats.items():
        if not np.isfinite(value).all():
            raise ValueError(
                f"Training stats contain non-finite values for key {key}: {value}"
            )
