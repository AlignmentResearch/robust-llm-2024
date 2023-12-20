from dataclasses import dataclass
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing import Optional

from robust_llm.configs import TrainingConfig
from robust_llm.dataset_management.tomita import TomitaBase
from robust_llm.dataset_management.tomita.tomita_dataset_generator import (
    get_tomita_dataset,
)
from robust_llm.dataset_management.tensor_trust.tensor_trust_dataset_generator import (
    get_tensor_trust_dataset,
)

from robust_llm.utils import tokenize_dataset


@dataclass
class RobustLLMDatasets:
    train_dataset: Dataset
    validation_dataset: Dataset

    tokenized_train_dataset: Dataset
    tokenized_validation_dataset: Dataset


def generate_robust_llm_datasets(
    dataset_type: str,
    language_generator: Optional[TomitaBase],
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingConfig,
    dataset_generation_style: str,
) -> RobustLLMDatasets:
    if dataset_generation_style not in ["random_words", "random_character_edit"]:
        raise ValueError(
            f"Unknown dataset_generation_style {dataset_generation_style}, exiting..."
        )

    if dataset_type.lower() == "tensor_trust":
        train_set, validation_set = get_tensor_trust_dataset(
            training_args=training_args,
            tokenizer=tokenizer,
            dataset_generation_style=dataset_generation_style,
        )

    elif dataset_type.lower() == "tomita":
        assert language_generator is not None and isinstance(
            language_generator, TomitaBase
        )
        if dataset_generation_style == "random_character_edit":
            raise ValueError(
                "Random character edit is not yet supported for Tomita datasets."
            )
        train_set, validation_set = get_tomita_dataset(
            training_args, language_generator, tokenizer
        )

    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    print("Tokenizing datasets...")
    # TODO: is the below necessary? Seems like the datasets are already tokenized by now.
    tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    tokenized_validation_dataset = Dataset.from_dict(
        tokenize_dataset(validation_set, tokenizer)
    )
    return RobustLLMDatasets(
        train_dataset=train_set,
        validation_dataset=validation_set,
        tokenized_train_dataset=tokenized_train_dataset,
        tokenized_validation_dataset=tokenized_validation_dataset,
    )
