from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from robust_llm.configs import EnvironmentConfig, TrainingConfig
from robust_llm.dataset_management.tensor_trust.tensor_trust_dataset_generator import (
    TENSOR_TRUST_MODIFIABLE_CHUNKS_SPEC,
    get_tensor_trust_dataset,
)
from robust_llm.dataset_management.tomita import Tomita
from robust_llm.dataset_management.tomita.tomita_dataset_generator import (
    get_tomita_dataset,
)
from robust_llm.utils import tokenize_dataset

# Tuple of bools specifying which chunks of the original text can be modified.
# For example, when (True,), the whole text can be modified. When (False, True),
# only the second part of the text can be modified for each example.
ModifiableChunksSpec = Tuple[bool, ...]


@dataclass
class RobustLLMDatasets:
    train_dataset: Dataset
    validation_dataset: Dataset

    tokenized_train_dataset: Dataset
    tokenized_validation_dataset: Dataset

    # Specification for which chunks of the original text can be modified
    modifiable_chunks_spec: ModifiableChunksSpec = (True,)


def generate_robust_llm_datasets(
    dataset_type: str,
    language_generator: Optional[Tomita],
    tokenizer: PreTrainedTokenizerBase,
    environment_config: EnvironmentConfig,
    training_config: TrainingConfig,
    dataset_generation_style: str,
    seed: int,
) -> RobustLLMDatasets:
    if dataset_generation_style not in ["random_words", "random_character_edit"]:
        raise ValueError(
            f"Unknown dataset_generation_style {dataset_generation_style}, exiting..."
        )

    modifiable_chunks_spec: tuple[bool, ...] = (True,)

    if dataset_type == "tensor_trust":
        train_set, validation_set = get_tensor_trust_dataset(
            environment_config=environment_config,
            tokenizer=tokenizer,
            dataset_generation_style=dataset_generation_style,
        )

        modifiable_chunks_spec = TENSOR_TRUST_MODIFIABLE_CHUNKS_SPEC

    elif dataset_type == "tomita":
        assert language_generator is not None and isinstance(language_generator, Tomita)
        if dataset_generation_style == "random_character_edit":
            raise ValueError(
                "Random character edit is not yet supported for Tomita datasets."
            )
        train_set, validation_set = get_tomita_dataset(
            environment_config=environment_config,
            training_config=training_config,
            language_generator=language_generator,
            tokenizer=tokenizer,
        )

    elif dataset_type.startswith("hf/"):
        dataset_name = dataset_type[len("hf/") :]
        hf_dataset = load_dataset(dataset_name)
        assert isinstance(hf_dataset, (DatasetDict, IterableDatasetDict))
        train_set, validation_set = hf_dataset["train"], hf_dataset["test"]

    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    if environment_config.shuffle_train_set:
        train_set = train_set.shuffle(seed=seed)
    if environment_config.shuffle_validation_set:
        validation_set = validation_set.shuffle(seed=seed)

    if environment_config.train_set_size is not None:
        train_set = train_set.select(range(environment_config.train_set_size))
    if environment_config.validation_set_size is not None:
        validation_set = validation_set.select(
            range(environment_config.validation_set_size)
        )

    print("Tokenizing datasets...")
    # TODO: is the below necessary?
    # Seems like the datasets are already tokenized by now.
    tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    tokenized_validation_dataset = Dataset.from_dict(
        tokenize_dataset(validation_set, tokenizer)
    )
    return RobustLLMDatasets(
        train_dataset=train_set,
        validation_dataset=validation_set,
        tokenized_train_dataset=tokenized_train_dataset,
        tokenized_validation_dataset=tokenized_validation_dataset,
        modifiable_chunks_spec=modifiable_chunks_spec,
    )
