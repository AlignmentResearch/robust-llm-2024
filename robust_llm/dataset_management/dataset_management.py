from dataclasses import dataclass

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerBase,
)

from robust_llm.configs import TrainingConfig
from robust_llm.dataset_management.tomita import TomitaBase
from robust_llm.dataset_management.tomita.tomita_dataset_generator import (
    load_adversarial_dataset,
)

from robust_llm.utils import tokenize_dataset


@dataclass
class RobustLLMDatasets:
    train_dataset: Dataset
    validation_dataset: Dataset

    tokenized_train_dataset: Dataset
    tokenized_validation_dataset: Dataset


def generateRobustLLMDatasets(
    language_generator: TomitaBase,
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingConfig,
) -> RobustLLMDatasets:
    if training_args.baseline.non_iterative_baseline:
        brute_force_dataset = load_adversarial_dataset(
            language_generator.name,
            training_args.iterative.brute_force_length,
        )
        tokenized_brute_force_dataset = Dataset.from_dict(
            tokenize_dataset(brute_force_dataset, tokenizer)
        )
        shuffled_brute_force_dataset = tokenized_brute_force_dataset.shuffle()
        train_set = shuffled_brute_force_dataset.select(
            range(
                int(
                    training_args.baseline.proportion
                    * len(tokenized_brute_force_dataset)
                )
            )
        )
        validation_set = brute_force_dataset

    else:
        train_set, validation_set, _ = language_generator.generate_dataset(
            train_size=training_args.train_set_size,
            validation_size=training_args.validation_set_size,
            test_size=0,
        )

    print("Tokenizing datasets...")
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
