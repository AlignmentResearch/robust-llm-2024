from dataclasses import dataclass

import hydra
from datasets import Dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
import yaml

import wandb
from robust_llm.dataset_management.tomita import TomitaBase, make_language_generator
from robust_llm.dataset_management.tomita.tomita_dataset_generator import (
    load_adversarial_dataset,
)
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import get_overlap, tokenize_dataset


@dataclass
class NonAdversarialBaselineTrainingConfig:
    # The proportion of the brute force dataset to use for training, when running a baseline.
    proportion: float = 0.1
    # Whether to run a non-adversarial baseline or not.
    non_adversarial_baseline: bool = False


@dataclass
class AdversarialTrainingConfig:
    """Configs used in adversarial training."""

    # Whether to use adversarial training.
    adversarial_training: bool = False
    # The minimum number of adversarial examples to add to the train set each attack round.
    min_num_adversarial_examples_to_add: int = 50
    # The maximum number of examples to search for adversarial examples in each attack round. Think 'compute budget'.
    max_num_search_for_adversarial_examples: int = 8192
    # The size of the minibatches to use when searching for adversarial examples.
    adversarial_example_search_minibatch_size: int = 64
    # The number of adversarial training rounds to do.
    num_adversarial_training_rounds: int = 3
    # If true, only checks robustness on a random subset of the brute force attack dataset.
    use_probabilistic_robustness_check: bool = False
    # Whether to skip the first training round or not.
    skip_first_training_round: bool = False

    # Up to which length strings should be exhaustively tested.
    brute_force_length: int = 5
    # Whether to exhaustively test all possible adversarial examples.
    brute_force_attack: bool = False


@dataclass
class EnvironmentConfig:
    """Configs used in environment setup."""

    # Choose the regular language to use (tomita1, tomita2, tomita4, tomita7).
    language_generator: str = "tomita4"
    # The maximum length of the strings to generate.
    max_length: int = 50
    # The seed to use for the random number generator used to make the dataset
    seed: int = 0


@dataclass
class TrainingConfig:
    """Configs used by multiple training types."""

    adversarial: AdversarialTrainingConfig = AdversarialTrainingConfig()
    baseline: NonAdversarialBaselineTrainingConfig = (
        NonAdversarialBaselineTrainingConfig()
    )
    # The size of the train set.
    train_set_size: int = 100
    # The size of the validation set.
    validation_set_size: int = 100
    # The number of epochs to train for.
    num_train_epochs: int = 3


# TODO(dan) guard against mutually exclusive options
@dataclass
class ExperimentConfig:
    training: TrainingConfig = TrainingConfig()

    environment: EnvironmentConfig = EnvironmentConfig()


@dataclass
class OverallConfig:
    experiment: ExperimentConfig = ExperimentConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=OverallConfig)


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
    if training_args.baseline.non_adversarial_baseline:
        brute_force_dataset = load_adversarial_dataset(
            language_generator.name,
            training_args.adversarial.brute_force_length,
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


@hydra.main(version_base=None, config_path="hydra_conf", config_name="default_config")
def main(args: OverallConfig) -> None:
    experiment = args.experiment
    print("Configuration arguments:\n")
    print(OmegaConf.to_yaml(experiment))
    print()

    language_generator = make_language_generator(
        experiment.environment.language_generator, experiment.environment.max_length
    )

    # Choose a model and a tokenizer
    model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    robust_llm_datasets = generateRobustLLMDatasets(
        language_generator, tokenizer, experiment.training
    )

    # NOTE: a confusing thing: the "validation" dataset is one of what will be
    # several datasets that we perform model evaluation on, hence "eval_dataset"
    # is a dict[str, Dataset] and not a Dataset.
    base_training_args = {
        "hparams": {},
        "train_dataset": robust_llm_datasets.tokenized_train_dataset,
        "eval_dataset": {
            "validation": robust_llm_datasets.tokenized_validation_dataset
        },
        "model": model,
        "train_epochs": experiment.training.num_train_epochs,
    }

    # Set up the training environment
    training: Training
    if experiment.training.adversarial.adversarial_training:
        training = AdversarialTraining(
            **base_training_args,
            num_adversarial_training_rounds=experiment.training.adversarial.num_adversarial_training_rounds,
            tokenizer=tokenizer,
            language_generator_name=experiment.environment.language_generator,
            brute_force_attack=experiment.training.adversarial.brute_force_attack,
            brute_force_length=experiment.training.adversarial.brute_force_length,
            min_num_adversarial_examples_to_add=experiment.training.adversarial.min_num_adversarial_examples_to_add,
            max_num_search_for_adversarial_examples=experiment.training.adversarial.max_num_search_for_adversarial_examples,
            adversarial_example_search_minibatch_size=experiment.training.adversarial.adversarial_example_search_minibatch_size,
            skip_first_training_round=experiment.training.adversarial.skip_first_training_round,
            use_probabilistic_robustness_check=experiment.training.adversarial.use_probabilistic_robustness_check,
            non_adversarial_baseline=experiment.training.baseline.non_adversarial_baseline,
        )
    else:
        training = Training(
            **base_training_args,
        )

    # Log the training arguments to wandb
    if not wandb.run:
        raise ValueError("wandb should have been initialized by now, exiting...")
    yaml_string = yaml.load(OmegaConf.to_yaml(experiment), Loader=yaml.FullLoader)
    wandb.run.summary[f"experiment_yaml"] = yaml_string

    # Log the train-val overlap to wandb
    if (
        experiment.training.train_set_size > 0
        and experiment.training.validation_set_size > 0
    ):
        if not wandb.run:
            raise ValueError("wandb should have been initialized by now, exiting...")
        train_val_overlap = get_overlap(
            smaller_dataset=robust_llm_datasets.validation_dataset,
            larger_dataset=robust_llm_datasets.train_dataset,
        )
        wandb.run.summary["train_val_overlap_size"] = len(train_val_overlap)
        wandb.run.summary["train_val_overlap_over_train_set_size"] = len(
            train_val_overlap
        ) / len(robust_llm_datasets.train_dataset["text"])
        wandb.run.summary["train_val_overlap_over_val_set_size"] = len(
            train_val_overlap
        ) / len(robust_llm_datasets.validation_dataset["text"])

    # Perform the training
    training.run_trainer()


if __name__ == "__main__":
    main()
