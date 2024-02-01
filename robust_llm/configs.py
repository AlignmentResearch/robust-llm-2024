from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

SHARED_DATA_DIR = "/robust_llm_data"


@dataclass
class BaselineTrainingConfig:
    """Configs used in baseline training."""

    # The proportion of the brute force dataset to use for training, when
    # running a baseline.
    proportion: float = 0.1
    # Whether to run a non-iterative baseline or not.
    non_iterative_baseline: bool = False


@dataclass
class TextAttackAttackConfig:
    """Options specific for TextAttack attacks."""

    # Query budget per example.
    query_budget: int = 100
    # Number of examples to attack. If -1, attack whole dataset.
    num_examples: int = -1


@dataclass
class BruteForceTomitaAttackConfig:
    """Options specific for BruteForceTomita attacks."""

    # Up to which length strings should be exhaustively tested.
    length: int = 5


@dataclass
class RandomTokenAttackConfig:
    """Options specific for RandomToken attacks."""

    # Minimum number of tokens to generate.
    min_tokens: int = 1
    # Maximum number of tokens to generate.
    max_tokens: int = 3


@dataclass
class AttackConfig:
    """Configs used in attack setup."""

    # The type of attack to use.
    attack_type: str = "identity"
    # Whether to repeat the attack every iterative training round or not.
    repeat_attack_every_round: bool = True
    # Random seed for the attack.
    seed: int = 0

    # Configs for specific types of attacks.
    brute_force_tomita_attack_config: BruteForceTomitaAttackConfig = (
        BruteForceTomitaAttackConfig()
    )
    text_attack_attack_config: TextAttackAttackConfig = TextAttackAttackConfig()
    random_token_attack_attack_config: RandomTokenAttackConfig = (
        RandomTokenAttackConfig()
    )


@dataclass
class IterativeTrainingConfig:
    """Configs used in iterative (often adversarial) training."""

    # Whether to use iterative training.
    iterative_training: bool = False
    # Whether to add strictly adversarial examples or not
    only_add_successful_adversarial_examples: bool = True
    # The minimum number of adversarial examples to add to the train set each
    # attack round.
    min_num_new_examples_to_add: int = 50
    # The maximum number of examples to search for adversarial examples in each
    # attack round. Think 'compute budget'.
    max_num_search_for_adversarial_examples: int = 8192
    # The size of the minibatches to use when searching for adversarial
    # examples.
    adversarial_example_search_minibatch_size: int = 64
    # The number of adversarial training rounds to do.
    num_iterative_training_rounds: int = 3
    # If true, only checks robustness on a random subset of the brute force
    # attack dataset.
    use_probabilistic_robustness_check: bool = False
    # Whether to skip the first training round or not.
    skip_first_training_round: bool = False
    # Config for the attack to use in adversarial training.
    training_attack: AttackConfig = AttackConfig()


@dataclass
class EnvironmentConfig:
    """Configs used in environment setup (including dataset)."""

    # Either HF name or path to model checkpoint.
    model_name_or_path: str = "bert-base-uncased"
    # Whether the architecture is Pythia or not. Needed for loading the model.
    is_pythia: bool = False
    # Dataset type (tomita, tensor_trust)
    dataset_type: str = "tomita"
    # How to generate the negative examples in the dataset
    # (only works with tensor trust for now)
    dataset_generation_style: str = (
        "random_words"  # random_word / random_character_edit
    )
    # Choose the regular language to use (tomita1, tomita2, tomita4, tomita7).
    language_generator: str = "tomita4"
    # The maximum length of the strings to generate.
    max_length: int = 50
    # The seed to use for the random number generator used to make the dataset
    seed: int = 0
    # The size of the train set. For generated datasets, must be set to positive
    # integer. For HF datasets, can be set to None to use the full dataset.
    train_set_size: Optional[int] = None
    # The size of the validation set. For generated datasets, must be set to positive
    # integer. For HF datasets, can be set to None to use the full dataset.
    validation_set_size: Optional[int] = None
    # Whether to shuffle the train set. Can matter if we subsample.
    shuffle_train_set: bool = False
    # Whether to shuffle the validation set. Can matter if we subsample.
    shuffle_validation_set: bool = False
    # The number of epochs to train for.


@dataclass
class TrainingConfig:
    """Configs used across different training procedures."""

    iterative: IterativeTrainingConfig = IterativeTrainingConfig()
    baseline: BaselineTrainingConfig = BaselineTrainingConfig()
    num_train_epochs: int = 3
    # Number of update steps between two evaluations
    eval_steps: Optional[int] = None
    # Number of update steps between two logs
    logging_steps: int | float = 500
    # The checkpoint to start from (relevant for Pythia only, for now)
    checkpoint: int = 142000
    # Whether to log datasets to wandb. Off by default, as it takes a lot of space.
    # For now, works only for the training pipeline.
    log_datasets_to_wandb: bool = False
    # Where to save the final checkpoint. If None, the model is not saved.
    # If "hf", the model is saved to HuggingFace. Otherwise, the model is
    # saved to a location starting with the specified prefix.
    model_save_path_prefix_or_hf: Optional[str] = SHARED_DATA_DIR


@dataclass
class EvaluationConfig:
    # The mini-batch size used to iterate over the dataset when evaluating.
    batch_size: int = 4
    # Config for the attack to use in evaluation.
    evaluation_attack: AttackConfig = AttackConfig()
    # The number of examples to generate for the evaluation attack. Should be specified
    # only if the attack does not require an input dataset. Otherwise, we want the
    # attack to generate an example per each sample in the input dataset.
    num_generated_examples: Optional[int] = None


# TODO(dan) guard against mutually exclusive options
@dataclass
class ExperimentConfig:
    # Experiment type, from the ones defined in __main__.py
    experiment_type: str = MISSING

    # The name of the overarching experiment being run. Used to set a "group" in
    # wandb. Each experiment has several jobs.
    # Example: "scaling-model-size_2023-11-22_1e88j"
    experiment_name: str = "default-experiment"

    # The name of the sub-experiment being run. Used to set a "job_type" in wandb.
    # Should correspond to one specific sub-experiment.
    # Each job can have several runs (with different seeds).
    # Example: "pythia-14m_step17000"
    job_type: str = "default-job"

    # The name of the individual run.
    # Don't need much here since group and job do most of the work distinguishing.
    # Random string is fine.
    # Example: "run_3f4ay"
    run_name: str = "default-run"

    environment: EnvironmentConfig = EnvironmentConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


@dataclass
class OverallConfig:
    experiment: ExperimentConfig = ExperimentConfig()
