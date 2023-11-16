from dataclasses import dataclass


@dataclass
class BaselineTrainingConfig:
    """Configs used in baseline training."""

    # The proportion of the brute force dataset to use for training, when running a baseline.
    proportion: float = 0.1
    # Whether to run a non-iterative baseline or not.
    non_iterative_baseline: bool = False


@dataclass
class IterativeTrainingConfig:
    """Configs used in iterative (often adversarial) training."""

    # Whether to use iterative training.
    iterative_training: bool = False
    # Whether to use the non-adversarial baseline or not
    non_adversarial_baseline: bool = False
    # The minimum number of adversarial examples to add to the train set each attack round.
    min_num_new_examples_to_add: int = 50
    # The maximum number of examples to search for adversarial examples in each attack round. Think 'compute budget'.
    max_num_search_for_adversarial_examples: int = 8192
    # The size of the minibatches to use when searching for adversarial examples.
    adversarial_example_search_minibatch_size: int = 64
    # The number of adversarial training rounds to do.
    num_iterative_training_rounds: int = 3
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

    # Dataset type (tomita, tensor_trust)
    dataset_type: str = "tomita"
    # Choose the regular language to use (tomita1, tomita2, tomita4, tomita7).
    language_generator: str = "tomita4"
    # The maximum length of the strings to generate.
    max_length: int = 50
    # The seed to use for the random number generator used to make the dataset
    seed: int = 0


@dataclass
class TrainingConfig:
    """Configs used by multiple training types."""

    iterative: IterativeTrainingConfig = IterativeTrainingConfig()
    baseline: BaselineTrainingConfig = BaselineTrainingConfig()
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
