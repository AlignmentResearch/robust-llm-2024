import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from robust_llm.config.attack_configs import AttackConfig
from robust_llm.config.constants import SHARED_DATA_DIR
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.defense_configs import DefenseConfig
from robust_llm.config.model_configs import ModelConfig


@dataclass
class EnvironmentConfig:
    """
    Configs used in environment setup.

    Attributes:
        device (str): Device to use for models.
        test_mode (bool): Whether or not we're currently testing
        logging_level:
            Logging level to use for console handler.
            Choose among logging.DEBUG, logging.INFO,
            logging.WARNING, logging.ERROR, logging.CRITICAL.
        logging_filename: If set, logs will be saved to this file.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_mode: bool = False
    logging_level: int = logging.INFO
    logging_filename: str = "robust_llm.log"


@dataclass
class AdversarialTrainingConfig:
    """
    Configs used in adversarial training.

    Attributes:
        num_examples_to_generate_each_round (int): The number of adversarial examples to
            generate each round for training.
        num_examples_to_log_to_wandb_each_round (int): The number of adversarial
            examples to log to wandb each round.
        only_add_successful_adversarial_examples (bool):
            Whether to add only successful adversarial examples to training set;
            otherwise, add all trials, successful or not.
        num_adversarial_training_rounds (int):
            The number of adversarial training rounds to do.
        skip_first_training_round (bool):
            Whether to skip the first training round or not.
        training_attack (AttackConfig):
            Config for the attack to use in adversarial training.
    """

    num_examples_to_generate_each_round: int = 500
    num_examples_to_log_to_wandb_each_round: int = 10
    only_add_successful_adversarial_examples: bool = False
    num_adversarial_training_rounds: int = 3
    skip_first_training_round: bool = False
    use_balanced_sampling: bool = False
    training_attack: AttackConfig = field(default_factory=AttackConfig)


@dataclass
class TrainingConfig:
    """Configs used across different training procedures.

    Attributes:
        adversarial (AdversarialTrainingConfig): Configs for adversarial training.
        num_train_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate to use in training.
        batch_size (int): Batch size to use in training (PER DEVICE!).
        eval_steps (Optional[int | float]): Number of update steps between two
            evaluations. Will default to the same value as logging_steps if not set.
            Should be an integer or a float in range [0,1). If smaller than 1, will
            be interpreted as ratio of total training steps.
        logging_steps (int | float): Number of update steps between two logs. Should
            be an integer or a float in range [0,1). If smaller than 1, will be
            interpreted as ratio of total training steps.
        save_strategy (str): The checkpoint save strategy to adopt during training.
            Possible values are:
            - "no": No save is done during training.
            - "epoch": Save is done at the end of each epoch.
            - "steps": Save is done every save_steps.
        save_steps (int | float): Number of updates steps before two checkpoint saves if
            save_strategy="steps". Should be an integer or a float in range [0,1). If
            smaller than 1, will be interpreted as ratio of total training steps.
        log_full_datasets_to_wandb (bool): Whether to log full datasets to wandb. Off
            by default, as it takes a lot of space.
        model_save_path_prefix_or_hf (Optional[str]): Where to save the final
            checkpoint. If None, the model is not saved. If "hf", the model is saved to
            HuggingFace. Otherwise, the model is saved to a location starting with the
            specified prefix.
        seed: seed to use for training. It will be set at the beginning of huggingface
            Trainer's training. In particular, it may affect random initialization
            (if any).
    """

    adversarial: Optional[AdversarialTrainingConfig] = None
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = False
    eval_steps: Optional[int] = None
    logging_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    log_full_datasets_to_wandb: bool = False
    model_save_path_prefix_or_hf: Optional[str] = SHARED_DATA_DIR
    force_name_to_save: Optional[str] = None
    seed: int = 0

    def __post_init__(self):
        assert self.num_train_epochs > 0, "Number of training epochs must be positive."


@dataclass
class EvaluationConfig:
    """Configs used in evaluation.

    Attributes:
        batch_size (int): The mini-batch size used to iterate over the dataset when
            evaluating (PER DEVICE!).
        evaluation_attack (AttackConfig): Config for the attack to use in evaluation.
        num_examples_to_log_detailed_info (Optional[int]): Number of adversarial
            examples for which we want to log detailed info, such as the original and
            attacked text, attack results and debug info. If None, do not log anything.
        final_success_binary_callback (str): The name of the ScoringCallback to use
            for final evaluation. Should refer to a BinaryCallback, because we need
            discrete success/failure for each attacked input.
    """

    batch_size: int = 8
    evaluation_attack: AttackConfig = MISSING
    num_examples_to_log_detailed_info: Optional[int] = 10
    final_success_binary_callback: str = "successes_from_text"


@dataclass
class ExperimentConfig:
    """
    Configs used in the experiment.

    Attributes:
        experiment_type (str): Type of experiment, from the ones defined in __main__.py.
        experiment_name (str): Name of the overarching experiment. Used to set a "group"
            in wandb. Each experiment can have several jobs.
        job_type (str): Name of the sub-experiment.
        run_name (str): Name of the individual run.
        environment (EnvironmentConfig): Configs for environment setup.
        training (Optional[TrainingConfig]): Configs for training.
        evaluation (EvaluationConfig): Configs for evaluation.
        defense (DefenseConfig): Configs for defense setup.
    """

    experiment_type: str = MISSING
    experiment_name: str = "default-experiment"
    job_type: str = "default-job"
    run_name: str = "default-run"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    training: Optional[TrainingConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    defense: Optional[DefenseConfig] = None

    def __post_init__(self):
        assert self.experiment_type in ["training", "evaluation"]
        if self.experiment_type == "training":
            assert self.training is not None
        if self.experiment_type == "evaluation":
            assert self.evaluation is not None


cs = ConfigStore.instance()
cs.store(name="DEFAULT", group="training", node=TrainingConfig)
cs.store(name="DEFAULT", group="evaluation", node=EvaluationConfig)
cs.store(name="DEFAULT", group="training/adversarial", node=AdversarialTrainingConfig)
