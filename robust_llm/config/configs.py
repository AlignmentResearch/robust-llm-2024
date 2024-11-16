import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI

from robust_llm.config.attack_configs import AttackConfig
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.defense_configs import DefenseConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.file_utils import get_save_root
from robust_llm.utils import deterministic_hash_config


@dataclass
class EnvironmentConfig:
    """
    Configs used in environment setup.

    Attributes:
        device: Device to use for models.
        test_mode: Whether or not we're currently testing
        save_root: Prefix to use for the local artifacts directory.
        minibatch_multiplier: Multiplier for the minibatch size.
            This is useful if we want to set default batch sizes for models in the
            ModelConfig but then adjust all of these values based on the GPU memory
            available or the dataset we're attacking.
        logging_level:
            Logging level to use for console handler.
            Choose among logging.DEBUG, logging.INFO,
            logging.WARNING, logging.ERROR, logging.CRITICAL.
        logging_filename: If set, logs will be saved to this file.
        wandb_info_filename: Log the W&B run name + ID to this file. Use this if
          you need to programatically get the run name and ID after running
          a job.
        allow_checkpointing: Whether to allow checkpointing during training and also
            attacks that support it.
        resume_mode: How often to resume from checkpoint during training.
            - "once": Resume from checkpoint and then run as normal
            - "always": Resume from checkpoint at the beginning of each epoch.
                The "always" mode is useful for debugging and ensuring determinism.
        deterministic: Whether to force use of deterministic CUDA algorithms at the
            cost of performance.
        cublas_config: The configuration string for cuBLAS, only used if
            deterministic is True.
            See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
            for why this is necessary.

    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_mode: bool = False
    save_root: str = get_save_root()
    minibatch_multiplier: float = 1.0
    logging_level: int = logging.INFO
    logging_filename: str = "robust_llm.log"
    wandb_info_filename: str | None = None
    allow_checkpointing: bool = True
    resume_mode: str = "once"
    deterministic: bool = True
    cublas_config: str = ":4096:8"


@dataclass
class AttackScheduleConfig:
    """Linear schedule for increasing attack iterations during training."""

    start: int | None = None
    end: int | None = None
    rate: float | None = None

    def __post_init__(self):
        assert any(
            [
                self.start is None,
                self.end is None,
                self.rate is None,
            ]
        ), "At least one of start, end, and rate must be implied."
        assert self.start is None or self.start >= 1, "iterations must be >= 1."
        assert self.end is None or self.end >= 1, "iterations must be >= 1."


@dataclass
class AdversarialTrainingConfig:
    """
    Configs used in adversarial training.

    Attributes:
        num_examples_to_generate_each_round: The number of adversarial examples to
            generate each round for training.
        num_examples_to_log_to_wandb_each_round: The number of adversarial
            examples to log to wandb each round.
        loss_rank_weight:
            The weight to give to the rank of the loss when ranking adversarial
            examples for training. Should be [0, 1] as we allocate the rest of the
            weight to recency.
        num_adversarial_training_rounds:
            The number of adversarial training rounds to do.
        training_attack:
            Config for the attack to use in adversarial training.
        max_adv_data_proportion:
            The maximum percentage of the training data made up of adversarial
            examples.
        max_augmented_data_size:
            The maximum number of datapoints to use for adversarial training.
        adv_sampling_decay:
            The decay factor for the sampling probability of adversarial examples.
        attack_schedule:
            The linear schedule for increasing the number of attack iterations during
            adversarial training.
        evaluate_during_training:
            Whether to evaluate the model during adversarial training.
    """

    num_examples_to_generate_each_round: int = 500
    num_examples_to_log_to_wandb_each_round: int = 10
    loss_rank_weight: float = 0.0
    num_adversarial_training_rounds: int = 3
    # skip_first_training_round: bool = False
    # use_balanced_sampling: bool = False
    training_attack: AttackConfig = field(default_factory=AttackConfig)
    max_adv_data_proportion: float = 0.5
    max_augmented_data_size: int = SI("${dataset.n_train}")
    adv_sampling_decay: float = 0.0
    # stopping_attack_success_rate: float = 0.0
    # target_adversarial_success_rate: Optional[float] = None
    attack_schedule: AttackScheduleConfig = field(
        default_factory=lambda: AttackScheduleConfig(start=10)
    )
    # stopping_flops: float = float("inf")
    evaluate_during_training: bool = False

    def __post_init__(self):
        assert 0 <= self.loss_rank_weight <= 1, "loss_rank_weight should be in [0, 1]."


class SaveTo(Enum):
    # Save to HF hub.
    HF = "hf"
    # Save to disk.
    DISK = "disk"
    # Save to both HF hub and disk.
    BOTH = "both"
    # Save to HF hub. If that fails, save to disk instead and mark it as needing
    # to be uploaded to HF.
    HF_ELSE_DISK = "hf_else_disk"
    # Model is not saved.
    NONE = "none"


@dataclass
class TrainingConfig:
    """Configs used across different training procedures.

    Attributes:
        adversarial: Configs for adversarial training.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate to use in training.
        lr_scheduler_type: The learning rate scheduler to use in training.
            Defaults to "linear", which is the Trainer default. Other options to
            consider are "constant", "constant_with_warmup", and "cosine".
            Full list here:
            At https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L410
        optimizer: The optimizer to use in training.
        save_total_limit: The maximum number of checkpoints to keep. If more than
            save_total_limit checkpoints are saved, the oldest ones are deleted.
        save_name: The name to use for saving checkpoints. If None, the name will be
            automatically generated.
        save_to: Where to save the model after running the trainer.
        seed: seed to use for training. It will be set at the beginning of huggingface
            Trainer's training. In particular, it may affect random initialization
            (if any).
        upload_retries: The number of times to retry uploading the model to the hub.
        upload_cooldown: The number of seconds to wait between retries.
    """  # noqa: E501

    adversarial: Optional[AdversarialTrainingConfig] = None
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "linear"
    # warmup_ratio: float = 0
    optimizer: str = "adamw_torch"
    # gradient_checkpointing: bool = False
    # group_by_length: bool = False
    # eval_steps: Optional[int] = None
    # logging_steps: int = 500
    # save_strategy: str = "steps"
    # save_steps: int = 500
    save_total_limit: int = 2
    # log_full_datasets_to_wandb: bool = False
    save_name: Optional[str] = None
    save_to: SaveTo = SaveTo.BOTH
    seed: int = 0
    upload_retries: int = 5
    upload_cooldown: float = 10

    def __post_init__(self):
        assert self.num_train_epochs > 0, "Number of training epochs must be positive."


@dataclass
class EvaluationConfig:
    """Configs used in evaluation.

    Attributes:
        evaluation_attack: Config for the attack to use in evaluation.
        num_iterations: Number of iterations to run the attack for.
        num_examples_to_log_detailed_info: Number of adversarial
            examples for which we want to log detailed info, such as the original and
            attacked text, attack results and debug info. If None, do not log anything.
        final_success_binary_callback: Config for the
            ScoringCallback to use for final evaluation. Should refer to a
            BinaryCallback, because we need discrete success/failure for each
            attacked input.
        compute_robustness_metric: Whether to compute the robustness metric.
        upload_artifacts: Whether to upload artifacts to wandb.
    """

    evaluation_attack: AttackConfig = MISSING
    num_iterations: int = 30
    num_examples_to_log_detailed_info: Optional[int] = 10
    final_success_binary_callback: CallbackConfig = field(
        default_factory=lambda: CallbackConfig(
            callback_name="successes_from_text", callback_return_type="binary"
        )
    )
    compute_robustness_metric: bool = True
    upload_artifacts: bool = True


@dataclass
class ExperimentConfig:
    """
    Configs used in the experiment.

    Attributes:
        experiment_type: Type of experiment, from the ones defined in __main__.py.
        experiment_name: Name of the overarching experiment. Used to set a "group"
            in wandb. Each experiment can have several jobs.
        job_type: Name of the sub-experiment.
        run_name: Name of the individual run.
        environment: Configs for environment setup.
        dataset: Configs for dataset setup.
        model: Configs for model setup.
        training (Optional[TrainingConfig]): Configs for training.
        evaluation: Configs for evaluation.
        defense: Configs for defense setup.
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
cs.store(name="DEFAULT", group="environment", node=EnvironmentConfig)
cs.store(name="DEFAULT", group="training", node=TrainingConfig)
cs.store(name="DEFAULT", group="evaluation", node=EvaluationConfig)
cs.store(name="DEFAULT", group="training/adversarial", node=AdversarialTrainingConfig)


def get_checkpoint_path(config: ExperimentConfig) -> Path:
    """Get the deterministic path for saving/loading checkpoints.

    This is designed to mirror wandb, so e.g. the run
    ian-135b-ft-pythia-harmless-0000 in group
    ian_135b_ft_pythia_harmless would have a path like
    /path/to/checkpoints/ian_135b_ft_pythia_harmless/ian-135b-ft-pythia-harmless-0000/abcdef1234.../
    """  # noqa: E501
    base_path = Path(config.environment.save_root) / "checkpoints"
    hex_hash = deterministic_hash_config(config)
    group_name = config.experiment_name
    run_name = config.run_name
    return base_path / group_name / run_name / hex_hash
