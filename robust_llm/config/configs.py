import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI

from robust_llm.config.attack_configs import AttackConfig
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.config.constants import SHARED_DATA_DIR
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.defense_configs import DefenseConfig
from robust_llm.config.model_configs import ModelConfig


@dataclass
class EnvironmentConfig:
    """
    Configs used in environment setup.

    Attributes:
        device: Device to use for models.
        test_mode: Whether or not we're currently testing
        minibatch_multiplier: Multiplier for the minibatch size.
            This is useful if we want to set default batch sizes for models in the
            ModelConfig but then adjust all of these values based on the GPU memory
            available or the dataset we're attacking.
        logging_level:
            Logging level to use for console handler.
            Choose among logging.DEBUG, logging.INFO,
            logging.WARNING, logging.ERROR, logging.CRITICAL.
        logging_filename: If set, logs will be saved to this file.
        allow_checkpointing: Whether to allow checkpointing during training and also
            attacks that support it.
        deterministic: Whether to force use of deterministic CUDA algorithms at the
            cost of performance.
        cublas_config: The configuration string for cuBLAS, only used if
            deterministic is True.
            See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
            for why this is necessary.

    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_mode: bool = False
    minibatch_multiplier: float = 1.0
    logging_level: int = logging.INFO
    logging_filename: str = "robust_llm.log"
    allow_checkpointing: bool = True
    deterministic: bool = True
    cublas_config: str = ":4096:8"


@dataclass
class AttackScheduleConfig:
    """Linear schedule for increasing attack iterations during training."""

    start: int | None = None
    end: int | None = None
    rate: float | None = None

    def __post_init__(self):
        assert (
            sum(
                [
                    self.start is None,
                    self.end is None,
                    self.rate is None,
                ]
            )
            == 1
        ), "Exactly two of start, end, and rate must be specified."
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
        skip_first_training_round:
            Whether to skip the first training round or not.
        use_balanced_sampling:
            Whether to use balanced sampling for adversarial training, i.e., sample
            alternately from the original and adversarial datasets.
        training_attack:
            Config for the attack to use in adversarial training.
        max_adv_data_proportion:
            The maximum percentage of the training data made up of adversarial
            examples.
        max_augmented_data_size:
            The maximum number of datapoints to use for adversarial training.
        adv_sampling_decay:
            The decay factor for the sampling probability of adversarial examples.
        stopping_attack_success_rate:
            The attack success rate on the validation dataset at which to stop
            adversarial training.
        target_adversarial_success_rate (float or None):
            The attack success rate on adversarial examples to target during
            adversarial training by modulating the iterations of the attack.
            If None, the attack will run for a fixed number of iterations.
        attack_schedule:
            The linear schedule for increasing the number of attack iterations during
            adversarial training.
        stopping_flops:
            The number of FLOPs to use as a stopping criterion for adversarial training.
    """

    num_examples_to_generate_each_round: int = 500
    num_examples_to_log_to_wandb_each_round: int = 10
    loss_rank_weight: float = 0.0
    num_adversarial_training_rounds: int = 3
    skip_first_training_round: bool = False
    use_balanced_sampling: bool = False
    training_attack: AttackConfig = field(default_factory=AttackConfig)
    max_adv_data_proportion: float = 0.5
    max_augmented_data_size: int = SI("${dataset.n_train}")
    adv_sampling_decay: float = 0.0
    stopping_attack_success_rate: float = 0.0
    target_adversarial_success_rate: Optional[float] = None
    attack_schedule: AttackScheduleConfig = field(
        default_factory=lambda: AttackScheduleConfig(start=10, rate=0)
    )
    stopping_flops: float = float("inf")

    def __post_init__(self):
        assert 0 <= self.loss_rank_weight <= 1, "loss_rank_weight should be in [0, 1]."


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
        warmup_ratio: Ratio of total training steps to use for learning rate
            warmup (when applicable to the LR scheduler).
        optimizer: The optimizer to use in training.
        gradient_checkpointing: Whether to use gradient checkpointing.
            This is a technique to reduce memory usage at the cost of some additional
            computation during backpropagation.
        group_by_length: Whether to group examples in batches by length, to
            speed up training by reducing padding.
        eval_steps: Number of update steps between two
            evaluations. Will default to the same value as logging_steps if not set.
            Should be an integer or a float in range [0,1). If smaller than 1, will
            be interpreted as ratio of total training steps.
        logging_steps: Number of update steps between two logs. Should
            be an integer or a float in range [0,1). If smaller than 1, will be
            interpreted as ratio of total training steps.
        save_strategy: The checkpoint save strategy to adopt during training.
            Possible values are:
            - "no": No save is done during training.
            - "epoch": Save is done at the end of each epoch.
            - "steps": Save is done every save_steps.
        save_steps: Number of updates steps before two checkpoint saves if
            save_strategy="steps". Should be an integer or a float in range [0,1). If
            smaller than 1, will be interpreted as ratio of total training steps.
        save_total_limit: The maximum number of checkpoints to keep. If more than
            save_total_limit checkpoints are saved, the oldest ones are deleted.
        log_full_datasets_to_wandb: Whether to log full datasets to wandb. Off
            by default, as it takes a lot of space.
        save_prefix: The parent directory to use for saving checkpoints.
        save_name: The name to use for saving checkpoints. If None, the name will be
            automatically generated.
        save_to: Where to save the model after running the trainer. Possible values are:
            - "hf": Save to the huggingface hub.
            - "disk": Save to disk.
            - None, the model will not be saved.
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
    warmup_ratio: float = 0
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = False
    group_by_length: bool = False
    eval_steps: Optional[int] = None
    logging_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 1
    log_full_datasets_to_wandb: bool = False
    save_prefix: str = SHARED_DATA_DIR
    save_name: Optional[str] = None
    save_to: str | None = "hf"
    seed: int = 0
    upload_retries: int = 3
    upload_cooldown: float = 5

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
