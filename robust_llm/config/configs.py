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
        device (str): Device to use for models.
        test_mode (bool): Whether or not we're currently testing
        minibatch_multiplier (float): Multiplier for the minibatch size.
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
        loss_rank_weight (float):
            The weight to give to the rank of the loss when ranking adversarial
            examples for training. Should be [0, 1] as we allocate the rest of the
            weight to recency.
        num_adversarial_training_rounds (int):
            The number of adversarial training rounds to do.
        skip_first_training_round (bool):
            Whether to skip the first training round or not.
        use_balanced_sampling (bool):
            Whether to use balanced sampling for adversarial training, i.e., sample
            alternately from the original and adversarial datasets.
        training_attack (AttackConfig):
            Config for the attack to use in adversarial training.
        max_adv_data_proportion (float):
            The maximum percentage of the training data made up of adversarial
            examples.
        max_augmented_data_size (int):
            The maximum number of datapoints to use for adversarial training.
        adv_sampling_decay (float):
            The decay factor for the sampling probability of adversarial examples.
        stopping_attack_success_rate (float):
            The attack success rate on the validation dataset at which to stop
            adversarial training.
        target_adversarial_success_rate (float or None):
            The attack success rate on adversarial examples to target during
            adversarial training by modulating the iterations of the attack.
            If None, the attack will run for a fixed number of iterations.
        min_attack_iterations (int):
            The minimum number of iterations to run the attack for.
        max_attack_iterations (int):
            The maximum number of iterations to run the attack for.
        stopping_flops (float):
            The number of FLOPs to use as a stopping criterion for adversarial training.
    """

    num_examples_to_generate_each_round: int = 500
    num_examples_to_log_to_wandb_each_round: int = 10
    only_add_successful_adversarial_examples: bool = False
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
    min_attack_iterations: int = 1
    max_attack_iterations: int = SI(
        "${mult: 10, ${training.adversarial.training_attack.initial_n_its}}"
    )
    stopping_flops: float = float("inf")

    def __post_init__(self):
        assert (
            self.min_attack_iterations > 0
        ), "Minimum number of iterations must be positive."
        assert 0 <= self.loss_rank_weight <= 1, "loss_rank_weight should be in [0, 1]."


@dataclass
class TrainingConfig:
    """Configs used across different training procedures.

    Attributes:
        adversarial (AdversarialTrainingConfig): Configs for adversarial training.
        num_train_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate to use in training.
        lr_scheduler_type (str): The learning rate scheduler to use in training.
            Defaults to "linear", which is the Trainer default. Other options to
            consider are "constant", "constant_with_warmup", and "cosine".
            Full list here:
            At https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L410
        optimizer (str): The optimizer to use in training.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.
            This is a technique to reduce memory usage at the cost of some additional
            computation during backpropagation.
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
        save_total_limit (int): The maximum number of checkpoints to keep. If more than
            save_total_limit checkpoints are saved, the oldest ones are deleted.
        log_full_datasets_to_wandb (bool): Whether to log full datasets to wandb. Off
            by default, as it takes a lot of space.
        model_save_path_prefix_or_hf (Optional[str]): Where to save the final
            checkpoint. If None, the model is not saved. If "hf", the model is saved to
            HuggingFace. Otherwise, the model is saved to a location starting with the
            specified prefix.
        force_name_to_save (Optional[str]): If set, the model will be saved with this
            name. Otherwise, the name will be automatically generated.
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
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = False
    eval_steps: Optional[int] = None
    logging_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 1
    log_full_datasets_to_wandb: bool = False
    model_save_path_prefix_or_hf: Optional[str] = SHARED_DATA_DIR
    force_name_to_save: Optional[str] = None
    seed: int = 0
    upload_retries: int = 3
    upload_cooldown: float = 5

    def __post_init__(self):
        assert self.num_train_epochs > 0, "Number of training epochs must be positive."


@dataclass
class EvaluationConfig:
    """Configs used in evaluation.

    Attributes:
        evaluation_attack (AttackConfig): Config for the attack to use in evaluation.
        num_examples_to_log_detailed_info (Optional[int]): Number of adversarial
            examples for which we want to log detailed info, such as the original and
            attacked text, attack results and debug info. If None, do not log anything.
        final_success_binary_callback (CallbackConfig): Config for the
            ScoringCallback to use for final evaluation. Should refer to a
            BinaryCallback, because we need discrete success/failure for each
            attacked input.
        compute_robustness_metric (bool): Whether to compute the robustness metric.
    """

    evaluation_attack: AttackConfig = MISSING
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
        experiment_type (str): Type of experiment, from the ones defined in __main__.py.
        experiment_name (str): Name of the overarching experiment. Used to set a "group"
            in wandb. Each experiment can have several jobs.
        job_type (str): Name of the sub-experiment.
        run_name (str): Name of the individual run.
        environment (EnvironmentConfig): Configs for environment setup.
        dataset (DatasetConfig): Configs for dataset setup.
        model (ModelConfig): Configs for model setup.
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
cs.store(name="DEFAULT", group="environment", node=EnvironmentConfig)
cs.store(name="DEFAULT", group="training", node=TrainingConfig)
cs.store(name="DEFAULT", group="evaluation", node=EvaluationConfig)
cs.store(name="DEFAULT", group="training/adversarial", node=AdversarialTrainingConfig)
