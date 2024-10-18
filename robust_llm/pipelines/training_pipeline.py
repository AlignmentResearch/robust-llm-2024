"""Pipeline for adversarial training."""

from typing import Any

import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import set_seed

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig
from robust_llm.logging_utils import LoggingContext
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import deterministic_hash, maybe_make_deterministic


def run_training_pipeline(args: ExperimentConfig) -> None:
    maybe_make_deterministic(
        args.environment.deterministic, args.environment.cublas_config
    )

    logging_context = LoggingContext(
        args=args,
        set_up_step_metrics=True,
    )
    logging_context.setup()

    assert args.training is not None
    assert (args.training.save_strategy == "no") or not dist.is_initialized(), (
        "Saving checkpoints does not work with distributed training. "
        "See: https://github.com/AlignmentResearch/robust-llm/issues/908"
    )
    logger.info("Configuration arguments:\n")
    logger.info("%s\n", OmegaConf.to_yaml(args))

    # TODO (ian): load the tokenizer first, then the datasets, then the model
    untokenized_train_set = load_rllm_dataset(args.dataset, split="train")
    untokenized_val_set = load_rllm_dataset(args.dataset, split="validation")

    num_classes = untokenized_train_set.num_classes
    set_seed(seed=args.training.seed, deterministic=args.environment.deterministic)
    victim = WrappedModel.from_config(
        args.model, accelerator=None, num_classes=num_classes
    )
    logging_context.maybe_log_model_info(
        model_size=victim.n_params,
        model_family=victim.family,
    )

    # We tokenize the datasets using the right-padding tokenizer
    # because training doesn't do autoregressive generation.
    train_set = untokenized_train_set.tokenize(victim.right_tokenizer)
    val_set = untokenized_val_set.tokenize(victim.right_tokenizer)

    # NOTE: the "validation" dataset is one of what will be
    # several datasets that we perform model evaluation on,
    # hence "eval_dataset" is a dict[str, Dataset], not a Dataset.
    base_training_args: dict[str, Any] = {
        "config": args.training,
        "train_rllm_dataset": train_set,
        "eval_rllm_dataset": {
            "validation": val_set,
        },
        "victim": victim,
        "model_name": args.model.name_or_path,
        "environment_config": args.environment,
        "evaluation_config": args.evaluation,
        "run_name": args.run_name,
        "hash": deterministic_hash(args),
        "local_files_path": logging_context.local_files_path,
    }

    # Set up the training environment
    training: Training
    if args.training.adversarial is not None:
        assert (
            args.evaluation is not None
        ), "Must provide EvaluationConfig for adversarial training"
        training = AdversarialTraining(
            **base_training_args,
            validation_attack_config=args.evaluation.evaluation_attack,
            validation_iterations=args.evaluation.num_iterations,
        )
    else:
        training = Training(**base_training_args)

    trainer = training.setup_trainer()

    logger.debug(f"Training arguments: {trainer.args.to_dict()}")

    # Perform the training
    training.run_trainer()

    logging_context.cleanup()
