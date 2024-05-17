"""Pipeline for adversarial training."""

from typing import Any

from omegaconf import OmegaConf

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig
from robust_llm.logging_utils import LoggingContext
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import make_unique_name_to_save


def run_training_pipeline(args: ExperimentConfig) -> None:
    assert args.training is not None
    logger.info("Configuration arguments:\n")
    logger.info("%s\n", OmegaConf.to_yaml(args))

    # TODO (ian): load the tokenizer first, then the datasets, then the model
    untokenized_train_set = load_rllm_dataset(args.dataset, split="train")
    untokenized_val_set = load_rllm_dataset(args.dataset, split="validation")

    num_classes = untokenized_train_set.num_classes
    victim = WrappedModel.from_config(
        args.model, accelerator=None, num_classes=num_classes
    )

    train_set = untokenized_train_set.tokenize(victim.tokenizer)
    val_set = untokenized_val_set.tokenize(victim.tokenizer)

    model_name_to_save = args.training.force_name_to_save or make_unique_name_to_save(
        args.model.name_or_path
    )
    # NOTE: the "validation" dataset is one of what will be
    # several datasets that we perform model evaluation on,
    # hence "eval_dataset" is a dict[str, Dataset], not a Dataset.
    base_training_args: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "job_type": args.job_type,
        "train_rllm_dataset": train_set,
        "eval_rllm_dataset": {
            "validation": val_set,
        },
        "victim": victim,
        "model_name_to_save": model_name_to_save,
        "model_save_path_prefix_or_hf": args.training.model_save_path_prefix_or_hf,
        "environment_config": args.environment,
        "evaluation_config": args.evaluation,
        "train_epochs": args.training.num_train_epochs,
        "learning_rate": args.training.learning_rate,
        "train_batch_size": args.training.batch_size,
        # TODO (ian): Choose a *training* eval batch size somewhere (previously
        # it was using the attack eval batch size).
        "eval_batch_size": args.training.batch_size,
        "optimizer": args.training.optimizer,
        "gradient_checkpointing": args.training.gradient_checkpointing,
        "eval_steps": args.training.eval_steps,
        "logging_steps": args.training.logging_steps,
        "save_strategy": args.training.save_strategy,
        "save_steps": args.training.save_steps,
        "seed": args.training.seed,
        "log_full_datasets_to_wandb": args.training.log_full_datasets_to_wandb,
    }

    # Set up the training environment
    training: Training
    if args.training.adversarial is not None:
        adv = args.training.adversarial
        assert (
            args.evaluation is not None
        ), "Must provide EvaluationConfig for adversarial training"
        training = AdversarialTraining(
            **base_training_args,
            num_adversarial_training_rounds=adv.num_adversarial_training_rounds,
            training_attack_config=adv.training_attack,
            validation_attack_config=args.evaluation.evaluation_attack,
            num_examples_to_generate_each_round=adv.num_examples_to_generate_each_round,
            num_examples_to_log_to_wandb_each_round=adv.num_examples_to_log_to_wandb_each_round,  # noqa: E501
            skip_first_training_round=adv.skip_first_training_round,
            use_balanced_sampling=adv.use_balanced_sampling,
            only_add_successful_adversarial_examples=adv.only_add_successful_adversarial_examples,  # noqa: E501
        )
    else:
        training = Training(**base_training_args)

    trainer = training.setup_trainer()

    logging_context = LoggingContext(
        is_main_process=trainer.is_world_process_zero(),
        args=args,
        set_up_step_metrics=True,
        num_parameters=victim.model.num_parameters(),
    )

    logging_context.setup()

    # Perform the training
    training.run_trainer()

    logging_context.cleanup()
