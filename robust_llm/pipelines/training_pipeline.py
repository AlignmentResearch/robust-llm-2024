"""Pipeline for adversarial training."""

from typing import Tuple

import wandb
from omegaconf import OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.configs import OverallConfig
from robust_llm.logging_utils import wandb_cleanup, wandb_initialize
from robust_llm.pipelines.utils import prepare_victim_models
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import get_unique_overlap, make_unique_name_to_save


def run_training_pipeline(
    args: OverallConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, PreTrainedModel | None]:
    experiment = args.experiment

    print("Configuration arguments:\n")
    print(OmegaConf.to_yaml(experiment))
    print()

    # TODO (ian): load the tokenizer first, then the datasets, then the model
    untokenized_train_set = load_rllm_dataset(experiment.dataset, split="train")
    untokenized_val_set = load_rllm_dataset(experiment.dataset, split="validation")

    num_classes = untokenized_train_set.num_classes
    model, tokenizer, decoder = prepare_victim_models(args, num_classes=num_classes)

    train_set = untokenized_train_set.tokenize(tokenizer)
    val_set = untokenized_val_set.tokenize(tokenizer)

    model_name_to_save = (
        args.experiment.training.force_name_to_save
        or make_unique_name_to_save(experiment.environment.model_name_or_path)
    )
    # NOTE: the "validation" dataset is one of what will be
    # several datasets that we perform model evaluation on,
    # hence "eval_dataset" is a dict[str, Dataset], not a Dataset.
    base_training_args = {
        "experiment_name": experiment.experiment_name,
        "run_name": experiment.run_name,
        "job_type": experiment.job_type,
        "train_rllm_dataset": train_set,
        "eval_rllm_dataset": {
            "validation": val_set,
        },
        "model": model,
        "tokenizer": tokenizer,
        "model_name_to_save": model_name_to_save,
        "environment_config": experiment.environment,
        "evaluation_config": experiment.evaluation,
        "train_epochs": experiment.training.num_train_epochs,
        "learning_rate": experiment.training.learning_rate,
        "train_batch_size": experiment.training.batch_size,
        "eval_batch_size": experiment.evaluation.batch_size,
        "optimizer": experiment.training.optimizer,
        "gradient_checkpointing": experiment.training.gradient_checkpointing,
        "eval_steps": experiment.training.eval_steps,
        "logging_steps": experiment.training.logging_steps,
        "save_strategy": experiment.training.save_strategy,
        "save_steps": experiment.training.save_steps,
        "seed": experiment.training.seed,
        "log_full_datasets_to_wandb": experiment.training.log_full_datasets_to_wandb,
    }

    # Set up the training environment
    training: Training
    it = experiment.training.iterative
    if it.iterative_training:
        training = AdversarialTraining(
            **base_training_args,
            num_iterative_training_rounds=it.num_iterative_training_rounds,
            training_attack_config=it.training_attack,
            validation_attack_config=experiment.evaluation.evaluation_attack,
            num_examples_to_generate_each_round=it.num_examples_to_generate_each_round,
            num_examples_to_log_to_wandb_each_round=it.num_examples_to_log_to_wandb_each_round,  # noqa: E501
            skip_first_training_round=it.skip_first_training_round,
            use_balanced_sampling=it.use_balanced_sampling,
            only_add_successful_adversarial_examples=it.only_add_successful_adversarial_examples,  # noqa: E501
        )
    else:
        training = Training(**base_training_args)

    trainer = training.setup_trainer()

    if trainer.is_world_process_zero():
        # Unlike in the evaluation pipeline, we don't set up our wandb step metrics here
        # (which logged values can be used as x-axes, and what the default x-axes
        # are set to be) because HuggingFace sets up its own metrics when we initialize
        # the Trainer, and we wait until that is done to overwrite them with our own.
        # We do this in the `CustomLoggingWandbCallback`'s `setup` method.
        wandb_initialize(experiment, set_up_step_metrics=False)

        # Log the train-val overlap to wandb
        assert wandb.run is not None
        if (
            experiment.dataset.n_train is not None
            and experiment.dataset.n_val is not None
        ):
            train_val_overlap = get_unique_overlap(
                smaller_dataset=val_set.ds,
                larger_dataset=train_set.ds,
            )
            wandb.run.summary["train_val_overlap_size"] = len(train_val_overlap)  # type: ignore  # noqa: E501
            wandb.run.summary["train_val_overlap_over_train_set_size"] = len(  # type: ignore  # noqa: E501
                train_val_overlap
            ) / len(
                train_set.ds["text"]
            )
            wandb.run.summary["train_val_overlap_over_val_set_size"] = len(  # type: ignore  # noqa: E501
                train_val_overlap
            ) / len(
                val_set.ds["text"]
            )

    # Perform the training
    training.run_trainer()

    training.maybe_save_model_to_path_or_hf(
        path_prefix_or_hf=experiment.training.model_save_path_prefix_or_hf
    )

    if trainer.is_world_process_zero():
        wandb_cleanup()

    return model, tokenizer, decoder
