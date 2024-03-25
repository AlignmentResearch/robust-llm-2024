"""Pipeline for adversarial training."""

from typing import Tuple

import wandb
from omegaconf import OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.configs import OverallConfig
from robust_llm.pipelines.utils import (
    prepare_datasets,
    prepare_language_generator,
    prepare_victim_models,
)
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import get_overlap, log_config_to_wandb, make_unique_name_to_save


def run_training_pipeline(
    args: OverallConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, PreTrainedModel | None]:
    experiment = args.experiment

    print("Configuration arguments:\n")
    print(OmegaConf.to_yaml(experiment))
    print()

    model, tokenizer, decoder = prepare_victim_models(args)

    # TODO(michal): Refactor this so that it is not created this early.
    # It is just a specific thing used in Tomita experiments.
    language_generator = prepare_language_generator(args)

    robust_llm_datasets = prepare_datasets(args, tokenizer, language_generator)

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
        "train_dataset": robust_llm_datasets.tokenized_train_dataset,
        "eval_dataset": {
            "validation": robust_llm_datasets.tokenized_validation_dataset
        },
        "model": model,
        "tokenizer": tokenizer,
        "model_name_to_save": model_name_to_save,
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
        "log_datasets_to_wandb": experiment.training.log_datasets_to_wandb,
        "ground_truth_label_fn": robust_llm_datasets.ground_truth_label_fn,
    }

    # Set up the training environment
    training: Training
    it = experiment.training.iterative
    if it.iterative_training:
        training = AdversarialTraining(
            **base_training_args,
            num_iterative_training_rounds=it.num_iterative_training_rounds,
            dataset_type=args.experiment.environment.dataset_type,
            language_generator=language_generator,
            training_attack_config=it.training_attack,
            validation_attack_config=experiment.evaluation.evaluation_attack,
            modifiable_chunks_spec=robust_llm_datasets.modifiable_chunks_spec,
            min_num_new_examples_to_add=it.min_num_new_examples_to_add,
            max_num_search_for_adversarial_examples=it.max_num_search_for_adversarial_examples,  # noqa: E501
            adversarial_example_search_minibatch_size=it.adversarial_example_search_minibatch_size,  # noqa: E501
            skip_first_training_round=it.skip_first_training_round,
            use_probabilistic_robustness_check=it.use_probabilistic_robustness_check,
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
        wandb.init(
            project="robust-llm",
            group=experiment.experiment_name,
            job_type=experiment.job_type,
            name=experiment.run_name,
        )

        log_config_to_wandb(args.experiment)

        # Log the train-val overlap to wandb
        assert wandb.run is not None
        if (
            experiment.environment.train_set_size is not None
            and experiment.environment.validation_set_size is not None
        ):
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

    training.maybe_save_model_to_path_or_hf(
        path_prefix_or_hf=experiment.training.model_save_path_prefix_or_hf
    )

    return model, tokenizer, decoder
