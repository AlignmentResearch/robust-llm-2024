"""Pipeline to evaluate a fixed model under attack."""

import wandb

from robust_llm.configs import OverallConfig
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.pipelines.utils import (
    prepare_attack,
    prepare_datasets,
    prepare_language_generator,
    prepare_victim_models,
)
from robust_llm.utils import log_config_to_wandb


def run_evaluation_pipeline(args: OverallConfig) -> None:
    wandb.init(
        project="robust-llm",
        group=args.experiment.experiment_name,
        job_type=args.experiment.job_type,
        name=args.experiment.run_name,
    )

    log_config_to_wandb(args.experiment)

    model, tokenizer, _ = prepare_victim_models(args)

    language_generator = prepare_language_generator(args)

    robust_llm_datasets = prepare_datasets(
        args, tokenizer=tokenizer, language_generator=language_generator
    )

    attack = prepare_attack(
        args=args,
        model=model,
        tokenizer=tokenizer,
        robust_llm_datasets=robust_llm_datasets,
        training=False,
    )

    dataset = (
        robust_llm_datasets.tokenized_validation_dataset
        if attack.REQUIRES_INPUT_DATASET
        else None
    )

    do_adversarial_evaluation(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_generated_examples=args.experiment.evaluation.num_generated_examples,
        attack=attack,
        batch_size=args.experiment.evaluation.batch_size,
    )
