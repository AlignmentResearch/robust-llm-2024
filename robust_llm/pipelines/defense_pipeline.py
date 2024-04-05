"""Pipeline to test defenses to adversarial attack.

NOTE: This pipeline has several issues (e.g. creates multiple wandb runs).
Use with caution. It will be refactored/integrated with other pipelines in the future.

This pipeline does the following:
* Fine-tunes the model for the task.
* Performs baseline adversarial evaluation.
* Defends model.
* Performs adversarial evaluation with defense.
"""

import wandb
from accelerate import Accelerator

from robust_llm.configs import OverallConfig
from robust_llm.defenses import make_defended_model
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.pipelines.utils import (
    prepare_attack,
    prepare_datasets,
    prepare_language_generator,
)


def run_defense_pipeline(args: OverallConfig):
    use_cpu = args.experiment.environment.device == "cpu"
    accelerator = Accelerator(cpu=use_cpu)
    model, tokenizer, decoder = run_training_pipeline(args)
    model, decoder = accelerator.prepare(model, decoder)
    language_generator = prepare_language_generator(args)
    robust_llm_datasets = prepare_datasets(
        args, tokenizer=tokenizer, language_generator=language_generator
    )
    print("Performing baseline adversarial evaluation")
    wandb.init(
        project="robust-llm",
        group=args.experiment.experiment_name,
        job_type="pre_" + args.experiment.job_type,
        name=args.experiment.run_name,
    )
    attack = prepare_attack(
        args=args,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        robust_llm_datasets=robust_llm_datasets,
        training=False,
    )
    do_adversarial_evaluation(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataset=robust_llm_datasets.validation_dataset,
        num_generated_examples=args.experiment.evaluation.num_generated_examples,
        attack=attack,
        batch_size=args.experiment.evaluation.batch_size,
        ground_truth_label_fn=robust_llm_datasets.ground_truth_label_fn,
        num_examples_to_log_detailed_info=args.experiment.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
    )
    wandb.finish()
    print("Performing adversarial evaluation with defense")
    defended_model = make_defended_model(
        defense_config=args.experiment.defense,
        init_model=model,
        tokenizer=tokenizer,
        dataset=robust_llm_datasets.train_dataset,
        decoder=decoder,
    )
    wandb.init(
        project="robust-llm",
        group=args.experiment.experiment_name,
        job_type="post_" + args.experiment.job_type,
        name=args.experiment.run_name,
    )
    new_attack = prepare_attack(
        args=args,
        model=defended_model,
        tokenizer=defended_model.tokenizer,
        accelerator=accelerator,
        robust_llm_datasets=robust_llm_datasets,
        training=False,
    )
    do_adversarial_evaluation(
        model=defended_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataset=robust_llm_datasets.validation_dataset,
        num_generated_examples=args.experiment.evaluation.num_generated_examples,
        attack=new_attack,
        batch_size=args.experiment.evaluation.batch_size,
        ground_truth_label_fn=robust_llm_datasets.ground_truth_label_fn,
        num_examples_to_log_detailed_info=args.experiment.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
    )
    wandb.finish()
    print("Finished defense pipeline")
