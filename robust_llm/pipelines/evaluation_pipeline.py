"""Pipeline to evaluate a fixed model under attack."""

from accelerate import Accelerator

from robust_llm.configs import OverallConfig
from robust_llm.defenses import make_defended_model
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import wandb_cleanup, wandb_initialize
from robust_llm.pipelines.utils import (
    prepare_attack,
    prepare_datasets,
    prepare_language_generator,
    prepare_victim_models,
)
from robust_llm.utils import prepare_model_with_accelerate


def run_evaluation_pipeline(args: OverallConfig) -> None:
    use_cpu = args.experiment.environment.device == "cpu"
    accelerator = Accelerator(cpu=use_cpu)

    if accelerator.is_main_process:
        wandb_initialize(args.experiment)

    model, tokenizer, decoder = prepare_victim_models(args)
    model = prepare_model_with_accelerate(accelerator, model)

    language_generator = prepare_language_generator(args)
    robust_llm_datasets = prepare_datasets(
        args, tokenizer=tokenizer, language_generator=language_generator
    )

    attack = prepare_attack(
        args=args,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        robust_llm_datasets=robust_llm_datasets,
        training=False,
    )

    dataset = (
        robust_llm_datasets.tokenized_validation_dataset
        if attack.REQUIRES_INPUT_DATASET
        else None
    )
    # TODO(niki): this is a temporary measure
    # In a future PR we will transition to assuming there is always an
    # input dataset.
    assert dataset is not None

    if attack.REQUIRES_TRAINING:
        assert dataset is not None
        print("Training attack before evaluation")
        # We train the attack on the *validation* dataset here.
        # In the future, we might want to (also or only) train on a
        # different dataset, such as the train dataset.
        attack.train(dataset=dataset)

    if args.experiment.defense.defense_type is not None:
        defense_prep_dataset = robust_llm_datasets.train_dataset
        if args.experiment.defense.num_preparation_examples is not None:
            defense_prep_dataset = defense_prep_dataset.select(
                range(args.experiment.defense.num_preparation_examples)
            )

        model = make_defended_model(
            defense_config=args.experiment.defense,
            init_model=model,
            tokenizer=tokenizer,
            dataset=defense_prep_dataset,
            decoder=decoder,
        )

    do_adversarial_evaluation(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataset=dataset,
        ground_truth_label_fn=robust_llm_datasets.ground_truth_label_fn,
        num_generated_examples=args.experiment.evaluation.num_generated_examples,
        attack=attack,
        batch_size=args.experiment.evaluation.batch_size,
        num_examples_to_log_detailed_info=args.experiment.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
    )

    if accelerator.is_main_process:
        wandb_cleanup()
