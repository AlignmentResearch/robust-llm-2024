"""Pipeline to evaluate a fixed model under attack."""

import wandb
from accelerate import Accelerator

from robust_llm.configs import OverallConfig
from robust_llm.defenses import make_defended_model
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import wandb_cleanup, wandb_initialize
from robust_llm.pipelines.utils import prepare_attack, prepare_victim_models
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.utils import prepare_model_with_accelerate


def run_evaluation_pipeline(args: OverallConfig) -> None:
    use_cpu = args.experiment.environment.device == "cpu"
    accelerator = Accelerator(cpu=use_cpu)

    if accelerator.is_main_process:
        wandb_initialize(args.experiment)

    validation = load_rllm_dataset(args.experiment.dataset, split="validation")
    num_classes = validation.num_classes

    model, tokenizer, decoder = prepare_victim_models(args, num_classes)

    assert wandb.run is not None
    # Log the model size to wandb for use in plots, so we don't
    # have to try to get it out of the model name.
    # We use `commit=False` to avoid incrementing the step counter.
    # TODO (GH#348): Move this to a more appropriate place.
    wandb.log({"model_size": model.num_parameters()}, commit=False)

    model = prepare_model_with_accelerate(accelerator, model)
    model.eval()
    if decoder is not None:
        decoder.eval()

    attack = prepare_attack(
        args=args,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        training=False,
    )

    if attack.REQUIRES_TRAINING:
        assert validation is not None
        print("Training attack before evaluation")
        # We train the attack on the *validation* dataset here.
        # In the future, we might want to (also or only) train on a
        # different dataset, such as the train dataset.
        attack.train(dataset=validation)

    if args.experiment.defense is not None:
        # TODO (GH#322): Propagate RLLMDataset into defenses.
        # For now, we just make the minimal change to make these compatible
        train = load_rllm_dataset(args.experiment.dataset, split="train")
        # NOTE: 'train.ds' has a column called 'clf_label' rather than 'label',
        # but the current defense code does not actually use the label column so this
        # is fine for now. This will also be fixed in GH#322.
        defense_prep_dataset = train.ds

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
        dataset=validation,
        attack=attack,
        batch_size=args.experiment.evaluation.batch_size,
        num_examples_to_log_detailed_info=args.experiment.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
    )

    if accelerator.is_main_process:
        wandb_cleanup()
