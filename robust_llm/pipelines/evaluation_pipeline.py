"""Pipeline to evaluate a fixed model under attack."""

import wandb
from accelerate import Accelerator

from robust_llm.config.configs import ExperimentConfig
from robust_llm.defenses import make_defended_model
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import wandb_cleanup, wandb_initialize
from robust_llm.models import WrappedModel
from robust_llm.pipelines.utils import prepare_attack
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset


def run_evaluation_pipeline(args: ExperimentConfig) -> None:
    assert args.evaluation is not None
    use_cpu = args.environment.device == "cpu"
    accelerator = Accelerator(cpu=use_cpu)

    if accelerator.is_main_process:
        wandb_initialize(args)

    validation = load_rllm_dataset(args.dataset, split="validation")
    num_classes = validation.num_classes

    victim = WrappedModel.from_config(args.model, accelerator, num_classes)

    assert wandb.run is not None
    # Log the model size to wandb for use in plots, so we don't
    # have to try to get it out of the model name.
    # We use `commit=False` to avoid incrementing the step counter.
    # TODO (GH#348): Move this to a more appropriate place.
    wandb.log({"model_size": victim.model.num_parameters()}, commit=False)

    victim.eval()

    attack = prepare_attack(
        args=args,
        victim=victim,
        training=False,
    )

    if attack.REQUIRES_TRAINING:
        assert validation is not None
        print("Training attack before evaluation")
        # We train the attack on the *validation* dataset here.
        # In the future, we might want to (also or only) train on a
        # different dataset, such as the train dataset.
        attack.train(dataset=validation)

    if args.defense is not None:
        # TODO (GH#322): Propagate RLLMDataset into defenses.
        # For now, we just make the minimal change to make these compatible
        train = load_rllm_dataset(args.dataset, split="train")
        # NOTE: 'train.ds' has a column called 'clf_label' rather than 'label',
        # but the current defense code does not actually use the label column so this
        # is fine for now. This will also be fixed in GH#322.
        defense_prep_dataset = train.ds

        if args.defense.num_preparation_examples is not None:
            defense_prep_dataset = defense_prep_dataset.select(
                range(args.defense.num_preparation_examples)
            )

        victim = make_defended_model(
            victim=victim,
            defense_config=args.defense,
            dataset=defense_prep_dataset,
        )

    do_adversarial_evaluation(
        victim=victim,
        dataset=validation,
        attack=attack,
        batch_size=args.evaluation.batch_size,
        num_examples_to_log_detailed_info=args.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
    )

    if accelerator.is_main_process:
        wandb_cleanup()
