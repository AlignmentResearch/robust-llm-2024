"""Pipeline to evaluate a fixed model under attack."""

from accelerate import Accelerator

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig
from robust_llm.defenses import make_defended_model
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import LoggingContext
from robust_llm.models import WrappedModel
from robust_llm.pipelines.utils import prepare_attack
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.scoring_callbacks import build_binary_scoring_callback


def run_evaluation_pipeline(args: ExperimentConfig) -> dict[str, float]:
    assert args.evaluation is not None
    use_cpu = args.environment.device == "cpu"

    accelerator = Accelerator(cpu=use_cpu)

    validation = load_rllm_dataset(args.dataset, split="validation")
    num_classes = validation.num_classes

    victim = WrappedModel.from_config(args.model, accelerator, num_classes)
    victim.eval()

    logging_context = LoggingContext(
        is_main_process=accelerator.is_main_process,
        args=args,
        model_size=victim.n_params,
        model_family=victim.family,
    )
    logging_context.setup()

    attack = prepare_attack(
        args=args,
        victim=victim,
        training=False,
    )

    if attack.REQUIRES_TRAINING:
        assert validation is not None
        logger.info("Training attack before evaluation")
        # We train the attack on the *validation* dataset here.
        # In the future, we might want to (also or only) train on a
        # different dataset, such as the train dataset.
        attack.train(dataset=validation)
        global_step_count = attack.logging_counter.root.step_count
        global_datapoint_count = attack.logging_counter.root.datapoint_count
    else:
        global_step_count = 0
        global_datapoint_count = 0

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
    final_callback_config = args.evaluation.final_success_binary_callback
    final_callback = build_binary_scoring_callback(final_callback_config)

    results = do_adversarial_evaluation(
        victim=victim,
        dataset=validation,
        attack=attack,
        final_success_binary_callback=final_callback,
        num_examples_to_log_detailed_info=args.evaluation.num_examples_to_log_detailed_info,  # noqa: E501
        # In evaluation pipeline, we do not have adversarial training, so these are 0
        adv_training_round=0,
        victim_training_step_count=0,
        victim_training_datapoint_count=0,
        # Set the global count using the attack's logging counter
        global_step_count=global_step_count,
        global_datapoint_count=global_datapoint_count,
    )

    logging_context.cleanup()

    return results
