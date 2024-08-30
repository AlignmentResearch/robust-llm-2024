"""Common building blocks for pipelines."""

from typing import Callable

from accelerate import find_executable_batch_size

from robust_llm import logger
from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.config.configs import ExperimentConfig
from robust_llm.models import WrappedModel


def prepare_attack(
    args: ExperimentConfig,
    victim: WrappedModel,
    training: bool,
) -> Attack:
    logger.info("Preparing attack...")

    if training:
        assert args.training is not None
        assert args.training.adversarial is not None
        logging_name = "training_attack"
        attack_config = args.training.adversarial.training_attack
    else:
        assert args.evaluation is not None
        logging_name = "eval_attack"
        attack_config = args.evaluation.evaluation_attack

    return create_attack(
        attack_config=attack_config,
        logging_name=logging_name,
        victim=victim,
        run_name=args.run_name,
    )


def safe_run_pipeline(pipeline: Callable, args: ExperimentConfig) -> None:
    starting_batch_size = args.model.train_minibatch_size

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def run_pipeline_with_batch_size(batch_size: int) -> None:
        logger.info(f"Calling {pipeline.__name__} with batch size {batch_size}")
        args.model.train_minibatch_size = batch_size
        pipeline(args)

    run_pipeline_with_batch_size()  # type: ignore[reportCallIssue]
