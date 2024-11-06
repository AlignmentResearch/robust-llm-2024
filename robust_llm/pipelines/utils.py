"""Common building blocks for pipelines."""

from typing import Callable

from accelerate import Accelerator, find_executable_batch_size

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig


def safe_run_pipeline(
    pipeline: Callable, args: ExperimentConfig, accelerator: Accelerator
):
    starting_batch_size = max(
        1, int(args.model.max_minibatch_size * args.model.env_minibatch_multiplier)
    )

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def run_pipeline_with_batch_size(batch_size: int, accelerator):
        logger.info(f"Calling {pipeline.__name__} with batch size {batch_size}")
        args.model.max_minibatch_size = batch_size
        args.model.env_minibatch_multiplier = 1.0
        return pipeline(args, accelerator)

    return run_pipeline_with_batch_size(accelerator)  # type: ignore[reportCallIssue]
