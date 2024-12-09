"""Main entry point.

Pipeline is chosen based on the experiment_type specified in the config.
"""

import os
import signal
import sys
from contextlib import contextmanager

import hydra
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig
from robust_llm.logging_utils import LoggingContext
from robust_llm.pipelines.evaluation_pipeline import run_evaluation_pipeline
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.pipelines.utils import safe_run_pipeline
from robust_llm.utils import maybe_make_deterministic

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)


EXPERIMENT_TYPE_TO_PIPELINE = {
    "training": run_training_pipeline,
    "evaluation": run_evaluation_pipeline,
}


@contextmanager
def handle_signals():
    """
    Context manager to handle interruption signals and ensure cleanup.

    Usage:
        with handle_signals():
            # Your main script here
    """

    def signal_handler(signum, frame):
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        if signal_name == "SIGTERM":
            logger.warning(
                "\n\nReceived SIGTERM."
                " This suggests the pod was either"
                " preempted, evicted, or its job was killed."
                " Shutting down...\n\n"
            )
        else:
            logger.warning(f"\n\nReceived {signal_name}. Shutting down...\n\n")

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Marking run as failed")
            wandb.summary["finish_reason"] = signal_name
            wandb.finish(exit_code=1)
        sys.exit(1)

    # Set up signal handlers
    original_term = signal.signal(signal.SIGTERM, signal_handler)
    original_int = signal.signal(signal.SIGINT, signal_handler)

    try:
        yield
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)
        logger.info("handle_signals cleanup completed")


@hydra.main(version_base=None, config_path="hydra_conf", config_name="base_config")
def main(args: DictConfig) -> None:
    # Get the experiment config
    cfg = OmegaConf.to_object(args)
    assert isinstance(cfg, ExperimentConfig)
    run(cfg)


def run(cfg: ExperimentConfig):
    if cfg.environment.device == "cpu":
        # We set the environment variable to force the accelerator to use CPU.
        # The reason we do this rather than setting Accelerator(cpu=True) is that
        # re-initializing the accelerator with arguments can cause issues, which
        # makes testing harder.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    accelerator = Accelerator()
    print(accelerator.device)

    maybe_make_deterministic(
        mode=cfg.environment.deterministic,
        cublas_config=cfg.environment.cublas_config,
    )

    set_up_step_metrics = cfg.experiment_type == "training"
    logging_context = LoggingContext(
        args=cfg,
        set_up_step_metrics=set_up_step_metrics,
    )
    logging_context.setup_logging()

    logger.info("Configuration arguments:\n")
    logger.info("%s\n", OmegaConf.to_yaml(cfg))

    # Run the relevant pipeline
    run_pipeline = EXPERIMENT_TYPE_TO_PIPELINE[cfg.experiment_type]
    with handle_signals():
        try:
            pipe_out = safe_run_pipeline(run_pipeline, cfg, accelerator)
        except Exception as e:
            logger.error("Pipeline failed with exception: %s", e, exc_info=True)
            if accelerator.is_main_process:
                wandb.summary["finish_reason"] = type(e).__name__
                wandb.finish(exit_code=1)
            raise e

    logging_context.cleanup()
    return pipe_out


if __name__ == "__main__":
    main()
