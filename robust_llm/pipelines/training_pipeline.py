"""Pipeline for adversarial training."""

from accelerate import Accelerator
from omegaconf import OmegaConf

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig
from robust_llm.logging_utils import LoggingContext
from robust_llm.training.train_loop import run_train_loop
from robust_llm.utils import maybe_make_deterministic, print_time


@print_time()
def run_training_pipeline(args: ExperimentConfig) -> None:
    accelerator = Accelerator(cpu=args.environment.device == "cpu")
    maybe_make_deterministic(
        args.environment.deterministic, args.environment.cublas_config
    )

    logging_context = LoggingContext(
        args=args,
        set_up_step_metrics=True,
    )
    logging_context.setup()

    assert args.training is not None
    logger.info("Configuration arguments:\n")
    logger.info("%s\n", OmegaConf.to_yaml(args))

    # set_seed(seed=args.training.seed, deterministic=args.environment.deterministic)
    run_train_loop(args, accelerator)

    logging_context.cleanup()
