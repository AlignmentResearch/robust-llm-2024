"""Pipeline for adversarial training."""

from robust_llm.config.configs import ExperimentConfig
from robust_llm.training.state_classes import TrainingPipelineState
from robust_llm.training.train_loop import run_train_loop
from robust_llm.utils import print_time


@print_time()
def run_training_pipeline(args: ExperimentConfig, accelerator) -> TrainingPipelineState:
    return run_train_loop(args, accelerator)
