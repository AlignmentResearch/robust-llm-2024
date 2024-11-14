"""
Script that mocks uploading to huggingface  to see if the improved upload method works.

Raises stochastic errors to test the retry logic. Run manually to test 5 times.
"""

import functools
import os
import random

from accelerate import Accelerator
from tqdm import tqdm

from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.configs import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.logging_utils import LoggingContext
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.utils import interpolate_config

os.environ["NCCL_TIMEOUT"] = "60"

accelerator = Accelerator()
config = ExperimentConfig(
    experiment_type="evaluation",
    environment=EnvironmentConfig(
        test_mode=True,
        logging_level=10,
    ),
    evaluation=EvaluationConfig(
        evaluation_attack=RandomTokenAttackConfig(
            n_attack_tokens=1,
        ),
        num_iterations=2,
    ),
    model=ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        # We have to set this explicitly because we are not loading with Hydra,
        # so interpolation doesn't happen.
        inference_type="classification",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
    ),
    dataset=DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        revision="2.1.0",
        n_train=0,
        n_val=5,
    ),
)
logging_context = LoggingContext(
    args=config,
)
logging_context.setup()
config = interpolate_config(config)
victim = WrappedModel.from_config(config.model, accelerator=accelerator, num_classes=2)
assert isinstance(victim, WrappedModel)
repo_id = "AlignmentResearch/robust_llm_test_upload_model"
for _ in tqdm(range(5)):

    def mock_upload(original_func):

        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            if random.random() < 0.2:
                raise RuntimeError("Mock upload failed")
            return original_func(*args, **kwargs)

        return wrapper

    # Apply the mock
    victim.model._upload_modified_files = mock_upload(
        victim.model._upload_modified_files
    )

    victim.push_to_hub(
        repo_id=repo_id,
        revision="main",
        retries=5,
        cooldown_seconds=5,
    )
