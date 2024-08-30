"""Main entry point.

Pipeline is chosen based on the experiment_type specified in the config.
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from robust_llm.config.configs import ExperimentConfig
from robust_llm.pipelines.evaluation_pipeline import run_evaluation_pipeline
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.pipelines.utils import safe_run_pipeline

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)


EXPERIMENT_TYPE_TO_PIPELINE = {
    "training": run_training_pipeline,
    "evaluation": run_evaluation_pipeline,
}


@hydra.main(version_base=None, config_path="hydra_conf", config_name="base_config")
def main(args: DictConfig) -> None:
    # Get the experiment config
    cfg = OmegaConf.to_object(args)
    assert isinstance(cfg, ExperimentConfig)

    # Run the relevant pipeline
    run_pipeline = EXPERIMENT_TYPE_TO_PIPELINE[cfg.experiment_type]
    safe_run_pipeline(run_pipeline, cfg)


if __name__ == "__main__":
    main()
