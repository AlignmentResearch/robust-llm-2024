"""Main entry point.

Pipeline is chosen based on the experiment_type specified in the config."""

import hydra
from hydra.core.config_store import ConfigStore

from robust_llm.configs import OverallConfig
from robust_llm.pipelines.evaluation_pipeline import run_evaluation_pipeline
from robust_llm.pipelines.scaling_experiments import run_scaling_experiments_pipeline
from robust_llm.pipelines.training_pipeline import run_training_pipeline

cs = ConfigStore.instance()
cs.store(name="base_config", node=OverallConfig)


EXPERIMENT_TYPE_TO_PIPELINE = {
    "training": run_training_pipeline,
    "evaluation": run_evaluation_pipeline,
    "scaling_experiments": run_scaling_experiments_pipeline,
}


@hydra.main(version_base=None, config_path="hydra_conf", config_name="default_config")
def main(args: OverallConfig) -> None:
    run_pipeline = EXPERIMENT_TYPE_TO_PIPELINE[args.experiment.experiment_type]
    run_pipeline(args)


if __name__ == "__main__":
    main()
