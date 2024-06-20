import glob
import os

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from robust_llm.config.configs import ExperimentConfig

BASE_PATH = "robust_llm/hydra_conf"
SUBDIRECTORIES = ("AdvTraining", "DefendedEval", "Eval", "Training")


def get_yaml_files(directory: str, subdirectories: tuple):
    """Get YAML files in the directory, excluding those starting with an underscore."""
    all_files = []
    for subdirectory in subdirectories:
        all_files += glob.glob(
            os.path.join(directory, "experiment", subdirectory, "*.yaml")
        )
    return [f for f in all_files if not os.path.basename(f).startswith("_")]


@pytest.mark.parametrize(
    "yaml_file",
    get_yaml_files(BASE_PATH, SUBDIRECTORIES),
)
def test_parse_experiment_config(yaml_file):
    """Test parsing YAML files into ExperimentConfig without raising an error."""
    try:
        with initialize(config_path="../robust_llm/hydra_conf"):
            # Extract the relative path to the yaml file within the base directory
            relative_path = os.path.relpath(yaml_file, BASE_PATH)

            # Compose the config from the relative path, removing the file extension
            cfg = compose(config_name=relative_path.replace(".yaml", ""))

            # Check that the configuration is valid
            assert isinstance(cfg, DictConfig), "Config is not a DictConfig"

            # Validate against the ExperimentConfig schema
            OmegaConf.merge(OmegaConf.structured(ExperimentConfig), cfg)

    except Exception as e:
        pytest.fail(f"Failed to parse {yaml_file}: {e}")
