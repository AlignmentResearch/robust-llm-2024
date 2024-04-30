from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Config used for dataset setup.

    Attributes:
        dataset_type (str): Type of dataset to use.
        n_train (int): Number of training examples.
        n_val (int): Number of validation examples.
        config_name (Optional[str]): config_name from hf datasets (if applicable).
        revision (str): The huggingface revision to use. Defaults to <1.0.0 to
            avoid unexpected breaking changes.
        inference_type (str): The type of inference performed ("classification"
            or "generation")
    """

    dataset_type: str = MISSING
    n_train: int = 0
    n_val: int = 0
    config_name: Optional[str] = None
    revision: str = "<1.0.0"
    inference_type: str = "classification"
