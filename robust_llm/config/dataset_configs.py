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
        revision (str): The huggingface revision to use. Defaults to '<1.1.0',
            which points to the compatibility versions of the old datasets. These
            have the exact same data as the old datasets but formatted to match the
            new expected columns.
        inference_type (str): The type of inference performed ("classification"
            or "generation")
        classification_as_generation (bool): Whether we are doing classification
            using a generation model, in which case the 'gen_target' column
            represents the classification target.
            TODO(ian): Find a way to avoid an explicit flag here.
    """

    dataset_type: str = MISSING
    n_train: int = 0
    n_val: int = 0
    config_name: Optional[str] = None
    revision: str = "<1.1.0"
    inference_type: str = "classification"
    classification_as_generation: bool = True
