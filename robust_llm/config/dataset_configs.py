from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Config used for dataset setup.

    Attributes:
        dataset_type: Type of dataset to use.
        n_train: Number of training examples.
        n_val: Number of validation examples.
        config_name: config_name from hf datasets (if applicable).
        revision: The huggingface revision to use. Use "<1.1.0" for
            compabitibility versions that match those used in the past, e.g.
            classification results in the workshop paper. Use "main" for the latest
            version.
        inference_type: The type of inference performed ("classification"
            or "generation")
        classification_as_generation: Whether we are doing classification
            using a generation model, in which case the 'gen_target' column
            represents the classification target.
        gen_target_override: A string to use as the gen_target
            everywhere, rather than the one given in the dataset.
        proxy_gen_target_override: A string to use as the
            proxy_gen_target everywhere, rather than the one given in the dataset.
            This is useful for generative tasks like targeting "I HATE YOU" in the
            output of the model.
        strip_leading_whitespace: Whether to strip leading whitespace
            from the gen_target.  This is necessary for some chat models where a
            leading space is undesirable because the chat template contains a
            newline. Defaults to False.
    """

    dataset_type: str = MISSING
    n_train: int = 0
    n_val: int = 0
    config_name: Optional[str] = None
    revision: str = MISSING
    inference_type: str = "classification"
    classification_as_generation: bool = True
    gen_target_override: Optional[str] = None
    proxy_gen_target_override: Optional[str] = None
    strip_leading_whitespace: bool = False


@dataclass
class ContactInfoDatasetConfig(DatasetConfig):
    """Config used for the ContactInfo dataset.

    Attributes:
        dataset_type: dataset type, always "ContactInfo".
        inference_type: The type of inference performed,
            always "generation"
        classification_as_generation: Whether we are doing classification
            using a generation model, which is never the case for this
            generative dataset.
        info_type:
            The type of information to generate, e.g. phone_number.
        revision: Unused for this dataset, set to a default value of "main".
    """

    dataset_type: str = "ContactInfo"
    inference_type: str = "generation"
    classification_as_generation: bool = False
    info_type: Optional[str] = None
    revision: str = "main"


cs = ConfigStore.instance()
cs.store(group="dataset", name="CONTACT_INFO", node=ContactInfoDatasetConfig)
