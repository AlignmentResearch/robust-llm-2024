from dataclasses import dataclass

from omegaconf import MISSING

from robust_llm.config.constants import MODEL_FAMILIES


@dataclass
class ModelConfig:
    """Config used for models.

    This includes the main (victim) model, as well as decoders for perplexity
    defenses and adversaries for TRL.
    name_or_path: Either HF name or path to model checkpoint.
    family (ModelFamily): Which model family to load (e.g. "pythia").
    revision: The huggingface revision of the model.
    """

    name_or_path: str = MISSING
    family: str = MISSING
    revision: str = "main"

    def __post_init__(self):
        assert self.family in MODEL_FAMILIES
