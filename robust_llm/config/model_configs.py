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
    inference_type: The type of inference the model is used for:
        ('classification', 'generation', or 'trl').
        The reason 'trl' is a separate category is that we need to load a
        model with a value head rather than a regular generation model.
    strict_load: Whether to enforce that no weights are ignored or randomly
        initialized while loading. Recommended to be True for evaluation (where the
        model should already be set up for the task) and False for training (where
        we may have to initialize a new classification head)
    """

    name_or_path: str = MISSING
    family: str = MISSING
    revision: str = "main"
    # This is variable interpolation, see:
    # https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html#interpolations
    inference_type: str = "${dataset.inference_type}"
    strict_load: bool = False
    padding_side: str = "right"

    def __post_init__(self):
        assert self.family in MODEL_FAMILIES
