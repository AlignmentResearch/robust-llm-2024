import math
from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI, OmegaConf

from robust_llm.config.constants import MODEL_FAMILIES

# This is a custom resolver that allows us to e.g. set batch sizes
# as multiples of each other.
# The omegaconf docs have a page about using 'eval' for this purpose;
# this achieves the same thing but is much safer:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#id1
OmegaConf.register_new_resolver("mult", lambda *args: int(math.prod(args)))


@dataclass
class GenerationConfig:
    """LM text generation settings.
    This dataclass mirrors arguments of the
    HF transformers.generation.configuration_utils.GenerationConfig. See its
    description for details about the arguments.
    """

    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = 10
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: bool = False
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0


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
    train_minibatch_size: The minibatch size to use for training. Defaults to
        small value of 16.
    eval_minibatch_size: The minibatch size to use for evaluation.
        Defaults to twice the training minibatch size (since for evaluation we
        don't need to store gradients).
    generation_config: The config to use for text generation.
    """

    name_or_path: str = MISSING
    family: str = MISSING
    revision: str = "main"
    # This is variable interpolation, see:
    # https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html#interpolations
    inference_type: str = "${dataset.inference_type}"
    strict_load: bool = False
    train_minibatch_size: int = 16
    # This is variable interpolation plus a custom resolver (see above).
    eval_minibatch_size: int = SI("${mult: 2, ${model.train_minibatch_size}}")
    generation_config: Optional[GenerationConfig] = None

    def __post_init__(self):
        assert self.family in MODEL_FAMILIES


cs = ConfigStore.instance()
cs.store(group="model/generation_config", name="DEFAULT", node=GenerationConfig)
