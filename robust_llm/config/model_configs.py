import math
from dataclasses import dataclass
from typing import Optional

import torch
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

    # RLLM-specific parameters
    trim_stop_strings: bool = False

    # Parameters that control the length of the output
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = 10
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: bool = False
    max_time: Optional[float] = None
    stop_strings: Optional[list[str]] = None

    # Parameters that control the generation strategy used
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    penalty_alpha: Optional[float] = None
    use_cache: bool = True

    # Parameters for manipulation of the model output logits
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    min_p: Optional[float] = None


@dataclass
class ModelConfig:
    """Config used for models.

    This includes the main (victim) model, as well as decoders for perplexity
    defenses and adversaries for TRL.
    name_or_path: Either HF name or path to model checkpoint.
    family: Which model family to load (e.g. "pythia"). MODEL_FAMILIES lists valid
        values.
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
    minibatch_multiplier: Multiplier for the minibatch size. This should usually
        be set by interpolation from the EnvironmentConfig rather in each model
        config directly.
    generation_config: The config to use for text generation.
    dtype: Data type, e.g., float32 or bfloat16.
        Defaults to float32 since bfloat16 has much less precision (e.g., the
        next bfloat16 after 1024 is 1032), which can affect generation quality.
    attention_implementation: The implementation with which to compute
        attention, with possible values listed at
        https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel.from_config.attn_implementation.
        In particular, "flash_attention_2" specifies FlashAttention-2.
        (Requires dtype float16 or bfloat16. Requires flash_attn package. Mainly
        provides speedup on long sequence lengths. Gradients on pythia-14m are
        known to misbehave:
        https://github.com/Dao-AILab/flash-attention/issues/1046)
    system_prompt: The prompt to pass as the "system prompt" for chat models.
        This is used to control the behavior of the model, e.g., to make it
        more conversational or more factual. If None, the default system prompt
        chosen by HuggingFace is used.
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
    minibatch_multiplier: float = SI("${environment.minibatch_multiplier}")
    generation_config: Optional[GenerationConfig] = None
    dtype: str = "float32"
    attention_implementation: Optional[str] = None
    system_prompt: str | None = None

    def __post_init__(self):
        assert self.family in MODEL_FAMILIES

        assert hasattr(torch, self.dtype), f"Invalid model dtype torch.{self.dtype}"
        assert isinstance(getattr(torch, self.dtype), torch.dtype)


cs = ConfigStore.instance()
cs.store(group="model/generation_config", name="DEFAULT", node=GenerationConfig)
