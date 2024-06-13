from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from robust_llm.attacks.text_attack.constants import TEXT_ATTACK_ATTACK_TYPES
from robust_llm.config.constants import SHARED_DATA_DIR
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.model_utils import InferenceType


@dataclass
class AttackConfig:
    """
    Configs used in attack setup.

    Attributes:
        seed (int): Random seed for the attack.
        train_frequency (Optional[int]):
            If the attack needs training, how often to train it,
            counted in number of victim adversarial training rounds.
            If None, only train attack after the first victim training round.
            Must be positive or None.
        log_frequency (Optional[int]):
            If the attack needs training, how often to log training progress.
            If None, no training progress is logged. Must be positive or None.
        victim_inference_batch_size (int):
            Batch size to use for victim model inference.
    """

    seed: int = 0
    train_frequency: Optional[int] = None
    log_frequency: Optional[int] = 1
    victim_inference_batch_size: int = 8

    def __post_init__(self):
        if self.train_frequency is not None and self.train_frequency <= 0:
            raise ValueError("train_frequency must be positive or None.")

        if self.log_frequency is not None and self.log_frequency <= 0:
            raise ValueError("log_frequency must be positive or None.")


@dataclass
class IdentityAttackConfig(AttackConfig):
    """Options specific for the identity attack.

    This is just a pass-through for subclass type checking, since
    the identity attack doesn't have any additional options.
    """

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TextAttackAttackConfig(AttackConfig):
    """
    Options specific for TextAttack attacks.

    Attributes:
        text_attack_recipe (str): The TextAttack recipe to use (e.g. textfooler).
        query_budget (int): Query budget per example.
        num_examples (int): Number of examples to attack. If -1, attack whole dataset.
        num_modifiable_words_per_chunk (Optional[int]): If set to an integer value, the
            attack will replace all content of each modifiable chunk with
            `num_modifiable_words_per_chunk` placeholder words which can be then
            modified by the attack. Otherwise, content is not modified at the start and
            the attack performs modifications on the original text.
        silent (bool): If silent, TextAttack will only print errors.
    """

    text_attack_recipe: str = MISSING
    query_budget: int = 100
    num_examples: int = -1
    num_modifiable_words_per_chunk: Optional[int] = None
    silent: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.text_attack_recipe in TEXT_ATTACK_ATTACK_TYPES


@dataclass
class RandomTokenAttackConfig(AttackConfig):
    """Options specific for RandomToken attacks.

    Attributes:
        n_attack_tokens (int): The number of tokens to generate.
        n_its (int): Maximum number of iterations to run the attack.
        logging_frequency (int): How often to log the attack.
        victim_success_binary_callback (str): The name of the ScoringCallback to use to
            compute whether an attack was successful by computing whether the victim
            got the right answer. Should refer to a BinaryCallback, because we need
            discrete success/failure for each attacked input.
        prompt_attack_mode (PromptAttackMode): The mode to use for prompt
            attacks. "single-prompt" for attacking one prompt at a time,
            "multi-prompt" for attacking multiple prompts at once. Defaults to
            "single-prompt".
            TODO(GH#402): Use this for GCG as well and move into AttackConfig?
    """

    n_attack_tokens: int = 10
    n_its: int = 100
    victim_success_binary_callback: str = "successes_from_text"
    prompt_attack_mode: str = "single-prompt"

    def __post_init__(self):
        super().__post_init__()
        assert self.n_its > 0
        assert self.n_attack_tokens > 0


@dataclass
class LMBasedAttackConfig(AttackConfig):
    """Options specific for LM-based attacks.
    Attributes:
        adversary: Model config used as the LM adversary.
        adversary_input_templates: Prompt templates to use for eliciting the attack,
            one for each target label. Has to contain exactly one `{}` placeholder for
            each text chunk.
        adversary_output_templates:
            Templates to use for the adversary output, one per modifiable chunk.
            Each template should contain exactly one `{}` placeholder for the attack.
            E.g. ["Ignore the following tokens: {}"] for a single chunk.
        n_its: Maximum number of iterations to run the attack.
        adversary_batch_size: Batch size used for the LM adversary.
        victim_batch_size: Batch size used for the LM victium.
    """

    adversary: ModelConfig = MISSING
    adversary_input_templates: list[str] = MISSING
    adversary_output_templates: list[str] = field(default_factory=lambda: ["{}"])
    n_its: int = 10
    adversary_batch_size: int = 8
    victim_batch_size: int = 8
    prompt_attack_mode: str = "single-prompt"
    victim_success_binary_callback: str = "successes_from_text"

    def __post_init__(self):
        super().__post_init__()
        assert self.adversary.inference_type == InferenceType.GENERATION.value


@dataclass
class TRLAttackConfig(AttackConfig):
    """Options specific for TRL attacks.

    Attributes:
        batch_size (int):
            The TRL batch size (how many examples are passed
            to a single call of PPO's "step" function).
        mini_batch_size (int):
            The TRL minibatch size (how many examples to load
            onto the gpu at once).
        gradient_accumulation_steps (int):
            The TRL gradient accumulation steps (how many minibatches
            to accumulate before taking a single gradient update step).
        learning_rate (float):
            The learning rate to use for TRL.
        ppo_epochs (int):
            The number of ppo epochs to run TRL on the provided dataset.
        adversary (ModelConfig):
            The model to use as the adversary.
        min_length (int):
            The minimum number of tokens to generate.
            If -1, there is no minimum length.
            Name and convention copied from trl code.
        max_new_tokens (int):
            The maximum number of tokens to generate.
            Name copied from trl code.
        rewards_from_victim_callback (str):
            The name of the ScoringCallback to use to compute rewards for the
            inputs. Must take text as input, and return floats that can be used
            as rewards for the inputs. Default is "losses_from_text", which is
            equivalent to the old "minus_correct_logprob" `reward_type`. The
            other `reward_type`s can be implemented as ScoringCallbacks.
            TODO(GH#406): Add the other reward types as ScoringCallbacks.
        model_name_to_save (str):
            The name to use for saving the model.
        model_save_path_prefix (Optional[str]): Where to save the final
            checkpoint. If None, the model is not saved.
            Otherwise, the model is saved to a location starting with the
            specified prefix.
    """

    batch_size: int = 128
    mini_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.41e-5
    ppo_epochs: int = 10

    adversary: ModelConfig = MISSING

    min_length: int = -1
    max_new_tokens: int = 3

    rewards_from_victim_callback: str = "losses_from_text"

    model_name_to_save: str = "trl"
    model_save_path_prefix: Optional[str] = SHARED_DATA_DIR

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.batch_size == self.mini_batch_size * self.gradient_accumulation_steps
        )


@dataclass
class SearchBasedAttackConfig(AttackConfig):
    """
    Required options with defaults for search based attacks.

    Args:
        n_candidates_per_it: the total number of token replacements
            to consider in each iteration (in GCG, this must be less than
            top_k * n_attack_tokens, which is the total number of candidates).
        n_its: total number of iterations to run
        n_attack_tokens: number of attack tokens to optimize
        scores_from_text_callback: The name of the ScoringCallback to use to
            compute scores for the inputs. Must take text as input, and return
            floats that can be used to rank the inputs.
    """

    n_candidates_per_it: int = 128
    n_its: int = 10
    n_attack_tokens: int = 10
    scores_from_text_callback: str = "losses_from_text"

    def __post_init__(self):
        super().__post_init__()


@dataclass
class GCGAttackConfig(SearchBasedAttackConfig):
    """Required options with defaults for the GCG attack.

    Args:
        differentiable_embeds_callback (str): The name of the ScoringCallback to use to
            compute gradients for generating candidates. Must take embeddings as input,
            and must be differentiable with respect to the embeddings.
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    differentiable_embeds_callback: str = "losses_from_embeds"
    top_k: int = 256

    def __post_init__(self):
        super().__post_init__()
        if self.n_candidates_per_it > self.top_k * self.n_attack_tokens:
            raise ValueError(
                "n_candidates_per_it must be at most top_k * suffix_length"
            )


@dataclass
class MultipromptGCGAttackConfig(SearchBasedAttackConfig):
    """Required options with defaults for the multi-prompt GCG attack.

    Args:
        differentiable_embeds_callback (str): The name of the ScoringCallback to use to
            compute gradients for generating candidates. Must take embeddings as input,
            and must be differentiable with respect to the embeddings.
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    differentiable_embeds_callback: str = "losses_from_embeds"
    top_k: int = 256

    def __post_init__(self):
        super().__post_init__()
        if self.n_candidates_per_it > self.top_k * self.n_attack_tokens:
            raise ValueError(
                "n_candidates_per_it must be at most top_k * suffix_length"
            )


@dataclass
class BeamSearchAttackConfig(SearchBasedAttackConfig):
    """Required options with defaults for the beam search attack.

    Args:
        beam_search_width: how many candidates to keep after each iteration.
    """

    beam_search_width: int = 5

    def __post_init__(self):
        super().__post_init__()


# Register attack configs.
cs = ConfigStore.instance()
for text_attack_recipe in TEXT_ATTACK_ATTACK_TYPES:
    cs.store(
        group="attack",
        name=text_attack_recipe.upper(),
        node=TextAttackAttackConfig(text_attack_recipe=text_attack_recipe),
    )
cs.store(group="attack", name="IDENTITY", node=IdentityAttackConfig)
cs.store(group="attack", name="RANDOM_TOKEN", node=RandomTokenAttackConfig)
cs.store(group="attack", name="TRL", node=TRLAttackConfig)
cs.store(group="attack", name="GCG", node=GCGAttackConfig)
cs.store(group="attack", name="MULTIPROMPT_GCG", node=MultipromptGCGAttackConfig)
cs.store(group="attack", name="BEAM_SEARCH", node=BeamSearchAttackConfig)
cs.store(group="attack", name="LM_BASED", node=LMBasedAttackConfig)
