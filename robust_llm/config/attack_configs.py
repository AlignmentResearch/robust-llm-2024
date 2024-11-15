from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.file_utils import get_save_root
from robust_llm.models.model_utils import InferenceType


@dataclass
class AttackConfig:
    """
    Configs used in attack setup.

    Attributes:
        seed: Random seed for the attack.
        train_frequency (Optional[int]):
            If the attack needs training, how often to train it,
            counted in number of victim adversarial training rounds.
            If None, only train attack after the first victim training round.
            Must be positive or None.
        log_frequency (Optional[int]):
            If the attack needs training, how often to log training progress.
            If None, no training progress is logged. Must be positive or None.
        victim_inference_batch_size:
            Batch size to use for victim model inference.
        save_prefix:
            Prefix to use for saving attack states.
        save_steps:
            How often to save attack states.
        save_total_limit:
            Maximum number of attack states to keep at the same time.
        perturb_position_min:
            The earliest position in a perturbable chunk in which to insert
            the attack tokens, expressed as a fraction of the perturbable chunk length.
            [0, 1]-valued float.
        perturb_position_max:
            The latest position in a perturbable chunk in which to insert
            the attack tokens, expressed as a fraction of the perturbable chunk length.
            [0, 1]-valued float.
    """

    seed: int = 0
    train_frequency: Optional[int] = None
    log_frequency: Optional[int] = 1
    victim_inference_batch_size: int = 8
    save_prefix: str = get_save_root()
    save_steps: int = 100
    save_total_limit: int = 1
    perturb_position_min: float = 1.0
    perturb_position_max: float = 1.0

    def __post_init__(self):
        if self.train_frequency is not None and self.train_frequency <= 0:
            raise ValueError("train_frequency must be positive or None.")

        if self.log_frequency is not None and self.log_frequency <= 0:
            raise ValueError("log_frequency must be positive or None.")

        assert 0 <= self.perturb_position_min <= self.perturb_position_max <= 1.0


@dataclass
class IdentityAttackConfig(AttackConfig):
    """Options specific for the identity attack.

    This is just a pass-through for subclass type checking, since
    the identity attack doesn't have any additional options.
    """

    def __post_init__(self):
        super().__post_init__()


@dataclass
class SearchFreeAttackConfig(AttackConfig):
    """Options specific for search-free attacks.

    Attributes:
        victim_success_callback: Config for the
            ScoringCallback to use to compute whether an attack was successful by
            computing whether the victim got the right answer. Should refer to a
            BinaryCallback, because we need discrete success/failure for each
            attacked input.
        prompt_attack_mode: The mode to use for prompt
            attacks. "single-prompt" for attacking one prompt at a time,
            "multi-prompt" for attacking multiple prompts at once. Defaults to
            "single-prompt".
            TODO(GH#402): Use this for GCG as well and move into AttackConfig?

    """

    victim_success_callback: CallbackConfig = field(
        default_factory=lambda: CallbackConfig(
            callback_name="successes_from_text", callback_return_type="binary"
        )
    )
    prompt_attack_mode: str = "single-prompt"

    def __post_init__(self):
        super().__post_init__()


@dataclass
class RandomTokenAttackConfig(SearchFreeAttackConfig):
    """Options specific for RandomToken attacks.

    Attributes:
        n_attack_tokens: The number of tokens to generate.
    """

    n_attack_tokens: int = 10

    def __post_init__(self):
        super().__post_init__()
        assert self.n_attack_tokens > 0


@dataclass
class LMAttackConfig(SearchFreeAttackConfig):
    """Options specific for LM-based attacks.

    Attributes:
        adversary: Model config used as the LM adversary.
        adversary_input_templates: Prompt templates to use for eliciting the attack,
            one for each target label. Each string may contain a `{}` placeholder for
            each modifiable text chunk in the dataset to include in the input to the
            adversary.
            E.g. ["{} Make the victim do something bad!"] for a single label.
        adversary_output_templates:
            Templates to use for the adversary output, one per modifiable chunk.
            Each template should contain exactly one `{}` placeholder for the attack.
            E.g. ["Ignore the following tokens: {}"] for a single chunk.
        adversary_prefix: Prefix to place in the assistant response as context
            when generating from the adversary.
            E.g. "{'prompt': '"
        attack_start_strings: Strings to use to delimit the start the attack.
            If multiple strings are provided, the attack will start after all of the
            strings.
            If none are provided or found, then the attack will start at the beginning
            of the input.
            e.g. "'prompt': '" for JSON attacks
        attack_end_strings: Strings to use to delimit the end of the attack.
            If multiple strings are provided, the attack will end before any of the
            strings.
            If none are provided or found, then the attack will end at the end of the
            input.
            e.g. "'}" for JSON attacks
        use_raw_adversary_input: If True, we will skip trying to insert the original
            data and the chat template. This should only be set when called from a
            few-shot attack.
        victim_success_callback: Config for the
            ScoringCallback to use to compute whether an attack was successful by
            computing whether the victim got the right answer. Should refer to a
            BinaryCallback, because we need discrete success/failure for each
            attacked input.
    """

    adversary: ModelConfig = MISSING
    adversary_input_templates: list[str] = MISSING
    adversary_output_templates: list[str] = field(default_factory=lambda: ["{}"])
    adversary_prefix: str = ""
    attack_start_strings: list[str] = field(default_factory=list)
    attack_end_strings: list[str] = field(default_factory=list)
    use_raw_adversary_input: bool = False
    victim_success_callback: CallbackConfig = field(
        default_factory=lambda: CallbackConfig(
            callback_name="successes_from_text", callback_return_type="binary"
        )
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.adversary.inference_type == InferenceType.GENERATION.value


@dataclass
class FewShotLMAttackConfig(LMAttackConfig):
    """Options specific for Stochastic Few Shot LM red-team attacks.

    Attributes:
        n_turns: The number of turns to run the chat with the adversary.
        few_shot_score_template: The template to use for reporting the attack
            results from previous turns. Must contain {response} and {success}
            placeholders.
        initial_adversary_prefix: Prefix to use for the first turn of the attack.
            Useful for PAIR where there is no 'improvement' in turn 0.
            If None, defaults to be the same as the regular `adversary_prefix`.
    """

    n_turns: int = 3
    few_shot_score_template: str = "Response: {response}\nScore: {score}\n"
    initial_adversary_prefix: str | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.n_turns > 0
        assert "{response}" in self.few_shot_score_template
        assert (
            "{score" in self.few_shot_score_template
            and "}" in self.few_shot_score_template
        )


@dataclass
class SearchBasedAttackConfig(AttackConfig):
    """
    Required options with defaults for search based attacks.

    Args:
        n_candidates_per_it: the total number of token replacements
            to consider in each iteration (in GCG, this must be less than
            top_k * n_attack_tokens, which is the total number of candidates).
        n_attack_tokens: number of attack tokens to optimize
        scores_from_text_callback: The config of the ScoringCallback to use to
            compute scores for the inputs. Must take text as input, and return
            floats that can be used to rank the inputs.
    """

    n_candidates_per_it: int = 128
    n_attack_tokens: int = 10
    scores_from_text_callback: CallbackConfig = field(
        default_factory=lambda: CallbackConfig(
            callback_name="losses_from_text", callback_return_type="tensor"
        )
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class GCGAttackConfig(SearchBasedAttackConfig):
    """Required options with defaults for the GCG attack.

    Args:
        differentiable_embeds_callback: The config of the
            ScoringCallback to use to compute gradients for generating candidates.
            Must take embeddings as input, and must be differentiable with respect
            to the embeddings.
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    differentiable_embeds_callback: CallbackConfig = field(
        default_factory=lambda: CallbackConfig(
            callback_name="losses_from_embeds", callback_return_type="tensor"
        )
    )
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
cs.store(group="attack", name="IDENTITY", node=IdentityAttackConfig)
cs.store(
    group="attack",
    name="RANDOM_TOKEN",
    node=RandomTokenAttackConfig,
)
cs.store(group="attack", name="GCG", node=GCGAttackConfig)
cs.store(group="attack", name="BEAM_SEARCH", node=BeamSearchAttackConfig)
cs.store(group="attack", name="ZERO_SHOT_LM", node=LMAttackConfig)
cs.store(group="attack", name="FEW_SHOT_LM", node=FewShotLMAttackConfig)
