import random
from functools import cached_property
from typing import Optional

from robust_llm.attacks.attack import PromptAttackMode
from robust_llm.attacks.search_free.search_free import SearchFreeAttack
from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.logging_utils import LoggingCounter
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.scoring_callbacks import CallbackRegistry
from robust_llm.utils import get_randint_with_exclusions


class RandomTokenAttack(SearchFreeAttack):
    """Random token attack.

    Replaces all the OVERWRITABLE text with random tokens
    from the victim tokenizer's vocabulary. The attack
    is repeated for each datapoint until it is successful,
    or until `attack_config.max_iterations` is reached.
    Appends the attack to the modifiable text instead
    of replacing it if the chunk is PERTURBABLE.
    """

    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: RandomTokenAttackConfig,
        victim: WrappedModel,
    ) -> None:
        """Constructor for RandomTokenAttack.

        Args:
            attack_config: Config of the attack.
            victim: The WrappedModel to be attacked.
        """
        super().__init__(attack_config)

        self.victim = victim

        self.rng = random.Random(self.attack_config.seed)

        self.n_attack_tokens = attack_config.n_attack_tokens
        self.n_its = attack_config.n_its

        self.prompt_attack_mode = PromptAttackMode(attack_config.prompt_attack_mode)
        self.victim_success_binary_callback = CallbackRegistry.get_binary_callback(
            name=attack_config.victim_success_binary_callback,
        )

        self.logging_counter = LoggingCounter(_name="random_token_attack")

    @cached_property
    def shared_attack_tokens(self) -> list[list[int]]:
        """Get the shared attack tokens for this instance of RandomTokenAttack.

        This is for multi-prompt attacks. We generate the attack tokens ahead of
        time and cache them so we can reuse them for each example. This lets us
        ensure that the attack tokens in each iteration are consistent across
        examples, so we can take advantage of caching by running all attacks on
        a single example at a time and then finding the most successful
        iteration across examples afterwards.

        NOTE: This implementation assumes that there is only a single modifiable
        chunk in the dataset.
        """
        # We forbid introducing special tokens in the attack tokens.
        excluded_token_ids = self.victim.all_special_ids

        all_attack_tokens = []
        for _ in range(self.n_its):
            attack_tokens = self._n_random_token_ids_with_exclusions(
                self.n_attack_tokens, excluded_token_ids
            )
            all_attack_tokens.append(attack_tokens)
        return all_attack_tokens

    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: Optional[int] = None,
    ) -> list[int]:
        """Returns the random attack tokens for the current iteration.

        If we are in single-prompt mode, we generate random tokens here
        to reduce variance between experiment runs (since for low n_its
        we could get unlucky and get weak attack tokens).

        If we are in multi-prompt mode, we return the cached attack tokens
        for the current iteration.

        Args:
            chunk_text: The text of the chunk to be attacked (not used).
            chunk_type: The type of the chunk to be attacked (not used).
            current_iteration: The current iteration of the attack (only used in
            the multi-prompt case).
            chunk_label: The label of the chunk to be attacked (for classification).
            chunk_seed: The seed for the chunk to be attacked (for generation).

        Returns:
            The attack tokens for the current iteration.
        """
        assert isinstance(chunk_text, str)
        assert isinstance(chunk_type, ChunkType)
        assert isinstance(chunk_label, int)
        assert chunk_seed is None
        match self.prompt_attack_mode:
            case PromptAttackMode.SINGLEPROMPT:
                return self._n_random_token_ids_with_exclusions(
                    n=self.n_attack_tokens,
                    excluded_token_ids=self.victim.all_special_ids,
                )
            case PromptAttackMode.MULTIPROMPT:
                return self.shared_attack_tokens[current_iteration]

    def _n_random_token_ids_with_exclusions(
        self, n: int, excluded_token_ids: list[int]
    ) -> list[int]:
        """Returns n random token ids with exclusions.

        Args:
            n: The number of random token ids to return.
            excluded_token_ids: The token ids to exclude from the random selection (e.g.
                special tokens).

        Returns:
            n random token ids with exclusions.
        """
        return [
            get_randint_with_exclusions(
                high=self.victim.vocab_size,
                exclusions=excluded_token_ids,
                rng=self.rng,
            )
            for _ in range(n)
        ]
