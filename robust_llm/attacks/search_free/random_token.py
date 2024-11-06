from typing import Any

from robust_llm.attacks.attack import PromptAttackMode
from robust_llm.attacks.search_free.search_free import SearchFreeAttack
from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.configs import ExperimentConfig
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.scoring_callbacks import build_binary_scoring_callback
from robust_llm.utils import get_randint_with_exclusions


class RandomTokenAttack(SearchFreeAttack):
    """Random token attack.

    Replaces all the OVERWRITABLE text with random tokens
    from the victim tokenizer's vocabulary. The attack
    is repeated for each datapoint until it is successful,
    or until `attack_config.n_its` is reached.
    Appends the attack to the modifiable text instead
    of replacing it if the chunk is PERTURBABLE.
    """

    def __init__(
        self,
        exp_config: ExperimentConfig,
        victim: WrappedModel,
        is_training: bool,
    ) -> None:
        """Constructor for RandomTokenAttack.

        Args:
            exp_config: ExperimentConfig object containing the configuration for the
                attack.
            victim: The model to be attacked.
            is_training: Whether the attack is being used for training or evaluation.
        """
        super().__init__(exp_config, victim=victim, is_training=is_training)
        assert isinstance(self.attack_config, RandomTokenAttackConfig)

        self.n_attack_tokens = self.attack_config.n_attack_tokens

        self.prompt_attack_mode = PromptAttackMode(
            self.attack_config.prompt_attack_mode
        )
        cb_config = self.attack_config.victim_success_callback
        self.victim_success_callback = build_binary_scoring_callback(cb_config)
        self._shared_attack_tokens: list[list[int]] = []

    def get_shared_attack_tokens(self, iteration: int) -> list[int]:
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
        cached_its = len(self._shared_attack_tokens)
        if iteration < cached_its:
            return self._shared_attack_tokens[iteration]

        # We forbid introducing special tokens in the attack tokens.
        excluded_token_ids = self.victim.all_special_ids

        for _ in range(cached_its, iteration + 1):
            attack_tokens = self._n_random_token_ids_with_exclusions(
                self.n_attack_tokens, excluded_token_ids
            )
            self._shared_attack_tokens.append(attack_tokens)
        return self._shared_attack_tokens[iteration]

    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_proxy_label: int,
        chunk_seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
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
            chunk_proxy_label: The proxy label to use as the target in the attack.
            chunk_seed: The seed for the chunk to be attacked (for generation).

        Returns:
            The attack tokens for the current iteration.
        """
        assert isinstance(chunk_text, str)
        assert isinstance(chunk_type, ChunkType)
        assert isinstance(chunk_proxy_label, int)
        assert isinstance(chunk_seed, int)
        match self.prompt_attack_mode:
            case PromptAttackMode.SINGLEPROMPT:
                return (
                    self._n_random_token_ids_with_exclusions(
                        n=self.n_attack_tokens,
                        excluded_token_ids=self.victim.all_special_ids,
                    ),
                    {},
                )
            case PromptAttackMode.MULTIPROMPT:
                return self.get_shared_attack_tokens(current_iteration), {}

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
