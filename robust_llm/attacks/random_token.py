import random
from typing import Any, Sequence

from tqdm import tqdm
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.logging_utils import LoggingCounter
from robust_llm.models.caching_wrapped_model import get_caching_model_with_example
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import CallbackRegistry
from robust_llm.utils import get_randint_with_exclusions


class RandomTokenAttack(Attack):
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

        self.victim_success_binary_callback = CallbackRegistry.get_binary_callback(
            name=attack_config.victim_success_binary_callback,
        )

        self.logging_counter = LoggingCounter(_name="random_token_attack")

    @override
    def get_attacked_dataset(
        self, dataset: RLLMDataset
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        """Returns the attacked dataset and the attack metadata."""
        attacked_texts = []
        success_indices = []
        success_index = None
        for example in (pbar := tqdm(dataset.ds, mininterval=5, position=-1)):
            pbar.set_description(f"Previous success index: {success_index}")
            assert isinstance(example, dict)
            attacked_text, success_index = self.attack_example(example, dataset)
            attacked_texts.append(attacked_text)
            success_indices.append(success_index)

        attacked_dataset = dataset.with_attacked_text(attacked_texts)
        metadata = {"success_indices": success_indices}
        return attacked_dataset, metadata

    def attack_example(
        self, example: dict[str, Any], dataset: RLLMDataset
    ) -> tuple[str, int | None]:
        """Attacks a single example.

        Note that we are going per example and running all the iterations at
        once to take advantage of caching and batching. Instead of batching
        across examples, we batch across iterations because for RandomToken the
        iterations are independent of each other. However, this means we may run
        more iterations than necessary, e.g. if one of the early iterations
        would be successful.

        TODO(GH#400): Maybe check if the attack is successful after each
        minibatch? Currently seems like it would require refactoring
        ScoringCallbacks to be generators.

        Args:
            example: The example to be attacked.
            dataset: The RLLMDataset the example belongs to. We pass this so we
                have access to the ModifiableChunkSpec and the
                ground_truth_label_fn.


        Returns:
            The attacked text and the number of iterations it took
            to get a successful attack (if not successful, this will
            return the last attacked_text and None).
        """
        attacked_inputs = self.get_attacked_inputs(
            example=example,
            n_attacks=self.n_its,
            modifiable_chunk_spec=dataset.modifiable_chunk_spec,
        )
        original_labels = [example["clf_label"]] * self.n_its
        updated_labels = dataset.maybe_recompute_labels(
            texts=attacked_inputs,
            labels=original_labels,
        )

        with get_caching_model_with_example(self.victim, example["text"]) as victim:
            victim_successes = self.victim_success_binary_callback(
                victim,
                attacked_inputs,
                updated_labels,
            )

        attacked_text, attack_success_index = get_attacked_text_from_successes(
            attacked_inputs, victim_successes
        )
        return attacked_text, attack_success_index

    def get_attacked_inputs(
        self,
        example: dict[str, Any],
        n_attacks: int,
        modifiable_chunk_spec: ModifiableChunkSpec,
    ) -> list[str]:
        """Returns the attacked inputs for one example.

        Args:
            example: The example to be attacked.
            n_attacks: The number of attacked inputs to generate. (Typically
                this will be the number of iterations we intend to run.)
            modifiable_chunk_spec: The spec indicating which chunks are
                modifiable, and how.

        Returns:
            The attacked inputs for the example.
        """

        # TODO (ian): Speed this up by parallelizing if it's a bottleneck.
        attacked_inputs = []
        for _ in range(n_attacks):
            attacked_input = self._get_attacked_input(example, modifiable_chunk_spec)
            attacked_inputs.append(attacked_input)
        return attacked_inputs

    def _get_attacked_input(
        self, example: dict[str, Any], modifiable_chunk_spec: ModifiableChunkSpec
    ) -> str:
        """Returns one attacked input for one example.

        Args:
            example: The example to be attacked.
            modifiable_chunk_spec: The spec indicating which chunks are
                modifiable, and how.

        Returns:
            One attacked input for the example.
        """
        # We forbid introducing special tokens in the attack tokens.
        excluded_token_ids = self.victim.tokenizer.all_special_ids

        current_input = ""
        for chunk_index, chunk_type in enumerate(modifiable_chunk_spec):
            chunk_text = example["chunked_text"][chunk_index]
            current_input += self._get_text_for_chunk(
                chunk_text=chunk_text,
                chunk_type=chunk_type,
                excluded_token_ids=excluded_token_ids,
                n_tokens=self.n_attack_tokens,
            )
        return current_input

    def _get_text_for_chunk(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        excluded_token_ids: list[int],
        n_tokens: int,
    ) -> str:
        """Returns the text for a chunk based on its type.

        Args:
            chunk: The chunk text.
            chunk_type: The chunk type.
            excluded_token_ids: The token ids to exclude from the random selection (e.g.
                special tokens).
            n_tokens: The number of random tokens to append to the chunk.

        Returns:
            The text for the chunk based on its type.
        """
        match chunk_type:
            case ChunkType.IMMUTABLE:
                return chunk_text

            case ChunkType.PERTURBABLE:
                token_ids = self._n_random_token_ids_with_exclusions(
                    n_tokens, excluded_token_ids
                )
                random_tokens = self.victim.tokenizer.decode(token_ids)
                return chunk_text + random_tokens

            case ChunkType.OVERWRITABLE:
                token_ids = self._n_random_token_ids_with_exclusions(
                    n_tokens, excluded_token_ids
                )
                random_tokens = self.victim.tokenizer.decode(token_ids)
                return random_tokens

            case _:
                raise ValueError(f"Unknown chunk type: {chunk_type}")

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


def get_attacked_text_from_successes(
    attacked_inputs: Sequence[str], victim_successes: Sequence[bool]
) -> tuple[str, int | None]:
    """Returns the attacked text and the index of the successful attack.

    Args:
        attacked_inputs: The attacked inputs.
        victim_successes: The successes of the model on the inputs.

    Returns:
        The attacked text and the index of the successful attack.
        Returns the last attacked text and None if no attack was successful.
    """
    try:
        # We look for False in the successes list, which indicates a successful
        # attack (i.e. that the model got the answer wrong after the attack).
        attack_success_index = victim_successes.index(False)
        return attacked_inputs[attack_success_index], attack_success_index
    except ValueError:
        return attacked_inputs[-1], None
