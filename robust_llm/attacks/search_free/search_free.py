import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets import Dataset
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from robust_llm.attacks.attack import Attack, AttackState, PromptAttackMode
from robust_llm.config.attack_configs import AttackConfig
from robust_llm.models.caching_wrapped_model import get_caching_model_with_example
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import (
    BinaryCallback,
    BinaryCallbackOutput,
    CallbackInput,
    CallbackOutput,
    TensorCallback,
)
from robust_llm.scoring_callbacks.scoring_callback_utils import TensorCallbackOutput

SUCCESS_TYPE = Sequence[bool] | Tensor
SUCCESSES_TYPE = list[SUCCESS_TYPE]


@dataclass
class SearchFreeAttackState(AttackState):
    victim_successes: SUCCESSES_TYPE = field(default_factory=list)
    success_indices: list[list[int]] = field(default_factory=list)


class SearchFreeAttack(Attack, ABC):
    """Attack where each iteration is independent, no searching.

    The attack is repeated for each datapoint until it is successful,
    or until `attack_config.n_its` is reached.
    Appends the attack to the modifiable text instead
    of replacing it if the chunk is PERTURBABLE.
    """

    CAN_CHECKPOINT = True
    REQUIRES_TRAINING = False
    n_its: int
    victim: WrappedModel
    prompt_attack_mode: PromptAttackMode
    victim_success_callback: BinaryCallback | TensorCallback

    def __init__(
        self,
        attack_config: AttackConfig,
        victim: WrappedModel,
        run_name: str,
        logging_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            attack_config, victim=victim, run_name=run_name, logging_name=logging_name
        )
        self.attack_state = SearchFreeAttackState()
        self.rng = random.Random(attack_config.seed)

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        resume_from_checkpoint: bool = True,
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        """Returns the attacked dataset and the attack metadata."""
        if resume_from_checkpoint and self.maybe_load_state():
            assert isinstance(self.attack_state, SearchFreeAttackState)
            attacked_texts = self.attack_state.attacked_texts
            all_successes = self.attack_state.victim_successes
            all_success_indices = self.attack_state.success_indices
            starting_index = self.attack_state.example_index + 1
        else:
            # Reset the state if not resuming from checkpoint
            attacked_texts = []
            all_successes = []
            all_success_indices = []
            self.attack_state = SearchFreeAttackState()
            starting_index = 0

        for example_index in tqdm(
            range(starting_index, len(dataset.ds)), mininterval=5, position=-1
        ):
            example = dataset.ds[example_index]
            assert isinstance(example, dict)
            attacked_text, victim_successes = self.attack_example(example, dataset)
            # We look for False in the successes list, which indicates a successful
            # attack (i.e. that the model got the answer wrong after the attack).
            success_indices = (
                []
                if isinstance(victim_successes, Tensor)
                else [i for i, s in enumerate(victim_successes) if not s]
            )
            attacked_texts.append(attacked_text)
            all_success_indices.append(success_indices)
            all_successes.append(victim_successes)

            # Save the state after each example in case of interruption.
            self.attack_state = SearchFreeAttackState(
                example_index=example_index,
                attacked_texts=attacked_texts,
                success_indices=all_success_indices,
                victim_successes=all_successes,
                example_info=self.attack_state.example_info,
            )
            if resume_from_checkpoint:
                self.maybe_save_state()

        if self.prompt_attack_mode == PromptAttackMode.MULTIPROMPT:
            attacked_texts = self._get_multi_prompt_attacked_texts(
                dataset=dataset,
                success_indices=all_success_indices,
            )

        attacked_dataset = dataset.with_attacked_text(attacked_texts)
        metadata = {"success_indices": all_success_indices}
        return attacked_dataset, metadata

    def use_callback_results(
        self, callback_input: CallbackInput, callback_output: CallbackOutput
    ) -> None:
        """Can be overriden to update some state based on the callback results."""

    def attack_example(
        self, example: dict[str, Any], dataset: RLLMDataset
    ) -> tuple[str, SUCCESS_TYPE]:
        """Attacks a single example.

        Note that we are going per example and running all the iterations at
        once to take advantage of caching and batching. Instead of batching
        across examples, we batch across iterations because for search-free attacks the
        iterations are independent of each other. However, this means we may run
        more iterations than necessary, e.g. if one of the early iterations
        would be successful.

        TODO(GH#400): Maybe check if the attack is successful after each
        minibatch? Currently seems like it would require refactoring
        ScoringCallbacks to be generators.

        Args:
            example: The example to be attacked.
            dataset: The RLLMDataset the example belongs to. We pass this so we
                have access to the ModifiableChunkSpec and methods for updating
                columns based on changes to the text.


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

        # Constructing a temporary dataset to get the attacked_text and labels
        temp_attack_ds = Dataset.from_list(
            [
                {
                    "text": attacked_input,
                    "clf_label": example["clf_label"],
                    "gen_target": example["gen_target"],
                }
                for attacked_input in attacked_inputs
            ]
        )
        temp_attack_ds = dataset.update_dataset_based_on_text(temp_attack_ds)

        callback_input = CallbackInput(
            # TODO(ian): Work out where to apply the chat template.
            input_data=self.victim.maybe_apply_chat_template(temp_attack_ds["text"]),
            clf_label_data=temp_attack_ds["clf_label"],
            gen_target_data=temp_attack_ds["gen_target"],
            original_input_data=[example["text"] for _ in attacked_inputs],
        )

        with get_caching_model_with_example(self.victim, example["text"]) as victim:
            victim_out = self.victim_success_callback(
                victim,
                callback_input,
            )
        self.use_callback_results(callback_input, victim_out)
        victim_successes: SUCCESS_TYPE
        if isinstance(victim_out, BinaryCallbackOutput):
            victim_successes = victim_out.successes
        else:
            assert isinstance(victim_out, TensorCallbackOutput)
            victim_successes = victim_out.losses

        attacked_text = get_attacked_text_from_successes(
            attacked_inputs, victim_successes
        )
        # TODO(ian): Work out if 'apply_chat_template' messes with the updating
        # done in 'with_attacked_text'.
        return victim.maybe_apply_chat_template(attacked_text), victim_successes

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
        for i in range(n_attacks):
            attacked_input = self._get_attacked_input(
                example, modifiable_chunk_spec, current_iteration=i
            )
            attacked_inputs.append(attacked_input)
        return attacked_inputs

    def _get_attacked_input(
        self,
        example: dict[str, Any],
        modifiable_chunk_spec: ModifiableChunkSpec,
        current_iteration: int,
    ) -> str:
        """Returns one attacked input for one example.

        Args:
            example: The example to be attacked.
            modifiable_chunk_spec: The spec indicating which chunks are
                modifiable, and how.
            current_iteration: The current iteration of the attack.

        Returns:
            One attacked input for the example.
        """
        if modifiable_chunk_spec.n_modifiable_chunks > 1:
            raise ValueError("Only one modifiable chunk per example supported for now.")

        current_input = ""
        chunk_count = 0
        for chunk_index, chunk_type in enumerate(modifiable_chunk_spec):
            chunk_text = example["chunked_text"][chunk_index]
            current_input += self._get_text_for_chunk(
                chunk_text=chunk_text,
                chunk_type=chunk_type,
                current_iteration=current_iteration,
                chunk_label=example["clf_label"],
                chunk_seed=example.get("seed"),
                chunk_index=chunk_count,
            )
            chunk_count += int(chunk_type != ChunkType.IMMUTABLE)
        return current_input

    @abstractmethod
    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: Optional[int],
    ) -> list[int]:
        """Returns the attack tokens for the current iteration.

        Args:
            chunk_text: The text of the chunk to be attacked.
            chunk_type: The type of the chunk.
            current_iteration: The current iteration of the attack (only used in
            the multi-prompt case).
            chunk_label: The label of the chunk (for classification).
            chunk_seed: The seed for the chunk (for generation).

        Subclasses return:
            The attack tokens for the current iteration.
        """
        raise NotImplementedError

    def post_process_attack_string(self, attack_tokens: str, chunk_index: int) -> str:
        """Post-processes the attack tokens into a string.
        Useful for overloading in subclasses.

        Args:
            attack_tokens: The decoded attack tokens.
            chunk_index: The index of the chunk in the example (useful in subclasses).

        Returns:
            The attack tokens after post-processing.
        """
        return attack_tokens

    def _get_text_for_chunk(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: Optional[int],
        chunk_index: int,
    ) -> str:
        """Returns the text for a chunk based on its type.

        If we are in single-prompt mode, we generate attack tokens here; if we
        are in multi-prompt mode, we return the cached attack tokens for the
        current iteration.

        Args:
            chunk_text: The chunk text.
            chunk_type: The chunk type.
            current_iteration: The current iteration of the attack.
            chunk_label: The label of the chunk (for classification).
            chunk_seed: The seed for the chunk (for generation).
            chunk_index: The index of the chunk in the example.

        Returns:
            The text for the chunk based on its type.
        """
        match chunk_type:
            case ChunkType.IMMUTABLE:
                return chunk_text

            case ChunkType.PERTURBABLE:
                token_ids = self._get_attack_tokens(
                    chunk_text,
                    chunk_type,
                    current_iteration,
                    chunk_label,
                    chunk_seed,
                )
                attack_tokens = self.victim.decode(token_ids)
                attack_tokens = self.post_process_attack_string(
                    attack_tokens, chunk_index
                )
                return chunk_text + attack_tokens

            case ChunkType.OVERWRITABLE:
                token_ids = self._get_attack_tokens(
                    chunk_text,
                    chunk_type,
                    current_iteration,
                    chunk_label,
                    chunk_seed,
                )
                attack_tokens = self.victim.decode(token_ids)
                attack_tokens = self.post_process_attack_string(
                    attack_tokens, chunk_index
                )
                return attack_tokens

            case _:
                raise ValueError(f"Unknown chunk type: {chunk_type}")

    def _get_multi_prompt_attacked_texts(
        self, dataset: RLLMDataset, success_indices: list[list[int]]
    ) -> list[str]:
        """Get attacked_texts for multi-prompt setting.

        If we are in multi-prompt attack mode, instead of using the first
        successful attack, we use the attack that had the most successes across
        all examples.
        """
        # The best attack iteration is the iteration that was most successful
        # across all prompts.
        best_attack_iteration = _get_most_frequent_index(success_indices)
        # If there were no successful attacks, we use the last attack arbitrarily.
        if best_attack_iteration is None:
            best_attack_iteration = self.n_its - 1

        def get_best_attacked_input(example) -> str:
            """Get the attacked input for the best attack iteration.

            NOTE: We don't type-hint example because it doesn't play nicely with
            the list comprehension and we'd just have to ignore it anyway.
            """
            return self._get_attacked_input(
                example=example,
                modifiable_chunk_spec=dataset.modifiable_chunk_spec,
                current_iteration=best_attack_iteration,
            )

        attacked_texts = [get_best_attacked_input(example) for example in dataset.ds]

        return attacked_texts


def _get_most_frequent_index(indices: list[list[int]]) -> int | None:
    """Returns the most frequent index from a list of lists of indices.

    Returns None if the lists of indices are all empty.
    """
    counts: dict[int, int] = defaultdict(int)
    for current_indices in indices:
        for index in current_indices:
            counts[index] += 1
    if len(counts) == 0:
        return None
    most_frequent_index = max(counts, key=lambda index: counts[index])
    return most_frequent_index


def get_attacked_text_from_successes(
    attacked_inputs: Sequence[str],
    victim_successes: SUCCESS_TYPE,
) -> str:
    """Returns the attacked text and the indices of successful attacks.

    Args:
        attacked_inputs: The attacked inputs.
        victim_successes: The successes of the model on the inputs. This can be
            binary success/fail or a floating point score.

    Returns:
        The first successful attacked text and the indices of successful attacks.
        Returns the last attacked text and empty list if no attack was successful.
    """
    if all([s is True for s in victim_successes]):
        # If there were no successful attacks, return the last attacked input.
        return attacked_inputs[-1]
    # The following one-liner is used to grab the first successful attack if
    # using binary success, or the *most* successful attack if using floats.
    # We take the min rather than the max because these successes are from
    # the perspective of the victim, so lower is better.
    _, attack_index = min((val, idx) for (idx, val) in enumerate(victim_successes))
    attacked_text = attacked_inputs[attack_index]
    return attacked_text
