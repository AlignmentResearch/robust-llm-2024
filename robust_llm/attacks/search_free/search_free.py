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

from robust_llm.attacks.attack import (
    Attack,
    AttackData,
    AttackOutput,
    AttackState,
    PromptAttackMode,
)
from robust_llm.config.attack_configs import AttackConfig, SearchFreeAttackConfig
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
    TensorCallback,
)
from robust_llm.scoring_callbacks.build_scoring_callback import build_scoring_callback
from robust_llm.scoring_callbacks.scoring_callback_utils import TensorCallbackOutput

SUCCESS_TYPE = Sequence[bool] | Tensor
SUCCESSES_TYPE = list[SUCCESS_TYPE]


@dataclass
class SearchFreeAttackState(AttackState):
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
        assert isinstance(attack_config, SearchFreeAttackConfig)
        self.rng = random.Random(attack_config.seed)
        self.prompt_attack_mode = PromptAttackMode(attack_config.prompt_attack_mode)
        cb_config = attack_config.victim_success_callback
        self.victim_success_callback = build_scoring_callback(cb_config)

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int,
        resume_from_checkpoint: bool = True,
    ) -> AttackOutput:
        """Returns the attacked dataset and the attack metadata."""
        if resume_from_checkpoint and self.maybe_load_state():
            assert isinstance(self.attack_state, SearchFreeAttackState)
            attacked_texts = self.attack_state.attacked_texts
            all_success_indices = self.attack_state.success_indices
            per_example_info = self.attack_state.attacks_info
            starting_index = self.attack_state.example_index + 1
        else:
            # Reset the state if not resuming from checkpoint
            attacked_texts = []
            all_success_indices = []
            per_example_info = {}
            self.attack_state = SearchFreeAttackState()
            starting_index = 0

        for example_index in tqdm(
            range(starting_index, len(dataset.ds)), mininterval=5, position=-1
        ):
            example = dataset.ds[example_index]
            assert isinstance(example, dict)
            example["seed"] = example_index
            # TODO (Oskar): account for examples in other partitions when indexing
            attacked_text, example_info, victim_successes = self.attack_example(
                example, dataset, n_its
            )
            attacked_text = self.victim.maybe_apply_user_template(attacked_text)
            # We look for False in the successes list, which indicates a successful
            # attack (i.e. that the model got the answer wrong after the attack).
            success_indices = (
                []
                if isinstance(victim_successes, Tensor)
                else [i for i, s in enumerate(victim_successes) if not s]
            )

            # Append data for this example
            attacked_texts.append(attacked_text)
            per_example_info = {
                k: per_example_info.get(k, []) + [v] for k, v in example_info.items()
            }
            all_success_indices.append(success_indices)

            # Save the state after each example in case of interruption.
            self.attack_state = SearchFreeAttackState(
                example_index=example_index,
                attacked_texts=attacked_texts,
                success_indices=all_success_indices,
                attacks_info=per_example_info,
            )
            if resume_from_checkpoint:
                self.maybe_save_state()

        if self.prompt_attack_mode == PromptAttackMode.MULTIPROMPT:
            attacked_texts = self._get_multi_prompt_attacked_texts(
                dataset=dataset,
                n_its=n_its,
                success_indices=all_success_indices,
            )

        attacked_dataset = dataset.with_attacked_text(attacked_texts)
        attack_out = AttackOutput(
            dataset=attacked_dataset,
            attack_data=AttackData(),
            per_example_info=per_example_info,
        )
        return attack_out

    def attack_example(
        self, example: dict[str, Any], dataset: RLLMDataset, n_its: int
    ) -> tuple[str, dict[str, Any], SUCCESS_TYPE]:
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
            n_its: The number of iterations to run.


        Returns:
            The attacked text, a dict of extra attack info for the chosen attack
            iteration, and the full list of successes.
        """
        attacked_inputs, attacked_info = self.get_attacked_inputs(
            example=example,
            n_attacks=n_its,
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
            input_data=self.victim.maybe_apply_user_template(temp_attack_ds["text"]),
            clf_label_data=temp_attack_ds["clf_label"],
            gen_target_data=temp_attack_ds["gen_target"],
            original_input_data=[example["text"] for _ in attacked_inputs],
        )

        with get_caching_model_with_example(self.victim, example["text"]) as victim:
            victim_out = self.victim_success_callback(
                victim,
                callback_input,
            )
        victim_successes: SUCCESS_TYPE
        if isinstance(victim_out, BinaryCallbackOutput):
            victim_successes = victim_out.successes
        else:
            assert isinstance(victim_out, TensorCallbackOutput)
            victim_successes = victim_out.losses

        # Copy callback output info to attacked_info
        attacked_info.update(victim_out.info)

        success_index = get_first_attack_success_index(victim_successes)
        attacked_text = attacked_inputs[success_index]
        attacked_info = {k: v[success_index] for k, v in attacked_info.items()}
        attacked_info["success_index"] = success_index
        return attacked_text, attacked_info, victim_successes

    def get_attacked_inputs(
        self,
        example: dict[str, Any],
        n_attacks: int,
        modifiable_chunk_spec: ModifiableChunkSpec,
    ) -> tuple[list[str], dict[str, Any]]:
        """Returns the attacked inputs for one example, one per iteration.

        Args:
            example: The example to be attacked.
            n_attacks: The number of attacked inputs to generate. (Typically
                this will be the number of iterations we intend to run.)
            modifiable_chunk_spec: The spec indicating which chunks are
                modifiable, and how.

        Returns:
            The attacked inputs for the example and a dict of attack-specific
            info for each iteration.
        """

        # TODO (ian): Speed this up by parallelizing if it's a bottleneck.
        attacked_inputs = []
        attacks_info: dict[str, list[Any]] = dict()
        for i in range(n_attacks):
            attacked_input, attack_info = self._get_attacked_input(
                example, modifiable_chunk_spec, current_iteration=i
            )
            attacked_inputs.append(attacked_input)
            attacks_info = {
                k: attacks_info.get(k, []) + [v] for k, v in attack_info.items()
            }
        return attacked_inputs, attacks_info

    def _get_attacked_input(
        self,
        example: dict[str, Any],
        modifiable_chunk_spec: ModifiableChunkSpec,
        current_iteration: int,
    ) -> tuple[str, dict[str, Any]]:
        """Returns one attacked input for one example.

        Args:
            example: The example to be attacked.
            modifiable_chunk_spec: The spec indicating which chunks are
                modifiable, and how.
            current_iteration: The current iteration of the attack.

        Returns:
            One attacked input for the example and a dict of attack-specific
            info for the current iteration.
        """
        if modifiable_chunk_spec.n_modifiable_chunks > 1:
            raise ValueError("Only one modifiable chunk per example supported for now.")

        current_input = ""
        chunk_count = 0
        attacked_info = {}
        for chunk_index, chunk_type in enumerate(modifiable_chunk_spec):
            chunk_text = example["chunked_text"][chunk_index]
            text, info = self._get_text_for_chunk(
                chunk_text=chunk_text,
                chunk_type=chunk_type,
                current_iteration=current_iteration,
                chunk_label=example["clf_label"],
                chunk_seed=example["seed"],
                chunk_index=chunk_count,
            )
            current_input += text
            if chunk_type != ChunkType.IMMUTABLE:
                chunk_count += 1
                attacked_info = info
        return current_input, attacked_info

    @abstractmethod
    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
        """Returns the attack tokens for the current iteration.

        Args:
            chunk_text: The text of the chunk to be attacked.
            chunk_type: The type of the chunk.
            current_iteration: The current iteration of the attack (only used in
            the multi-prompt case).
            chunk_label: The label of the chunk (for classification).
            chunk_seed: The seed for the chunk (for generation).

        Subclasses return:
            The attack tokens for the current iteration and a dict of attack-specific
            info for the current iteration.
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
        chunk_seed: int,
        chunk_index: int,
    ) -> tuple[str, dict[str, Any]]:
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
            The text for the chunk based on its type and a dictionary of attack-specific
            information for a single iteration.
        """
        match chunk_type:
            case ChunkType.IMMUTABLE:
                return chunk_text, {}

            case ChunkType.PERTURBABLE:
                token_ids, info = self._get_attack_tokens(
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
                return chunk_text + attack_tokens, info

            case ChunkType.OVERWRITABLE:
                token_ids, info = self._get_attack_tokens(
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
                return attack_tokens, info

            case _:
                raise ValueError(f"Unknown chunk type: {chunk_type}")

    def _get_multi_prompt_attacked_texts(
        self, dataset: RLLMDataset, n_its: int, success_indices: list[list[int]]
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
            best_attack_iteration = n_its - 1

        def get_best_attacked_input(example_index, example) -> str:
            """Get the attacked input for the best attack iteration.

            NOTE: We don't type-hint example because it doesn't play nicely with
            the list comprehension and we'd just have to ignore it anyway.
            """
            assert isinstance(example, dict)
            assert isinstance(example_index, int)
            example["seed"] = example_index
            return self._get_attacked_input(
                example=example,
                modifiable_chunk_spec=dataset.modifiable_chunk_spec,
                current_iteration=best_attack_iteration,
            )[0]

        attacked_texts = [
            get_best_attacked_input(example_index, example)
            for example_index, example in enumerate(dataset.ds)
        ]

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


def get_first_attack_success_index(
    victim_successes: SUCCESS_TYPE,
) -> int:
    """Returns the attacked text and the indices of successful attacks.

    Args:
        victim_successes: The successes of the model on the inputs. This can be
            binary success/fail or a floating point score.

    Returns:
        The index of the first successful attack (or the last attack if all failed).
    """
    if all([s is True for s in victim_successes]):
        # If there were no successful attacks, return the last attacked input.
        return len(victim_successes) - 1
    # The following one-liner is used to grab the first successful attack if
    # using binary success, or the *most* successful attack if using floats.
    # We take the min rather than the max because these successes are from
    # the perspective of the victim, so lower is better.
    _, attack_index = min((val, idx) for (idx, val) in enumerate(victim_successes))
    return attack_index
