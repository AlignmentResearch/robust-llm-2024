from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from torch import Tensor
from typing_extensions import override

from robust_llm.attacks.attack import (
    Attack,
    AttackedExample,
    AttackedRawInputOutput,
    AttackOutput,
    AttackState,
    PromptAttackMode,
)
from robust_llm.config.attack_configs import SearchFreeAttackConfig
from robust_llm.config.configs import ExperimentConfig
from robust_llm.dist_utils import DistributedRNG
from robust_llm.models.caching_wrapped_model import get_caching_model_with_example
from robust_llm.models.prompt_templates import AttackChunks
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


@dataclass(frozen=True)
class SearchFreeAttackedExample(AttackedExample):
    """Search-free attacked examples include success indices."""

    success_indices: list[int]


@dataclass(frozen=True)
class SearchFreeAttackState(AttackState):
    """State for search-free attacks, which includes success_indices."""

    previously_attacked_examples: tuple[SearchFreeAttackedExample, ...]

    def get_all_success_indices(self) -> list[list[int]]:
        """Returns the success indices for each example."""
        return [ex.success_indices for ex in self.previously_attacked_examples]


class SearchFreeAttack(Attack[SearchFreeAttackState], ABC):
    """Attack where each iteration is independent, no searching.

    The attack is repeated for each datapoint until it is successful,
    or until `attack_config.n_its` is reached.
    Appends the attack to the modifiable text instead
    of replacing it if the chunk is PERTURBABLE.
    """

    CAN_CHECKPOINT = True
    victim: WrappedModel
    prompt_attack_mode: PromptAttackMode
    victim_success_callback: BinaryCallback | TensorCallback

    def __init__(
        self,
        exp_config: ExperimentConfig,
        victim: WrappedModel,
        is_training: bool,
    ) -> None:
        super().__init__(exp_config, victim=victim, is_training=is_training)
        assert isinstance(self.attack_config, SearchFreeAttackConfig)
        self.prompt_attack_mode = PromptAttackMode(
            self.attack_config.prompt_attack_mode
        )
        # TODO(ian): Avoid this concession to the old structure of search-free
        # attacks by passing the rng state around better.
        self.rng = DistributedRNG(self.attack_config.seed, victim.accelerator)
        cb_config = self.attack_config.victim_success_callback
        self.victim_success_callback = build_scoring_callback(cb_config)

    @property
    def attack_state_class(self) -> type[AttackState]:
        """The AttackState class to use for this attack.

        TODO(ian): Find a cleaner way to get this information.
        """
        return SearchFreeAttackState

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int,
        resume_from_checkpoint: bool = True,
        epoch: int | None = None,
    ) -> AttackOutput:
        """Returns the attacked dataset and the attack metadata."""
        accelerator = self.victim.accelerator
        assert accelerator is not None

        start_attack_state = self.get_first_attack_state(resume_from_checkpoint, epoch)
        assert isinstance(start_attack_state, SearchFreeAttackState)
        checkpoint_path = self.get_attack_checkpoint_path(epoch)

        # TODO(ian): Avoid this concession to the existing structure by passing rng
        # state around better.
        self.rng = start_attack_state.rng_state.distributed_rng

        end_attack_state = self.run_attack_loop(
            dataset=dataset,
            attack_state=start_attack_state,
            checkpoint_path=checkpoint_path if resume_from_checkpoint else None,
            n_its=n_its,
        )

        attacked_input_texts = end_attack_state.get_attacked_input_texts()
        logits_cache = end_attack_state.get_logits_cache()
        all_iteration_texts = end_attack_state.get_all_iteration_texts()
        attack_flops = end_attack_state.get_attack_flops()
        all_success_indices = end_attack_state.get_all_success_indices()

        if self.prompt_attack_mode == PromptAttackMode.MULTIPROMPT:
            attacked_input_texts = self._get_multi_prompt_attacked_texts(
                dataset=dataset,
                n_its=n_its,
                success_indices=all_success_indices,
            )

        attacked_dataset = dataset.with_attacked_text(attacked_input_texts)

        attack_out = AttackOutput(
            dataset=attacked_dataset,
            attack_data=AttackedRawInputOutput(
                iteration_texts=all_iteration_texts, logits=logits_cache
            ),
            flops=attack_flops,
        )
        return attack_out

    def run_attack_on_example(
        self,
        dataset: RLLMDataset,
        n_its: int,
        attack_state: SearchFreeAttackState,
    ) -> SearchFreeAttackState:
        example_index = attack_state.example_index
        example = dataset.ds[example_index]
        assert isinstance(example, dict)
        example["seed"] = example_index
        # TODO (Oskar): account for examples in other partitions when indexing
        with self.victim.flop_count_context() as flop_count:
            attacked_text, example_info, victim_successes = self.attack_example(
                example, dataset, n_its
            )
        example_flops = flop_count.flops
        # TODO(ian): Work out if we should do this, or if it's handled by
        # _get_attacked_input.
        # attacked_text = self.victim.maybe_apply_user_template(attacked_text)

        # We look for False in the successes list, which indicates a successful
        # attack (i.e. that the model got the answer wrong after the attack).
        success_indices = (
            []
            if isinstance(victim_successes, Tensor)
            else [i for i, s in enumerate(victim_successes) if not s]
        )

        example_iteration_texts = example_info["iteration_texts"]
        example_logits: list[list[float]] | list[None] = example_info["logits"]

        # TODO(ian): Work out what to do with per_example_info
        # per_example_info = {
        #     k: per_example_info.get(k, []) + [v] for k, v in example_info.items()
        # }

        # Save the state after each example in case of interruption.
        new_attacked_example = SearchFreeAttackedExample(
            example_index=example_index,
            attacked_text=attacked_text,
            iteration_texts=example_iteration_texts,
            logits=example_logits,
            flops=example_flops,
            success_indices=success_indices,
        )
        new_previous_examples = attack_state.previously_attacked_examples + (
            new_attacked_example,
        )

        new_attack_state = SearchFreeAttackState(
            previously_attacked_examples=new_previous_examples,
            rng_state=attack_state.rng_state.update_states(),
        )
        return new_attack_state

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
        attacked_info = {
            k: v[success_index] for k, v in attacked_info.items() if v is not None
        }
        attacked_info["success_index"] = success_index

        inps, logits = _prepare_attack_data(
            attacked_inputs, victim_out.info.get("logits"), success_index
        )
        attacked_info["iteration_texts"] = inps
        attacked_info["logits"] = logits

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
                chunk_proxy_label=example["proxy_clf_label"],
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
        chunk_proxy_label: int,
        chunk_seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
        """Returns the attack tokens for the current iteration.

        Args:
            chunk_text: The text of the chunk to be attacked.
            chunk_type: The type of the chunk.
            current_iteration: The current iteration of the attack (only used in
            the multi-prompt case).
            chunk_proxy_label: The proxy label to use as the target in the attack.
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
        chunk_proxy_label: int,
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
            chunk_proxy_label: The proxy label to use as the target in the attack.
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
                attack_triple = AttackChunks("", chunk_text, "")
                template = attack_triple.get_prompt_template(
                    perturb_min=self.attack_config.perturb_position_min,
                    perturb_max=self.attack_config.perturb_position_max,
                    rng=self.rng,
                )
                token_ids, info = self._get_attack_tokens(
                    chunk_text,
                    chunk_type,
                    current_iteration,
                    chunk_proxy_label,
                    chunk_seed,
                )
                attack_tokens = self.victim.decode(token_ids)
                attack_tokens = self.post_process_attack_string(
                    attack_tokens, chunk_index
                )
                return (
                    template.before_attack + attack_tokens + template.after_attack,
                    info,
                )

            case ChunkType.OVERWRITABLE:
                token_ids, info = self._get_attack_tokens(
                    chunk_text,
                    chunk_type,
                    current_iteration,
                    chunk_proxy_label,
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


def _prepare_attack_data(
    attacked_inputs: list[str],
    logits: list[list[float]] | None,
    success_index: int,
) -> tuple[list[str], list[list[float]] | None]:
    """Prepare the attacked inputs and logits for saving in AttackedRawInputOutput.

    In a search-free attack, we don't want to save attacked strings beyond
    the first successful attack, since the rest are redundant. We also don't
    want to save logits beyond the first successful attack, since they are
    not useful. Thus we overwrite them with the final useful value.
    """
    useful_attacked_inputs = attacked_inputs[: success_index + 1]
    n_repeats = len(attacked_inputs) - len(useful_attacked_inputs)
    repeated_attacked_inputs = [useful_attacked_inputs[-1]] * n_repeats
    returned_attacked_inputs = useful_attacked_inputs + repeated_attacked_inputs

    if logits is None:
        return returned_attacked_inputs, None

    useful_logits = logits[: success_index + 1]
    repeated_logits = [useful_logits[-1]] * n_repeats
    returned_logits = useful_logits + repeated_logits
    return returned_attacked_inputs, returned_logits
