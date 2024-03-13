import copy
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import wandb
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, pipeline
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.logging_utils import LoggingCounter
from robust_llm.utils import LanguageModel


class RandomTokenAttack(Attack):
    """Random token attack for non-Tomita datasets.

    Replaces all the modifiable text with random tokens
    from the victim tokenizer's vocabulary. The attack
    is repeated for each datapoint until it is successful,
    or until `attack_config.max_iterations` is reached.
    Appends the attack to the modifiable text instead
    of replacing it if `attack_config.append_to_modifiable_chunk`
    is True.
    """

    REQUIRES_INPUT_DATASET = True
    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: AttackConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        dataset_type: str,
        victim_model: LanguageModel,
        victim_tokenizer: PreTrainedTokenizerBase,
        ground_truth_label_fn: Optional[Callable[[str], int]],
    ) -> None:
        """Constructor for RandomTokenAttack.

        Args:
            attack_config: config of the attack
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            dataset_type: used dataset type
            victim_model: the model to be attacked
            victim_tokenizer: tokenizer used by the victim model
            ground_truth_label_fn: function to get the ground truth label
        """
        super().__init__(attack_config, modifiable_chunks_spec)

        assert True in modifiable_chunks_spec

        if dataset_type == "tomita":
            raise ValueError(
                "Random token attack is not supported for dataset type "
                f"{dataset_type}, exiting..."
            )

        self.victim_model = victim_model
        self.victim_tokenizer = victim_tokenizer
        self.ground_truth_label_fn = ground_truth_label_fn

        self.torch_rng = torch.random.manual_seed(self.attack_config.seed)

        self.min_tokens = self.attack_config.random_token_attack_config.min_tokens
        self.max_tokens = self.attack_config.random_token_attack_config.max_tokens
        self.max_iterations = (
            self.attack_config.random_token_attack_config.max_iterations
        )
        self.logging_frequency = (
            self.attack_config.random_token_attack_config.logging_frequency
        )
        self.batch_size = self.attack_config.random_token_attack_config.batch_size

        self.victim_pipeline = pipeline(
            "text-classification",
            model=victim_model,  # type: ignore
            tokenizer=victim_tokenizer,  # type: ignore
            device=victim_model.device,
        )

        self.logging_counter = LoggingCounter(_name="random_token_attack")

    @override
    def get_attacked_dataset(
        self, dataset: Optional[Dataset] = None, max_n_outputs: Optional[int] = None
    ) -> Tuple[Dataset, Dict[str, Any]]:

        assert dataset is not None and "text_chunked" in dataset.column_names

        if max_n_outputs is not None:
            dataset = dataset.select(range(max_n_outputs))

        attacked_text_chunked = copy.deepcopy(dataset["text_chunked"])
        original_labels = dataset["label"]
        attack_success = [False] * len(original_labels)

        # Replace all the modifiable text with random tokens from
        # the tokenizer's vocabulary.
        for iteration in range(self.max_iterations):
            if sum(attack_success) == len(attack_success):
                break

            num_skips = sum(attack_success)

            attacked_text_chunked = self._batch_get_adversarial_tokens(
                chunked_datapoints=attacked_text_chunked, successes=attack_success
            )
            attack_success = self._batch_check_success(
                attacked_chunked_datapoints=attacked_text_chunked,
                original_labels=original_labels,
                previous_attack_success=attack_success,
            )

            self._maybe_log_to_wandb(
                iteration=iteration, attack_success=attack_success, num_skips=num_skips
            )

        # Also replace dataset text, and delete tokenized text
        attacked_text = ["".join(line) for line in attacked_text_chunked]

        new_dataset = Dataset.from_dict(
            {
                "text": attacked_text,
                "original_text": dataset["text"],
                "label": dataset["label"],
            }
        )

        return new_dataset, {}

    def _batch_get_adversarial_tokens(
        self, chunked_datapoints: Sequence[Sequence[str]], successes: Sequence[bool]
    ) -> List[List[str]]:
        """Gets new random tokens in the modifiable chunks.

        Operates on a list of chunked datapoints (strings which have been split into
        "chunks", some of which are modifiable, some of which are not, as determined
        by the self.modifiable_chunks_spec). This method replaces those chunks which
        are modifiable with random tokens from the victim tokenizer's vocabulary.
        If attack_config.append_to_modifiable_chunk is True, the attack will add new
        chunks after the modifiable chunks instead of replacing them. Note that this
        will make it not match up with the modifiable_chunks_spec.

        Args:
            chunked_datapoints: The datapoints to operate on
            successes: A list of booleans indicating whether the attack has already
                been successful on this datapoint or not

        Returns:
            A list of chunked datapoints where previously successfully attacked
            datapoints remain unchanched, and previously not successfully attacked
            ones have their modifiable chunks replaced with random tokens.
        """

        assert len(chunked_datapoints) == len(successes)

        num_modifiable_chunks = sum(self.modifiable_chunks_spec)
        assert num_modifiable_chunks > 0

        num_not_success = len(chunked_datapoints) - sum(successes)

        all_num_tokens = torch.randint(
            low=self.min_tokens,
            high=self.max_tokens + 1,
            size=(num_not_success, num_modifiable_chunks),
            generator=self.torch_rng,
        )
        total_num_tokens = int(torch.sum(all_num_tokens).item())

        flat_random_tokens = torch.randint(
            low=0,
            high=self.victim_tokenizer.vocab_size,  # type: ignore
            size=(total_num_tokens,),
            generator=self.torch_rng,
        )
        decoded_flat_random_tokens = deque(
            self.victim_tokenizer.batch_decode(flat_random_tokens)
        )

        new_chunked_datapoints = []
        for chunked_datapoint, success in zip(chunked_datapoints, successes):
            if success:
                assert isinstance(chunked_datapoint, list)
                new_chunked_datapoints.append(chunked_datapoint)
            else:
                new_chunked_datapoint = []
                for text, is_modifiable in zip(
                    chunked_datapoint, self.modifiable_chunks_spec
                ):
                    if not is_modifiable:
                        new_chunked_datapoint.append(text)
                    else:
                        if self.attack_config.append_to_modifiable_chunk:
                            new_chunked_datapoint.append(text)
                        new_chunked_datapoint.append(
                            decoded_flat_random_tokens.popleft()
                        )
                new_chunked_datapoints.append(new_chunked_datapoint)

        return new_chunked_datapoints

    def _batch_check_success(
        self,
        attacked_chunked_datapoints: Sequence[Sequence[str]],
        original_labels: Sequence[int],
        previous_attack_success: Sequence[bool],
    ) -> List[bool]:
        """Checks the success of the attack on a batch of datapoints.

        Args:
            attacked_chunked_datapoints: The attacked texts
            original_labels: The original labels of the texts. Will only be
                used if the attack does not have a ground truth label function.
            previous_attack_success: A list of booleans indicating whether the attack
                has already been successful on each of the attacked texts.

        Returns:
            A list of booleans indicating whether the attack was successful on
            each of the attacked texts.
        """
        # Only run the victim model on the texts that
        # have not yet been successfully attacked
        datapoints = [
            "".join(text_chunked)
            for text_chunked, success in zip(
                attacked_chunked_datapoints, previous_attack_success
            )
            if not success
        ]

        results = self.victim_pipeline(datapoints, batch_size=self.batch_size)

        assert isinstance(results, list)
        assert len(results) == len(datapoints)

        result_labels = [result["label"] for result in results]  # type: ignore
        result_int_labels = [
            self.victim_model.config.label2id[result_label]  # type: ignore
            for result_label in result_labels
        ]
        assert all(isinstance(label, int) for label in result_int_labels)

        true_labels: Sequence[int]
        if self.ground_truth_label_fn is not None:
            true_labels = [self.ground_truth_label_fn(convo) for convo in datapoints]
        else:
            true_labels = [
                label
                for label, suc in zip(original_labels, previous_attack_success)
                if not suc
            ]

        new_successes = deque(
            [result != true for result, true in zip(result_int_labels, true_labels)]
        )

        updated_successes = []
        for success in previous_attack_success:
            if success:
                updated_successes.append(True)
            else:
                updated_successes.append(new_successes.popleft())

        assert len(updated_successes) == len(previous_attack_success)
        for old_suc, new_suc in zip(previous_attack_success, updated_successes):
            if old_suc is True:
                assert new_suc is True

        return updated_successes

    def _maybe_log_to_wandb(
        self,
        iteration: int,
        attack_success: Sequence[bool],
        num_skips: int,
    ):

        self.logging_counter.increment(
            step_count_to_add=1,
            datapoint_count_to_add=len(attack_success) - num_skips,
            commit=False,
        )

        if iteration % self.logging_frequency == 0:
            wandb.log(
                {
                    "num_attack_success": sum(attack_success),
                    "attack_success_rate": sum(attack_success) / len(attack_success),
                },
                commit=True,
            )

            print("num_attack_success", sum(attack_success))
            print("attack_success_rate", sum(attack_success) / len(attack_success))
