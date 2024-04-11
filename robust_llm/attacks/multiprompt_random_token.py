import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import wandb
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, pipeline
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.configs import AttackConfig, EnvironmentConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.logging_utils import LoggingCounter
from robust_llm.utils import LanguageModel


@dataclass
class IterationResult:
    iteration: int
    attack_sequences: list[str]
    attack_success_rate: float


class MultiPromptRandomTokenAttack(Attack):
    """Random token attack for *all* examples at once.

    Replaces all the modifiable text with random tokens
    from the victim tokenizer's vocabulary. The attack
    is repeated until it is successful for all examples,
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
        environment_config: EnvironmentConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        dataset_type: str,
        victim_model: LanguageModel,
        victim_tokenizer: PreTrainedTokenizerBase,
        ground_truth_label_fn: Optional[Callable[[str], int]],
    ) -> None:
        """Constructor for MultiPromptRandomTokenAttack.

        Args:
            attack_config: config of the attack
            environment_config: config of the environment
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            dataset_type: used dataset type
            victim_model: the model to be attacked
            victim_tokenizer: tokenizer used by the victim model
            ground_truth_label_fn: function to get the ground truth label
        """
        super().__init__(attack_config, environment_config, modifiable_chunks_spec)

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

        self.logging_counter = LoggingCounter(_name="multiprompt_random_token_attack")

    @override
    def get_attacked_dataset(
        self, dataset: Optional[Dataset] = None, max_n_outputs: Optional[int] = None
    ) -> Tuple[Dataset, Dict[str, Any]]:

        assert dataset is not None and "text_chunked" in dataset.column_names

        if max_n_outputs is not None:
            dataset = dataset.select(range(max_n_outputs))

        attacked_text_chunked = copy.deepcopy(dataset["text_chunked"])
        original_labels = dataset["label"]

        # Replace all the modifiable text with random tokens from
        # the tokenizer's vocabulary.

        iteration_results = []
        for iteration in (pbar := tqdm(range(self.max_iterations))):

            # no arguments because it's random each time
            attack_sequences = self._get_new_attack_sequences()
            attacked_text_chunked = self._construct_attacked_text_chunked(
                chunked_datapoints=attacked_text_chunked,
                attack_sequences=attack_sequences,
            )
            attack_success = self._batch_check_success(
                attacked_chunked_datapoints=attacked_text_chunked,
                original_labels=original_labels,
            )
            attack_success_rate = sum(attack_success) / len(attack_success)
            iteration_result = IterationResult(
                iteration=iteration,
                attack_sequences=attack_sequences,
                attack_success_rate=attack_success_rate,
            )
            iteration_results.append(iteration_result)

            self._maybe_log_to_wandb(
                iteration=iteration,
                attack_sequences=attack_sequences,
                attack_success=attack_success,
            )
            pbar.set_description(f"it {iteration}, success {attack_success_rate}")

        best_iteration_result = self._get_best_iteration_result(iteration_results)
        best_it = best_iteration_result.iteration
        best_rate = best_iteration_result.attack_success_rate
        print(f"Best iteration: {best_it} with {best_rate} success rate.")

        attacked_text_chunked = self._construct_attacked_text_chunked(
            chunked_datapoints=attacked_text_chunked,
            attack_sequences=best_iteration_result.attack_sequences,
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

    def _get_best_iteration_result(
        self,
        iteration_results: List[IterationResult],
    ) -> IterationResult:
        best_iteration_result = max(
            iteration_results, key=lambda x: x.attack_success_rate
        )
        return best_iteration_result

    def _construct_attacked_text_chunked(
        self,
        chunked_datapoints: Sequence[Sequence[str]],
        attack_sequences: List[str],
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
            attack_sequences: The attack sequences to insert into the modifiable chunks

        Returns:
            A list of chunked datapoints with the new attack sequences.
        """

        num_modifiable_chunks = sum(self.modifiable_chunks_spec)
        assert num_modifiable_chunks > 0

        new_chunked_datapoints = []
        for chunked_datapoint in chunked_datapoints:
            new_chunked_datapoint = []
            modifiable_chunk_idx = 0
            for text, is_modifiable in zip(
                chunked_datapoint, self.modifiable_chunks_spec
            ):
                if not is_modifiable:
                    new_chunked_datapoint.append(text)
                else:
                    if self.attack_config.append_to_modifiable_chunk:
                        new_chunked_datapoint.append(text)
                    new_chunked_datapoint.append(attack_sequences[modifiable_chunk_idx])
                    modifiable_chunk_idx += 1
            new_chunked_datapoints.append(new_chunked_datapoint)

        return new_chunked_datapoints

    def _get_new_attack_sequences(self) -> list[str]:
        num_modifiable_chunks = sum(self.modifiable_chunks_spec)
        new_attack_sequences = []
        for _ in range(num_modifiable_chunks):
            num_tokens = torch.randint(
                low=self.min_tokens,
                high=self.max_tokens,
                size=(1,),
                generator=self.torch_rng,
            ).item()

            random_tokens = torch.randint(
                low=0,
                high=self.victim_tokenizer.vocab_size,  # type: ignore
                size=(int(num_tokens),),
                generator=self.torch_rng,
            )
            decoded_sequence = self.victim_tokenizer.decode(random_tokens)
            new_attack_sequences.append(decoded_sequence)
        return new_attack_sequences

    def _batch_check_success(
        self,
        attacked_chunked_datapoints: Sequence[Sequence[str]],
        original_labels: Sequence[int],
    ) -> List[bool]:
        """Checks the success of the attack on a batch of datapoints.

        Args:
            attacked_chunked_datapoints: The attacked texts
            original_labels: The original labels of the texts. Will only be
                used if the attack does not have a ground truth label function.

        Returns:
            A list of booleans indicating whether the attack was successful on
            each of the attacked texts.
        """
        datapoints = [
            "".join(text_chunked) for text_chunked in attacked_chunked_datapoints
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
            true_labels = original_labels

        successes = [
            result != true for result, true in zip(result_int_labels, true_labels)
        ]

        return successes

    def _maybe_log_to_wandb(
        self,
        iteration: int,
        attack_sequences: List[str],
        attack_success: Sequence[bool],
    ):

        self.logging_counter.increment(
            step_count_to_add=1,
            datapoint_count_to_add=len(attack_success),
            commit=False,
        )

        if iteration % self.logging_frequency == 0:
            wandb.log(
                {
                    "num_attack_success": sum(attack_success),
                    "attack_success_rate": sum(attack_success) / len(attack_success),
                    "attack_sequences": attack_sequences,
                },
                commit=True,
            )

            print(
                f"iteration {iteration} attack_success_rate: "
                f"{sum(attack_success) / len(attack_success)}"
            )
