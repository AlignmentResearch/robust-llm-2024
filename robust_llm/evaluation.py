"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, TextClassificationPipeline

from robust_llm.attacks.attack import Attack
from robust_llm.defenses.perplexity import PerplexityDefendedModel
from robust_llm.utils import LanguageModel, div_maybe_nan


class FilteredEvaluationPipeline(TextClassificationPipeline):
    @contextmanager
    def override_model(self):
        # Save the old value
        old_model = self.model  # type: ignore
        # Set the new value
        self.model = getattr(self.model, "init_model", old_model)  # type: ignore

        try:
            # Now allow the code block to run with the new value
            yield
        finally:
            # Revert the value back to the old value
            self.model = old_model

    def check_model_type(self, supported_models: Union[list[str], dict]):
        """
        Check if the model class is supported by the pipeline.

        Args:
            supported_models (`list[str]` or `dict`):
                The list of models supported by the pipeline, or
                a dictionary with model class values.
        """
        with self.override_model():
            super().check_model_type(supported_models)

    def postprocess(  # type: ignore
        self, model_outputs, function_to_apply=None, top_k=1, _legacy=True
    ) -> tuple[dict[str, str | float], bool]:
        """Postprocess the output of the model. While it replaces the original
        `postprocess` method, its return type is different, since it also
        optionally returns the filter value.
        TODO(niki): is there a better way of doing this? We could artificially
        add a new key to the output, but that's not so clean either.

        Args:
            model_outputs: The output of the model.
            function_to_apply: The function to apply to the model output
                in order to get the labels and scores.
            top_k: The number of labels to return, sorted in descending order.
            _legacy: Whether to use the legacy postprocessing logic.

        Returns:
            The labels and scores, and optionally, the filter value.
        """
        labels_and_scores = super().postprocess(
            model_outputs, function_to_apply, top_k, _legacy
        )
        filter_value: Optional[bool] = None
        if "filters" in model_outputs:
            filter_value = model_outputs["filters"][0].item()

        return labels_and_scores, filter_value  # type: ignore


@dataclass
class AttackResults:
    """Results of an attack on a dataset.

    Note that we have a simplified division into classes for now. In general, there
    are three different possible outcomes (correct prediction, incorrect prediction,
    filtered out) for both pre-attack and post-attack predictions, which gives us 9
    different combinations. However, for simplification and to eliminate some degenerate
    cases, we make the following assumptions:
    - if an example was filtered out pre-attack, we do not care what happens next and
    always treat it as a "filtered_out_before_attack" class;
    - if an example was misclassified pre-attack, we do not care what happens next and
    always treat it as a "mistake_before_attack" class;
    - otherwise (correct prediction pre-attack), we subdivide into 3 different classes
    based on the post-attack prediction (correct, incorrect, filtered out).

    Note 1: If no filtering defense is used, there are only 3 classes left and so there
    is no controversy about the division.

    Note 2: If we want to refine our class division in the future, we might consider
    not using defense pre-attack -> having only correct / incorrect options pre-attack.

    Attributes:
        n_filtered_out_pre_attack: Number of examples that were filtered out by the
            defense pre-attack. This is a false positive from the perspective of the
            defender.
        n_mistakes_pre_attack: Number of examples for which the model makes a mistake
            even before the attack, so for the sake of the metrics we do not care about
            them; attack might have been skipped for these cases.
        n_filtered_out_post_attack: Number of examples that were filtered out by the
            defense post-attack. This is a true positive from the perspective of the
            defender.
        n_failures: Number of examples that were not successfully attacked.
        n_successes: Number of examples that were successfully attacked.
        n_post_attack_false_positives: Number of examples that were filtered out by the
            defense post-attack, but which the model would have got correct.
        post_attack_losses: Losses of the model on the attacked examples
            (only considers examples that were failures or successes, not ones that were
            mistakes pre attack or filtered out pre or post-attack).
        post_attack_correct_class_probs: Probabilities of the correct class for the
            attacked examples (only considers examples that were failures or successes,
            not ones that were mistakes pre attack or filtered out pre or post-attack).
    """

    n_filtered_out_pre_attack: int = 0
    n_mistakes_pre_attack: int = 0
    n_filtered_out_post_attack: int = 0
    n_failures: int = 0
    n_successes: int = 0
    n_post_attack_false_positives: int = 0
    post_attack_losses: list[float] = dataclasses.field(default_factory=list)
    post_attack_correct_class_probs: list[float] = dataclasses.field(
        default_factory=list
    )

    @property
    def p_filtered_out_post_attack(self) -> float:
        return div_maybe_nan(self.n_filtered_out_post_attack, self.n_attempted)

    @classmethod
    def from_labels_and_predictions(
        cls,
        original_labels: Sequence[int],
        attacked_labels: Sequence[int],
        original_pred_labels: Optional[Sequence[int | None]],
        attacked_pred_labels: Sequence[int],
        attacked_filter_values: Sequence[bool | None],
        attacked_pred_logits: Sequence[Sequence[float | None]],
    ) -> "AttackResults":
        """Generates `AttackResults` from labels and model predictions.

        Args:
            original_labels: true labels of the original examples (pre-attack)
            attacked_labels: true labels of the attacked examples (post-attack)
            original_pred_labels: optional predictions of the model pre-attack. Should
                be provided iff the attack uses input dataset
            attacked_pred_labels: predictions of the model post-attack
            attacked_filter_values: values of the filter applied post-attack
            attacked_pred_logits: logits of the model post-attack

        Returns:
            `AttackResults` object
        """
        results = cls()

        if original_pred_labels is None:
            for (
                attacked_label,
                attacked_pred_label,
                attacked_filter_value,
                logits,
            ) in zip(
                attacked_labels,
                attacked_pred_labels,
                attacked_filter_values,
                attacked_pred_logits,
            ):
                results._add_example_with_post_attack_only(
                    attacked_label=attacked_label,
                    attacked_pred_label=attacked_pred_label,
                    attacked_filter_value=attacked_filter_value,
                    logits=logits,
                )
        else:
            for (
                original_label,
                attacked_label,
                original_pred_label,
                attacked_pred_label,
                attacked_filter_value,
                logits,
            ) in zip(
                original_labels,
                attacked_labels,
                original_pred_labels,
                attacked_pred_labels,
                attacked_filter_values,
                attacked_pred_logits,
            ):
                results._add_example(
                    original_label=original_label,
                    attacked_label=attacked_label,
                    original_pred_label=original_pred_label,
                    attacked_pred_label=attacked_pred_label,
                    attacked_filter_value=attacked_filter_value,
                    logits=logits,
                )

        return results

    def _add_example_with_post_attack_only(
        self,
        attacked_label: int,
        attacked_pred_label: int,
        attacked_filter_value: bool | None,
        logits: Sequence[float | None],
    ) -> None:
        # Filtered out after attack -> true positive.
        if attacked_filter_value:
            self.n_filtered_out_post_attack += 1
            if attacked_label == attacked_pred_label:
                self.n_post_attack_false_positives += 1
        else:
            # Correct prediction after attack -> failure.
            if attacked_label == attacked_pred_label:
                self.n_failures += 1
            # Incorrect prediction after attack -> success.
            else:
                self.n_successes += 1

            # In case of failure or success, compute the loss and the probability of
            # the correct class.
            assert all(logit is not None for logit in logits)
            logprobs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=0)
            loss = float(-logprobs[attacked_label])
            correct_class_prob = float(torch.exp(logprobs[attacked_label]))
            self.post_attack_losses.append(loss)
            self.post_attack_correct_class_probs.append(correct_class_prob)

    def _add_example(
        self,
        original_label: int,
        attacked_label: int,
        original_pred_label: int | None,
        attacked_pred_label: int,
        attacked_filter_value: bool | None,
        logits: Sequence[float | None],
    ) -> None:
        # Example was filtered out pre-attack -> false positive.
        if original_pred_label is None:
            self.n_filtered_out_pre_attack += 1
        # Prediction was already incorrect, so we expect this example should have
        # been skipped and we do not consider it either a success or a failure.
        elif original_pred_label != original_label:
            self.n_mistakes_pre_attack += 1
        # Prediction was correct pre-attack. Now we need to check the post-attack.
        else:
            self._add_example_with_post_attack_only(
                attacked_label=attacked_label,
                attacked_pred_label=attacked_pred_label,
                attacked_filter_value=attacked_filter_value,
                logits=logits,
            )

    @property
    def n_total(self) -> int:
        """Total number of examples used."""
        return (
            self.n_filtered_out_pre_attack
            + self.n_mistakes_pre_attack
            + self.n_filtered_out_post_attack
            + self.n_failures
            + self.n_successes
        )

    @property
    def n_attempted(self) -> int:
        """Total number of examples for which the attack should have been attempted."""
        return self.n_successes + self.n_failures + self.n_filtered_out_post_attack

    def compute_adversarial_evaluation_metrics(self) -> dict[str, float]:
        """Computes final metrics to report."""

        assert (
            len(self.post_attack_losses)
            == len(self.post_attack_correct_class_probs)
            == self.n_failures + self.n_successes
        )

        return {
            "adversarial_eval/pre_attack_accuracy": div_maybe_nan(
                self.n_attempted, self.n_total
            ),
            "adversarial_eval/post_attack_accuracy": div_maybe_nan(
                self.n_failures,
                self.n_total
                - self.n_filtered_out_post_attack
                - self.n_filtered_out_pre_attack,
            ),
            "adversarial_eval/post_attack_accuracy_without_previous_mistakes": div_maybe_nan(  # noqa: E501
                self.n_failures,
                self.n_total
                - self.n_filtered_out_post_attack
                - self.n_filtered_out_pre_attack
                - self.n_mistakes_pre_attack,
            ),
            "adversarial_eval/p_filtered_out_post_attack": self.p_filtered_out_post_attack,  # noqa: E501
            "adversarial_eval/post_attack_false_positive_rate": div_maybe_nan(
                self.n_post_attack_false_positives, self.n_filtered_out_post_attack
            ),
            "adversarial_eval/attack_success_rate": div_maybe_nan(
                self.n_successes, self.n_attempted
            ),
            "adversarial_eval/n_total_examples_used": self.n_total,
            "adversarial_eval/filtering_false_positive_rate": div_maybe_nan(
                self.n_filtered_out_pre_attack,
                self.n_total,
            ),
            "adversarial_eval/filtering_true_positive_rate": div_maybe_nan(
                self.n_filtered_out_post_attack,
                self.n_attempted,
            ),
            "adversarial_eval/avg_post_attack_loss": float(
                np.mean(self.post_attack_losses)
            ),
            "adversarial_eval/avg_post_attack_correct_class_prob": float(
                np.mean(self.post_attack_correct_class_probs)
            ),
        }


def _maybe_record_defense_specific_metrics(
    model: LanguageModel, dataset: Dataset, attacked_dataset: Dataset
) -> dict:

    metrics = {}

    if (
        isinstance(model, PerplexityDefendedModel)
        and model.defense_config.perplexity_defense_config.save_perplexity_curves
    ):
        # Get the approximate perplexities of the decoder on
        # on both datasets.

        # We don't pass in input_ids or attention_mask here because
        # get_all_perplexity_thresholds gets confused
        original_text_dataset = Dataset.from_dict({"text": dataset["text"]})
        metrics["perplexity/decoder_perplexities_original"] = (  # type: ignore
            model.get_all_perplexity_thresholds(dataset=original_text_dataset)  # type: ignore  # noqa: E501
        )

        # The attacked dataset only has `text`, `original_text`, and `label`,
        # so we're fine to pass it in as is.
        metrics["perplexity/decoder_perplexities_attacked"] = (  # type: ignore
            model.get_all_perplexity_thresholds(dataset=attacked_dataset)  # type: ignore  # noqa: E501
        )

    return metrics


def _get_prediction_logits_and_maybe_filter(
    hf_pipeline: FilteredEvaluationPipeline,
    dataset: Dataset,
    batch_size: int,
) -> Tuple[list[list[float]], list[bool | None]]:
    """Returns prediction logits, of shape (n_examples, n_classes)."""

    # TODO(michal): consider dropping HF pipelines and instead use data loaders + direct
    # model calls. Pipelines complicate the code without a clear benefit.
    preds_with_filter_values = hf_pipeline(
        dataset["text"],
        batch_size=batch_size,
        truncation=True,
        function_to_apply="none",
    )

    # We use explicit loops below instead of list comprehension to be able to have type
    # asserts so that typechecker does not complain.
    prediction_logits = []
    filter_values: list[bool | None] = []
    assert isinstance(preds_with_filter_values, list)
    for pred, filter_value in preds_with_filter_values:  # type: ignore
        assert isinstance(filter_value, bool) or filter_value is None
        filter_values.append(filter_value)

        # `pred` contains prediction information for a single example. It is a list of
        # dictionaries, each dictionary contains information about a single class.
        # Specifically, this dictionary includes a "score" for each class given
        # by the model. Dictionaries appear in the order of the classes in the model.
        assert type(pred) is list
        logits = []
        for i, cls_dict in enumerate(pred):
            assert type(cls_dict) is dict
            assert type(cls_dict["score"]) is float
            assert cls_dict["label"] == f"LABEL_{i}"
            logits.append(cls_dict["score"])
        prediction_logits.append(logits)

    return prediction_logits, filter_values


def _get_prediction_labels(
    prediction_logits: Sequence[Sequence[float]],
) -> list[int]:
    return [int(np.argmax(logits)) for logits in prediction_logits]


def get_prediction_logits_and_labels(
    dataset: Dataset,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> tuple[list[list[float]], list[int]]:
    hf_pipeline = FilteredEvaluationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        # Even though this option is deprecated, we use it instead of setting
        # `top_k=None`. They differ by the order of returned classes. In the following
        # option, we get classes in the order of the classes defined in the model.
        return_all_scores=True,
        framework="pt",
    )
    pred_logits, _ = _get_prediction_logits_and_maybe_filter(
        hf_pipeline=hf_pipeline,
        dataset=dataset,
        batch_size=batch_size,
    )
    pred_labels = _get_prediction_labels(pred_logits)

    return pred_logits, pred_labels


def _log_examples_to_wandb(
    original_texts: Optional[Sequence[str]],
    attacked_texts: Sequence[str],
    original_labels: Optional[Sequence[int]],
    attacked_labels: Sequence[int],
    original_pred_labels: Optional[Sequence[int | None]],
    attacked_pred_labels: Sequence[int | None],
    attacked_filter_values: Sequence[bool | None],
    attacked_pred_logits: Sequence[Sequence[float | None]],
    num_examples_to_log_detailed_info: int,
) -> None:
    assert (
        (original_texts is None)
        == (original_labels is None)
        == (original_pred_labels is None)
    )

    if original_texts is not None:
        # Have those asserts here so that type checker does not complain.
        assert original_labels is not None
        assert original_pred_labels is not None

        table = wandb.Table(
            columns=[
                "original_text",
                "attacked_text",
                "original_label",
                "attacked_label",
                "original_pred",
                "attacked_pred",
                "attacked_filter",
                "attacked_logits",
            ]
        )
        for i in range(min(num_examples_to_log_detailed_info, len(attacked_texts))):
            table.add_data(
                original_texts[i],
                attacked_texts[i],
                original_labels[i],
                attacked_labels[i],
                original_pred_labels[i],
                attacked_pred_labels[i],
                attacked_filter_values[i],
                attacked_pred_logits[i],
            )

    else:
        table = wandb.Table(
            columns=[
                "attacked_text",
                "attacked_label",
                "attacked_pred",
                "attacked_filter",
                "attacked_logits",
            ]
        )
        for i in range(min(num_examples_to_log_detailed_info, len(attacked_texts))):
            table.add_data(
                attacked_texts[i],
                attacked_labels[i],
                attacked_pred_labels[i],
                attacked_filter_values[i],
                attacked_pred_logits[i],
            )

    wandb.log({"adversarial_eval/examples": table}, commit=False)


def _get_prediction_logits_and_labels_and_maybe_filter(
    dataset: Dataset,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> tuple[list[list[float]], list[int], list[bool | None]]:

    hf_pipeline = FilteredEvaluationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        # Even though this option is deprecated, we use it instead of setting
        # `top_k=None`. They differ by the order of returned classes. In the following
        # option, we get classes in the order of the classes defined in the model.
        return_all_scores=True,
        framework="pt",
    )

    pred_logits, pred_filter = _get_prediction_logits_and_maybe_filter(
        hf_pipeline=hf_pipeline,
        dataset=dataset,
        batch_size=batch_size,
    )
    pred_labels = _get_prediction_labels(pred_logits)

    return pred_logits, pred_labels, pred_filter


def compute_attack_results(
    dataset: Dataset,
    attacked_dataset: Dataset,
    ground_truth_label_fn: Optional[Callable[[str], int]],
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    batch_size: int,
    num_examples_to_log_detailed_info: Optional[int],
) -> AttackResults:
    """Performs an attack and reports its results."""

    assert len(dataset) == len(attacked_dataset)
    assert dataset["text"] == attacked_dataset["original_text"]
    assert dataset["label"] == attacked_dataset["label"]

    model.eval()

    assert dataset is not None  # will be assumed after refactors anyway.
    original_pred_logits, original_pred_labels, original_filter_values = (
        _get_prediction_logits_and_labels_and_maybe_filter(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    )

    attacked_pred_logits, attacked_pred_labels, attacked_filter_values = (
        _get_prediction_logits_and_labels_and_maybe_filter(
            dataset=attacked_dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    )

    # If the dataset allows for recomputation of the ground truth label, use it.
    if ground_truth_label_fn is not None:
        attacked_labels = [
            ground_truth_label_fn(text) for text in attacked_dataset["text"]
        ]
    # Otherwise, assume that the labels have not changed (may or may not be accurate).
    else:
        attacked_labels = attacked_dataset["label"]

    results = AttackResults.from_labels_and_predictions(
        original_labels=attacked_dataset["label"],
        attacked_labels=attacked_labels,
        original_pred_labels=original_pred_labels,
        attacked_pred_labels=attacked_pred_labels,
        attacked_filter_values=attacked_filter_values,
        attacked_pred_logits=attacked_pred_logits,
    )

    if num_examples_to_log_detailed_info is not None and accelerator.is_main_process:
        _log_examples_to_wandb(
            original_texts=dataset["text"] if dataset else None,
            attacked_texts=attacked_dataset["text"],
            original_labels=dataset["label"] if dataset else None,
            attacked_labels=attacked_labels,
            original_pred_labels=original_pred_labels,
            attacked_pred_labels=attacked_pred_labels,
            attacked_filter_values=attacked_filter_values,
            attacked_pred_logits=attacked_pred_logits,
            num_examples_to_log_detailed_info=num_examples_to_log_detailed_info,
        )

    return results


def do_adversarial_evaluation(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    dataset: Dataset,
    ground_truth_label_fn: Optional[Callable[[str], int]],
    num_generated_examples: Optional[int],
    attack: Attack,
    batch_size: int,
    num_examples_to_log_detailed_info: Optional[int],
) -> dict[str, float]:
    """Performs adversarial evaluation and logs the results."""
    if num_generated_examples is not None:
        dataset = dataset.select(range(min(num_generated_examples, len(dataset))))
        # We have already limited the dataset, so do not pass this value into
        # the `get_attacked_dataset` call below (note also that some some
        # attacks do not support passing it into `get_attacked_dataset`).
        num_generated_examples = None

    # Sanity check in case of a distributed run (with accelerate): check if every
    # process has the same dataset.
    # TODO(michal): Look into datasets code and make sure the dataset creation is
    # deterministic given seeds; this is especially important when using accelerate.
    _assert_same_data_between_processes(accelerator, dataset["text"])
    _assert_same_data_between_processes(accelerator, dataset["label"])

    print("Doing adversarial evaluation...")

    attacked_dataset, info_dict = attack.get_attacked_dataset(
        dataset=dataset, max_n_outputs=num_generated_examples
    )

    print("got the attacked dataset")

    attack_results = compute_attack_results(
        dataset=dataset,
        attacked_dataset=attacked_dataset,
        ground_truth_label_fn=ground_truth_label_fn,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        batch_size=batch_size,
        num_examples_to_log_detailed_info=num_examples_to_log_detailed_info,
    )
    print(attack_results)

    metrics = attack_results.compute_adversarial_evaluation_metrics()
    assert len(set(metrics.keys()) & set(info_dict.keys())) == 0
    metrics |= info_dict

    metrics |= _maybe_record_defense_specific_metrics(
        model=model, dataset=dataset, attacked_dataset=attacked_dataset  # type: ignore
    )

    # TODO(GH#158): Refactor/unify logging.
    if accelerator.is_main_process:
        wandb.log(metrics, commit=True)
        print("Adversarial evaluation metrics:")
        print(metrics)

    return metrics


def _assert_same_data_between_processes(
    accelerator: Accelerator, data: Sequence[Any]
) -> None:
    length = len(data)
    data_gathered = accelerator.gather_for_metrics(data)
    for i in range(accelerator.num_processes):
        assert data_gathered[i * length : (i + 1) * length] == data
