"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, TextClassificationPipeline

from robust_llm.attacks.attack import Attack
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

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        if "filters" in model_outputs and model_outputs["filters"][0]:
            # If the model has filtered out the example (e.g. PerplexityDefendedModel),
            # then there is no prediction to return.
            return {"label": None, "score": None}
        return super().postprocess(model_outputs, function_to_apply, top_k, _legacy)


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
    post_attack_losses: list[float] = dataclasses.field(default_factory=list)
    post_attack_correct_class_probs: list[float] = dataclasses.field(
        default_factory=list
    )

    @classmethod
    def from_labels_and_predictions(
        cls,
        original_labels: Sequence[int],
        attacked_labels: Sequence[int],
        original_pred_labels: Optional[Sequence[int | None]],
        attacked_pred_labels: Sequence[int | None],
        attacked_pred_logits: Sequence[Sequence[float | None]],
    ) -> "AttackResults":
        """Generates `AttackResults` from labels and model predictions.

        Args:
            original_labels: true labels of the original examples (pre-attack)
            attacked_labels: true labels of the attacked examples (post-attack)
            original_pred_labels: optional predictions of the model pre-attack. Should
                be provided iff the attack uses input dataset
            attacked_pred_labels: predictions of the model post-attack
            attacked_pred_logits: logits of the model post-attack

        Returns:
            `AttackResults` object
        """
        results = cls()

        if original_pred_labels is None:
            for attacked_label, attacked_pred_label, logits in zip(
                attacked_labels, attacked_pred_labels, attacked_pred_logits
            ):
                results._add_example_with_post_attack_only(
                    attacked_label=attacked_label,
                    attacked_pred_label=attacked_pred_label,
                    logits=logits,
                )
        else:
            for (
                original_label,
                attacked_label,
                original_pred_label,
                attacked_pred_label,
                logits,
            ) in zip(
                original_labels,
                attacked_labels,
                original_pred_labels,
                attacked_pred_labels,
                attacked_pred_logits,
            ):
                results._add_example(
                    original_label=original_label,
                    attacked_label=attacked_label,
                    original_pred_label=original_pred_label,
                    attacked_pred_label=attacked_pred_label,
                    logits=logits,
                )

        return results

    def _add_example_with_post_attack_only(
        self,
        attacked_label: int,
        attacked_pred_label: int | None,
        logits: Sequence[float | None],
    ) -> None:
        # Filtered out after attack -> true positive.
        if attacked_pred_label is None:
            self.n_filtered_out_post_attack += 1
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
        attacked_pred_label: int | None,
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
            # TODO(michal): Rethink how to compute those statistics if we care about
            # experiments with filtering.
            "adversarial_eval/pre_attack_accuracy": div_maybe_nan(
                self.n_attempted, self.n_total
            ),
            "adversarial_eval/post_attack_accuracy": div_maybe_nan(
                self.n_failures,
                self.n_total
                - self.n_filtered_out_post_attack
                - self.n_filtered_out_pre_attack,
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


def _get_prediction_logits(
    hf_pipeline: FilteredEvaluationPipeline,
    dataset: Dataset,
    batch_size: int,
) -> list[list[float | None]]:
    """Returns prediction logits, of shape (n_examples, n_classes)."""

    # TODO(michal): consider dropping HF pipelines and instead use data loaders + direct
    # model calls. Pipelines complicate the code without a clear benefit.
    preds = hf_pipeline(
        dataset["text"],
        batch_size=batch_size,
        truncation=True,
        function_to_apply="none",
    )
    assert preds is not None

    # We use explicit loops below instead of list comprehension to be able to have type
    # asserts so that typechecker does not complain.
    prediction_logits = []
    for pred in preds:
        # `pred` contains prediction information for a single example. It is a list of
        # dictionaries, each dictionary contains information about a single class.
        # Specifically, this dictionary includes a "score" for each class given
        # by the model. Dictionaries appear in the order of the classes in the model.
        assert type(pred) is list
        logits = []
        for i, cls_dict in enumerate(pred):
            assert type(cls_dict) is dict
            assert type(cls_dict["score"]) is float or cls_dict["score"] is None
            assert cls_dict["label"] == f"LABEL_{i}"
            logits.append(cls_dict["score"])
        prediction_logits.append(logits)

    return prediction_logits


def _get_prediction_labels_from_logits(
    prediction_logits: Sequence[Sequence[float | None]],
) -> list[int | None]:
    return [
        (None if None in logits else int(np.argmax(logits)))  # type: ignore
        for logits in prediction_logits
    ]


def get_prediction_logits_and_labels(
    dataset: Dataset,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> tuple[list[list[float | None]], list[int | None]]:
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
    pred_logits = _get_prediction_logits(
        hf_pipeline=hf_pipeline,
        dataset=dataset,
        batch_size=batch_size,
    )
    pred_labels = _get_prediction_labels_from_logits(pred_logits)

    return pred_logits, pred_labels


def _log_examples_to_wandb(
    original_texts: Optional[Sequence[str]],
    attacked_texts: Sequence[str],
    original_labels: Optional[Sequence[int]],
    attacked_labels: Sequence[int],
    original_pred_labels: Optional[Sequence[int | None]],
    attacked_pred_labels: Sequence[int | None],
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
                attacked_pred_logits[i],
            )

    else:
        table = wandb.Table(
            columns=[
                "attacked_text",
                "attacked_label",
                "attacked_pred",
                "attacked_logits",
            ]
        )
        for i in range(min(num_examples_to_log_detailed_info, len(attacked_texts))):
            table.add_data(
                attacked_texts[i],
                attacked_labels[i],
                attacked_pred_labels[i],
                attacked_pred_logits[i],
            )

    wandb.log({"adversarial_eval/examples": table}, commit=False)


def compute_attack_results(
    dataset: Optional[Dataset],
    attacked_dataset: Dataset,
    ground_truth_label_fn: Optional[Callable[[str], int]],
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    batch_size: int,
    num_examples_to_log_detailed_info: Optional[int],
) -> AttackResults:
    """Performs an attack and reports its results."""

    # Here we assume that either there was no original dataset, or that for every
    # example, there will be a corresponding item in the attacked dataset. Any new
    # attacks should conform to this assumption as it is needed for the evaluation
    # logic.
    if dataset is not None:
        assert len(dataset) == len(attacked_dataset)
        assert dataset["text"] == attacked_dataset["original_text"]
        assert dataset["label"] == attacked_dataset["label"]

    model.eval()

    assert dataset is not None  # will be assumed after refactors anyway.
    original_pred_logits, original_pred_labels = get_prediction_logits_and_labels(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    attacked_pred_logits, attacked_pred_labels = get_prediction_logits_and_labels(
        dataset=attacked_dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
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
            attacked_pred_logits=attacked_pred_logits,
            num_examples_to_log_detailed_info=num_examples_to_log_detailed_info,
        )

    return results


def do_adversarial_evaluation(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    dataset: Optional[Dataset],
    ground_truth_label_fn: Optional[Callable[[str], int]],
    num_generated_examples: Optional[int],
    attack: Attack,
    batch_size: int,
    num_examples_to_log_detailed_info: Optional[int],
) -> dict[str, float]:
    """Performs adversarial evaluation and logs the results."""
    if dataset is None:
        assert num_generated_examples is not None

    if dataset is not None and num_generated_examples is not None:
        dataset = dataset.select(range(min(num_generated_examples, len(dataset))))
        # We have already limited the dataset, so do not pass this value into
        # the `get_attacked_dataset` call below (note also that some some
        # attacks do not support passing it into `get_attacked_dataset`).
        num_generated_examples = None

    if dataset:
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
