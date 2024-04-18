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
        flag_value: Optional[bool] = None
        if "filters" in model_outputs:
            flag_value = model_outputs["filters"][0].item()

        return labels_and_scores, flag_value  # type: ignore


@dataclass
class AttackResults:
    """Results of an attack on a dataset.

    When we run an attack on a dataset, we do the following things, in order:
    - run the victim model on the original dataset, and record which examples
        it gets correct or incorrect;
    - run the defense on the original dataset, and record how many examples it
        incorrectly flags as attacked;
    - run the attack on the examples from the original dataset that the model
        got correct. This gives us the "post-attack" dataset.
    - run the defense on the post-attack dataset, and record which examples
        it correctly flags as attacked.
    - run the victim model on the post-attack dataset, and record which examples
        it gets correct or incorrect, and whether or not that example was flagged
        by the defense.

    For an illustration of how the attack proceeds and what is being stored when,
    see the diagram at the following link:
    https://docs.google.com/presentation/d/1mxVyxYoZ3YXhK4MCvy3pdhVw-BOQTY-p-URmHeKP5DI
    In the list of attributes below, we put the letter corresponding to that section
    of the diagram in parentheses next to the attribute name,
    for those that are applicable.

    Please note that the attributes listed below are not disjoint. For example,
    a datapoint can be both flagged post-attack and correctly classified post-attack.

    Attributes:
        n_examples (a): Total number of datapoints in the original dataset.
        n_incorrect_pre_attack (b): Number of examples for which the model makes
            a mistake even before the attack.
        n_flagged_pre_attack (d): Number of examples from the original dataset
            that were flagged by the defense. This is a "false positive"
            for the defense: none of the original dataset has been attacked.
        n_flagged_post_attack (f): Number of examples from the attacked dataset
            that were flagged by the defense. This is a "true positive"
            for the defense: all of the attacked dataset has been attacked.
        n_flagged_post_attack_then_correct (i): Number of examples that were
            flagged by the defense post-attack, but which the model got correct.
        n_not_flagged_post_attack_then_correct (k): Number of examples that were
            not flagged by the defense post-attack, and which the model got correct.
        post_attack_losses: Losses of the victim model on the attacked examples.
        post_attack_correct_class_probs: Probabilities of the correct class for the
            attacked examples, according to the victim model.
    """

    # NOTE(niki): we assume that the length of the dataset
    # is the same pre and post attack.

    n_examples: int = 0
    n_incorrect_pre_attack: int = 0
    n_flagged_pre_attack: int = 0
    n_flagged_post_attack: int = 0
    n_flagged_post_attack_then_correct: int = 0
    n_not_flagged_post_attack_then_correct: int = 0

    post_attack_losses: list[float] = dataclasses.field(default_factory=list)
    post_attack_correct_class_probs: list[float] = dataclasses.field(
        default_factory=list
    )

    @property
    def n_correct_pre_attack(self) -> int:
        return self.n_examples - self.n_incorrect_pre_attack

    @property
    def n_not_flagged_pre_attack(self) -> int:
        return self.n_examples - self.n_flagged_pre_attack

    @property
    def n_not_flagged_post_attack(self) -> int:
        return self.n_correct_pre_attack - self.n_flagged_post_attack

    @property
    def n_flagged_post_attack_then_incorrect(self) -> int:
        return self.n_flagged_post_attack - self.n_flagged_post_attack_then_correct

    @property
    def n_not_flagged_post_attack_then_incorrect(self) -> int:
        return (
            self.n_not_flagged_post_attack - self.n_not_flagged_post_attack_then_correct
        )

    @property
    def n_post_attack_correct(self) -> int:
        return (
            self.n_flagged_post_attack_then_correct
            + self.n_not_flagged_post_attack_then_correct
        )

    @property
    def n_post_attack_incorrect(self) -> int:
        return (
            self.n_flagged_post_attack_then_incorrect
            + self.n_not_flagged_post_attack_then_incorrect
        )

    @classmethod
    def from_labels_and_predictions(
        cls,
        original_labels: Sequence[int],
        original_pred_labels: Optional[Sequence[int]],
        original_flag_values: Optional[Sequence[bool | None]],
        attacked_labels: Sequence[int],
        attacked_pred_labels: Sequence[int],
        attacked_flag_values: Sequence[bool | None],
        attacked_pred_logits: Sequence[Sequence[float | None]],
    ) -> "AttackResults":
        """Generates `AttackResults` from labels and model predictions.

        Args:
            original_labels: True labels of the original examples (pre-attack)
            original_pred_labels: Predictions of the model pre-attack. Should
                be provided iff the attack uses input dataset.
            original_flag_values: Values of the filter applied pre-attack. Should be
                provided iff the attack uses input dataset.
            attacked_labels: True labels of the attacked examples (post-attack)
            attacked_pred_labels: Predictions of the model post-attack
            attacked_flag_values: Values of the filter applied post-attack
            attacked_pred_logits: Logits of the model post-attack

        Returns:
            `AttackResults` object
        """
        results = cls()
        # TODO(niki): don't record the length here - instead, compute it as
        # we add examples in an _add_example... method below. For now, this
        # is not possible because we have two different methods, but after
        # an upcoming refactor in which we will have only one method, this
        # will be possible.
        results.n_examples = len(original_labels)

        if original_pred_labels is None:
            for (
                attacked_label,
                attacked_pred_label,
                attacked_flag_value,
                logits,
            ) in zip(
                attacked_labels,
                attacked_pred_labels,
                attacked_flag_values,
                attacked_pred_logits,
            ):
                results._add_example_with_post_attack_only(
                    attacked_label=attacked_label,
                    attacked_pred_label=attacked_pred_label,
                    attacked_flag_value=attacked_flag_value,
                    logits=logits,
                )
        else:
            assert original_flag_values is not None
            for (
                original_label,
                original_pred_label,
                original_flag_value,
                attacked_label,
                attacked_pred_label,
                attacked_flag_value,
                logits,
            ) in zip(
                original_labels,
                original_pred_labels,
                original_flag_values,
                attacked_labels,
                attacked_pred_labels,
                attacked_flag_values,
                attacked_pred_logits,
            ):
                results._add_example(
                    original_label=original_label,
                    original_pred_label=original_pred_label,
                    original_flag_value=original_flag_value,
                    attacked_label=attacked_label,
                    attacked_pred_label=attacked_pred_label,
                    attacked_flag_value=attacked_flag_value,
                    logits=logits,
                )

        return results

    def _add_example_with_post_attack_only(
        self,
        attacked_label: int,
        attacked_pred_label: int,
        attacked_flag_value: bool | None,
        logits: Sequence[float | None],
    ) -> None:
        if attacked_flag_value is True:
            self.n_flagged_post_attack += 1
            if attacked_label == attacked_pred_label:
                self.n_flagged_post_attack_then_correct += 1
        # This branch catches both the case where the defense did not flag the
        # example (False), and the case in which the defense was not run (None)
        else:
            if attacked_label == attacked_pred_label:
                self.n_not_flagged_post_attack_then_correct += 1

        # Compute the loss and the probability of the correct class.
        assert all(logit is not None for logit in logits)
        logprobs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=0)
        loss = float(-logprobs[attacked_label])
        correct_class_prob = float(torch.exp(logprobs[attacked_label]))
        self.post_attack_losses.append(loss)
        self.post_attack_correct_class_probs.append(correct_class_prob)

    def _add_example(
        self,
        original_label: int,
        original_pred_label: int,
        original_flag_value: bool | None,
        attacked_label: int,
        attacked_pred_label: int,
        attacked_flag_value: bool | None,
        logits: Sequence[float | None],
    ) -> None:
        if original_flag_value:
            self.n_flagged_pre_attack += 1

        if original_pred_label != original_label:
            self.n_incorrect_pre_attack += 1
            # Since the model got this example wrong pre-attack,
            # there was no need to run the attack, and thus we
            # ignore the attack's output even if it is present.
            return

        # If we got here, the model gave the correct answer
        # pre-attack and the attack was run.
        self._add_example_with_post_attack_only(
            attacked_label=attacked_label,
            attacked_pred_label=attacked_pred_label,
            attacked_flag_value=attacked_flag_value,
            logits=logits,
        )

    def compute_adversarial_evaluation_metrics(self) -> dict[str, float]:
        """Computes final metrics to report."""

        assert (
            len(self.post_attack_losses)
            == len(self.post_attack_correct_class_probs)
            == self.n_correct_pre_attack
        )

        return {
            # "Base" metrics
            "adversarial_eval/n_examples": self.n_examples,
            "adversarial_eval/n_correct_pre_attack": self.n_correct_pre_attack,
            "adversarial_eval/n_incorrect_pre_attack": self.n_incorrect_pre_attack,
            "adversarial_eval/n_flagged_pre_attack": self.n_flagged_pre_attack,
            "adversarial_eval/n_not_flagged_pre_attack": self.n_not_flagged_pre_attack,
            "adversarial_eval/n_flagged_post_attack": self.n_flagged_post_attack,
            "adversarial_eval/n_not_flagged_post_attack": (
                self.n_not_flagged_post_attack
            ),
            "adversarial_eval/n_flagged_post_attack_then_correct": (
                self.n_flagged_post_attack_then_correct
            ),
            "adversarial_eval/n_flagged_post_attack_then_incorrect": (
                self.n_flagged_post_attack_then_incorrect
            ),
            "adversarial_eval/n_not_flagged_post_attack_then_correct": (
                self.n_not_flagged_post_attack_then_correct
            ),
            "adversarial_eval/n_not_flagged_post_attack_then_incorrect": (
                self.n_not_flagged_post_attack_then_incorrect
            ),
            "adversarial_eval/n_post_attack_correct": self.n_post_attack_correct,
            "adversarial_eval/n_post_attack_incorrect": self.n_post_attack_incorrect,
            # Computed metrics
            "adversarial_eval/pre_attack_flagging_rate": div_maybe_nan(
                self.n_flagged_pre_attack, self.n_examples
            ),
            "adversarial_eval/pre_attack_accuracy": div_maybe_nan(
                self.n_correct_pre_attack, self.n_examples
            ),
            "adversarial_eval/post_attack_flagging_rate": div_maybe_nan(
                self.n_flagged_post_attack,
                self.n_correct_pre_attack,
            ),
            "adversarial_eval/post_attack_accuracy_including_original_mistakes": (
                div_maybe_nan(self.n_post_attack_correct, self.n_examples)
            ),
            "adversarial_eval/attack_success_rate": div_maybe_nan(
                self.n_post_attack_incorrect, self.n_correct_pre_attack
            ),
            "adversarial_eval/post_attack_accuracy_on_pre_attack_correct_examples": (
                div_maybe_nan(self.n_post_attack_correct, self.n_correct_pre_attack)
            ),
            "adversarial_eval/post_attack_accuracy_on_not_flagged_examples": (
                div_maybe_nan(
                    self.n_not_flagged_post_attack_then_correct,
                    self.n_not_flagged_post_attack,
                )
            ),
            "adversarial_eval/attack_post_defense_success_rate": div_maybe_nan(
                self.n_not_flagged_post_attack_then_incorrect, self.n_correct_pre_attack
            ),
            "adversarial_eval/post_attack_robustness_rate": div_maybe_nan(
                self.n_flagged_post_attack
                + self.n_not_flagged_post_attack_then_correct,
                self.n_correct_pre_attack,
            ),
            "adversarial_eval/defense_post_attack_true_positive_rate": div_maybe_nan(
                self.n_flagged_post_attack, self.n_correct_pre_attack
            ),
            "adversarial_eval/defense_post_attack_true_negative_rate": div_maybe_nan(
                self.n_not_flagged_pre_attack, self.n_examples
            ),
            "adversarial_eval/defense_post_attack_false_positive_rate": div_maybe_nan(
                self.n_flagged_pre_attack, self.n_examples
            ),
            "adversarial_eval/defense_post_attack_false_negative_rate": div_maybe_nan(
                self.n_not_flagged_post_attack, self.n_correct_pre_attack
            ),
            "adversarial_eval/post_attack_flagged_but_correct_rate": div_maybe_nan(
                self.n_flagged_post_attack_then_correct, self.n_flagged_post_attack
            ),
            "adversarial_eval/post_attack_not_flagged_but_incorrect_rate": (
                div_maybe_nan(
                    self.n_not_flagged_post_attack_then_incorrect,
                    self.n_not_flagged_post_attack,
                )
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
) -> tuple[list[list[float]], list[bool | None]]:
    """Returns prediction logits, of shape (n_examples, n_classes)."""

    # TODO(michal): consider dropping HF pipelines and instead use data loaders + direct
    # model calls. Pipelines complicate the code without a clear benefit.
    preds_with_flag_values = hf_pipeline(
        dataset["text"],
        batch_size=batch_size,
        truncation=True,
        function_to_apply="none",
    )

    # We use explicit loops below instead of list comprehension to be able to have type
    # asserts so that typechecker does not complain.
    prediction_logits = []
    flag_values: list[bool | None] = []
    assert isinstance(preds_with_flag_values, list)
    for pred, flag_value in preds_with_flag_values:  # type: ignore
        assert isinstance(flag_value, bool) or flag_value is None
        flag_values.append(flag_value)

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

    return prediction_logits, flag_values


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
    original_labels: Optional[Sequence[int]],
    original_pred_labels: Optional[Sequence[int | None]],
    attacked_texts: Sequence[str],
    attacked_labels: Sequence[int],
    attacked_pred_labels: Sequence[int | None],
    attacked_flag_values: Sequence[bool | None],
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
                attacked_flag_values[i],
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
                attacked_flag_values[i],
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

    pred_logits, pred_flag_values = _get_prediction_logits_and_maybe_filter(
        hf_pipeline=hf_pipeline,
        dataset=dataset,
        batch_size=batch_size,
    )
    pred_labels = _get_prediction_labels(pred_logits)

    return pred_logits, pred_labels, pred_flag_values


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
    original_pred_logits, original_pred_labels, original_flag_values = (
        _get_prediction_logits_and_labels_and_maybe_filter(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    )

    attacked_pred_logits, attacked_pred_labels, attacked_flag_values = (
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
        original_pred_labels=original_pred_labels,
        original_flag_values=original_flag_values,
        attacked_labels=attacked_labels,
        attacked_pred_labels=attacked_pred_labels,
        attacked_flag_values=attacked_flag_values,
        attacked_pred_logits=attacked_pred_logits,
    )

    if num_examples_to_log_detailed_info is not None and accelerator.is_main_process:
        _log_examples_to_wandb(
            original_texts=dataset["text"] if dataset else None,
            original_labels=dataset["label"] if dataset else None,
            original_pred_labels=original_pred_labels,
            attacked_texts=attacked_dataset["text"],
            attacked_labels=attacked_labels,
            attacked_pred_labels=attacked_pred_labels,
            attacked_flag_values=attacked_flag_values,
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
