"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, TextClassificationPipeline

from robust_llm.attacks.attack import Attack
from robust_llm.defenses.perplexity import PerplexityDefendedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
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
        """Check if the model class is supported by the pipeline.

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
class EvaluationOutput:
    """Represents the output of evaluating an attack on a single example.

    Attributes:
        original_label: True label of the original example.
        original_pred_label: Prediction of the model on the original example.
        original_flag_value: Value of the filter applied pre-attack. Will be
            `None` iff the defense was not run pre-attack.
        attacked_label: True label of the attacked example.
        attacked_pred_label: Prediction of the model on the attacked example.
        attacked_flag_value: Value of the filter applied post-attack. Will be
            `None` iff the defense was not applied post-attack.
        attacked_pred_logits: Logits of the model on the attacked example.
    """

    original_label: int
    original_pred_label: int
    original_flag_value: Optional[bool]
    attacked_label: int
    attacked_pred_label: int
    attacked_flag_value: Optional[bool]
    attacked_pred_logits: list[float]

    @classmethod
    def get_list_from_labels_and_predictions(
        cls,
        original_labels: Sequence[int],
        original_pred_labels: Sequence[int],
        original_flag_values: Sequence[bool] | Sequence[None],
        attacked_labels: Sequence[int],
        attacked_pred_labels: Sequence[int],
        attacked_flag_values: Sequence[bool] | Sequence[None],
        attacked_pred_logits: Sequence[Sequence[float]],
    ) -> list[EvaluationOutput]:

        assert (
            len(original_labels)
            == len(original_pred_labels)
            == len(original_flag_values)
            == len(attacked_labels)
            == len(attacked_pred_labels)
            == len(attacked_flag_values)
            == len(attacked_pred_logits)
        )

        evaluation_outputs = []
        for i in range(len(original_labels)):
            evaluation_outputs.append(
                cls(
                    original_label=original_labels[i],
                    original_pred_label=original_pred_labels[i],
                    original_flag_value=original_flag_values[i],
                    attacked_label=attacked_labels[i],
                    attacked_pred_label=attacked_pred_labels[i],
                    attacked_pred_logits=list(attacked_pred_logits[i]),
                    attacked_flag_value=attacked_flag_values[i],
                )
            )

        return evaluation_outputs


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

    def __init__(self, evaluation_outputs: Sequence[EvaluationOutput]):
        # NOTE(niki): we assume that the length of the dataset
        # is the same pre and post attack.
        self.n_examples: int = 0
        self.n_incorrect_pre_attack: int = 0
        self.n_flagged_pre_attack: int = 0
        self.n_flagged_post_attack: int = 0
        self.n_flagged_post_attack_then_correct: int = 0
        self.n_not_flagged_post_attack_then_correct: int = 0

        self.post_attack_losses: list[float] = []
        self.post_attack_correct_class_probs: list[float] = []

        for evaluation_output in evaluation_outputs:
            self._add_example(evaluation_output)

    def __repr__(self):
        return f"""
    AttackResults(
        n_examples={self.n_examples},
        n_correct_pre_attack={self.n_correct_pre_attack},
        n_incorrect_pre_attack={self.n_incorrect_pre_attack},
        n_flagged_pre_attack={self.n_flagged_pre_attack},
        n_not_flagged_pre_attack={self.n_not_flagged_pre_attack},
        n_flagged_post_attack={self.n_flagged_post_attack},
        n_not_flagged_post_attack={self.n_not_flagged_post_attack},
        n_flagged_post_attack_then_correct={self.n_flagged_post_attack_then_correct},
        n_flagged_post_attack_then_incorrect={self.n_flagged_post_attack_then_incorrect},
        n_not_flagged_post_attack_then_correct={self.n_not_flagged_post_attack_then_correct},
        n_not_flagged_post_attack_then_incorrect={self.n_not_flagged_post_attack_then_incorrect},
        n_post_attack_correct={self.n_post_attack_correct},
        n_post_attack_incorrect={self.n_post_attack_incorrect},
    )
    """

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
        original_pred_labels: Sequence[int],
        original_flag_values: Sequence[bool] | Sequence[None],
        attacked_labels: Sequence[int],
        attacked_pred_labels: Sequence[int],
        attacked_flag_values: Sequence[bool] | Sequence[None],
        attacked_pred_logits: Sequence[Sequence[float]],
    ) -> AttackResults:
        """Generates `AttackResults` from labels and model predictions.

        Args:
            original_labels: True labels of the original examples (pre-attack)
            original_pred_labels: Predictions of the model pre-attack.
            original_flag_values: Values of the filter applied pre-attack.
            attacked_labels: True labels of the attacked examples (post-attack)
            attacked_pred_labels: Predictions of the model post-attack
            attacked_flag_values: Values of the filter applied post-attack
            attacked_pred_logits: Logits of the model post-attack

        Returns:
            An `AttackResults` object
        """

        evaluation_outputs: list[EvaluationOutput] = (
            EvaluationOutput.get_list_from_labels_and_predictions(
                original_labels=original_labels,
                original_pred_labels=original_pred_labels,
                original_flag_values=original_flag_values,
                attacked_labels=attacked_labels,
                attacked_pred_labels=attacked_pred_labels,
                attacked_flag_values=attacked_flag_values,
                attacked_pred_logits=attacked_pred_logits,
            )
        )

        return cls(evaluation_outputs=evaluation_outputs)

    def _add_example_with_post_attack_only(
        self, evaluation_output: EvaluationOutput
    ) -> None:
        if evaluation_output.attacked_flag_value is True:
            self.n_flagged_post_attack += 1
            if (
                evaluation_output.attacked_label
                == evaluation_output.attacked_pred_label
            ):
                self.n_flagged_post_attack_then_correct += 1
        # This branch catches both the case where the defense did not flag the
        # example (False), and the case in which the defense was not run (None)
        else:
            if (
                evaluation_output.attacked_label
                == evaluation_output.attacked_pred_label
            ):
                self.n_not_flagged_post_attack_then_correct += 1

        # Compute the loss and the probability of the correct class.
        assert all(
            logit is not None for logit in evaluation_output.attacked_pred_logits
        )
        logprobs = torch.nn.functional.log_softmax(
            torch.tensor(evaluation_output.attacked_pred_logits), dim=0
        )
        loss = float(-logprobs[evaluation_output.attacked_label])
        correct_class_prob = float(
            torch.exp(logprobs[evaluation_output.attacked_label])
        )
        self.post_attack_losses.append(loss)
        self.post_attack_correct_class_probs.append(correct_class_prob)

    def _add_example(self, evaluation_output: EvaluationOutput) -> None:
        self.n_examples += 1
        if evaluation_output.original_flag_value is True:
            self.n_flagged_pre_attack += 1
        if evaluation_output.original_pred_label != evaluation_output.original_label:
            self.n_incorrect_pre_attack += 1
            # Since the model got this example wrong pre-attack,
            # there was no need to run the attack, and thus we
            # ignore the attack's output even if it is present.
            return

        # If we got here, the model gave the correct answer
        # pre-attack and the attack was run.
        self._add_example_with_post_attack_only(evaluation_output)

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
            "adversarial_eval/defense_true_positive_rate": div_maybe_nan(
                self.n_flagged_post_attack, self.n_correct_pre_attack
            ),
            "adversarial_eval/defense_true_negative_rate": div_maybe_nan(
                self.n_not_flagged_pre_attack, self.n_examples
            ),
            "adversarial_eval/defense_false_positive_rate": div_maybe_nan(
                self.n_flagged_pre_attack, self.n_examples
            ),
            "adversarial_eval/defense_false_negative_rate": div_maybe_nan(
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


def _make_flag_values_from_pipeline_output(
    preds_with_flag_values,
) -> list[None] | list[bool]:
    if all(flag is None for _, flag in preds_with_flag_values):
        return [None for _ in preds_with_flag_values]
    elif all(isinstance(flag, bool) for _, flag in preds_with_flag_values):
        return [flag for _, flag in preds_with_flag_values]
    else:
        raise ValueError("Flag values should be either all None or all bools")


def _make_prediction_logits_from_pipeline_output(
    preds_with_flag_values,
) -> list[list[float]]:
    # We use explicit loops below instead of list comprehension to be able to have type
    # asserts so that typechecker does not complain.
    prediction_logits = []
    assert isinstance(preds_with_flag_values, list)
    for pred, _ in preds_with_flag_values:  # type: ignore
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
    return prediction_logits


def _get_prediction_logits_and_maybe_flag_values(
    hf_pipeline: FilteredEvaluationPipeline,
    dataset: Dataset,
    batch_size: int,
    input_column: str = "text",
) -> tuple[list[list[float]], list[bool] | list[None]]:
    """Returns prediction logits, of shape (n_examples, n_classes),
    and flag values, of shape (n_examples,)."""

    # TODO(michal): consider dropping HF pipelines and instead use data loaders + direct
    # model calls. Pipelines complicate the code without a clear benefit.
    preds_with_flag_values = hf_pipeline(
        dataset[input_column],
        batch_size=batch_size,
        truncation=True,
        function_to_apply="none",
    )

    # TODO(niki): type check the following two functions
    prediction_logits = _make_prediction_logits_from_pipeline_output(
        preds_with_flag_values
    )
    flag_values = _make_flag_values_from_pipeline_output(preds_with_flag_values)

    return prediction_logits, flag_values


def _get_prediction_labels(
    prediction_logits: Sequence[Sequence[float]],
) -> list[int]:
    return [int(np.argmax(logits)) for logits in prediction_logits]


def get_prediction_logits_and_labels_and_maybe_flag_values(
    dataset: Dataset,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    input_column: str = "text",
) -> tuple[list[list[float]], list[int], list[bool] | list[None]]:
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
    pred_logits, pred_flag_values = _get_prediction_logits_and_maybe_flag_values(
        hf_pipeline=hf_pipeline,
        dataset=dataset,
        batch_size=batch_size,
        input_column=input_column,
    )
    pred_labels = _get_prediction_labels(pred_logits)

    return pred_logits, pred_labels, pred_flag_values


def _log_examples_to_wandb(
    original_texts: Sequence[str],
    original_labels: Sequence[int],
    original_pred_labels: Sequence[int],
    original_flag_values: Sequence[bool] | Sequence[None],
    attacked_texts: Sequence[str],
    attacked_labels: Sequence[int],
    attacked_pred_labels: Sequence[int],
    attacked_flag_values: Sequence[bool] | Sequence[None],
    attacked_pred_logits: Sequence[Sequence[float]],
    num_examples_to_log_detailed_info: int,
) -> None:

    table = wandb.Table(
        columns=[
            "original_text",
            "attacked_text",
            "original_label",
            "attacked_label",
            "original_pred",
            "attacked_pred",
            "original_filter",
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
            original_flag_values[i],
            attacked_flag_values[i],
            attacked_pred_logits[i],
        )

    wandb.log({"adversarial_eval/examples": table}, commit=False)


def compute_attack_results(
    attacked_dataset: RLLMDataset,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    batch_size: int,
    num_examples_to_log_detailed_info: Optional[int],
) -> AttackResults:
    """Performs an attack and reports its results."""

    assert attacked_dataset.ds is not None
    model.eval()

    _, original_pred_labels, original_flag_values = (
        get_prediction_logits_and_labels_and_maybe_flag_values(
            dataset=attacked_dataset.ds,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            input_column="text",
        )
    )
    attacked_pred_logits, attacked_pred_labels, attacked_flag_values = (
        get_prediction_logits_and_labels_and_maybe_flag_values(
            dataset=attacked_dataset.ds,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            input_column="attacked_text",
        )
    )

    results = AttackResults.from_labels_and_predictions(
        original_labels=attacked_dataset.ds["clf_label"],
        original_pred_labels=original_pred_labels,
        original_flag_values=original_flag_values,
        attacked_labels=attacked_dataset.ds["attacked_clf_label"],
        attacked_pred_labels=attacked_pred_labels,
        attacked_flag_values=attacked_flag_values,
        attacked_pred_logits=attacked_pred_logits,
    )

    if num_examples_to_log_detailed_info is not None and accelerator.is_main_process:
        _log_examples_to_wandb(
            original_texts=attacked_dataset.ds["text"],
            original_labels=attacked_dataset.ds["clf_label"],
            original_pred_labels=original_pred_labels,
            original_flag_values=original_flag_values,
            attacked_texts=attacked_dataset.ds["attacked_text"],
            attacked_labels=attacked_dataset.ds["attacked_clf_label"],
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
    dataset: RLLMDataset,
    attack: Attack,
    batch_size: int,
    num_examples_to_log_detailed_info: Optional[int],
) -> dict[str, float]:
    """Performs adversarial evaluation and logs the results."""

    # Sanity check in case of a distributed run (with accelerate): check if every
    # process has the same dataset.
    # TODO(michal): Look into datasets code and make sure the dataset creation is
    # deterministic given seeds; this is especially important when using accelerate.
    _assert_same_data_between_processes(accelerator, dataset.ds["text"])
    _assert_same_data_between_processes(accelerator, dataset.ds["clf_label"])

    model.eval()

    print("Doing adversarial evaluation...")

    attacked_dataset, info_dict = attack.get_attacked_dataset(dataset=dataset)

    attack_results = compute_attack_results(
        attacked_dataset=attacked_dataset,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        batch_size=batch_size,
        num_examples_to_log_detailed_info=num_examples_to_log_detailed_info,
    )

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
