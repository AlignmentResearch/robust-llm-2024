"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import wandb
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

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is supported by the pipeline.

        Args:
            supported_models (`List[str]` or `dict`):
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
    """

    n_filtered_out_pre_attack: int = 0
    n_mistakes_pre_attack: int = 0
    n_filtered_out_post_attack: int = 0
    n_failures: int = 0
    n_successes: int = 0

    @classmethod
    def from_labels_and_predictions(
        cls,
        labels: Sequence[int],
        original_preds: Optional[Sequence[int | None]],
        attacked_preds: Sequence[int | None],
    ) -> "AttackResults":
        """Generates `AttackResults` from labels and model predictions.

        Args:
            labels: true labels of the examples
            original_preds: optional predictions of the model pre-attack. Should
                be provided iff the attack uses input dataset
            attacked_preds: predictions of the model post-attack

        Returns:
            `AttackResults` object
        """
        results = cls()

        if original_preds is None:
            for label, attacked_pred in zip(labels, attacked_preds):
                results._add_example_with_post_attack_only(
                    label=label,
                    attacked_pred=attacked_pred,
                )
        else:
            for label, original_pred, attacked_pred in zip(
                labels, original_preds, attacked_preds
            ):
                results._add_example(
                    label=label,
                    original_pred=original_pred,
                    attacked_pred=attacked_pred,
                )

        return results

    def _add_example_with_post_attack_only(
        self,
        label: int,
        attacked_pred: int | None,
    ) -> None:
        # Filtered out after attack -> true positive.
        if attacked_pred is None:
            self.n_filtered_out_post_attack += 1
        # Correct prediction after attack -> failure.
        elif label == attacked_pred:
            self.n_failures += 1
        # Incorrect prediction after attack -> success.
        else:
            self.n_successes += 1

    def _add_example(
        self,
        label: int,
        original_pred: int | None,
        attacked_pred: int | None,
    ) -> None:
        # Example was filtered out pre-attack -> false positive.
        if original_pred is None:
            self.n_filtered_out_pre_attack += 1
        # Prediction was already incorrect, so we expect this example should have
        # been skipped and we do not consider it either a success or a failure.
        elif original_pred != label:
            self.n_mistakes_pre_attack += 1
        # Prediction was correct pre-attack. Now we need to check the post-attack.
        else:
            self._add_example_with_post_attack_only(
                label=label, attacked_pred=attacked_pred
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

    def compute_adversarial_evaluation_metrics(self) -> Dict[str, float]:
        """Computes final metrics to report."""

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
        }


def _get_predictions(
    hf_pipeline: FilteredEvaluationPipeline,
    dataset: Dataset,
    model: LanguageModel,
    batch_size: int,
) -> Sequence[int | None]:
    preds = hf_pipeline(
        dataset["text"],
        batch_size=batch_size,
        padding="max_length",
        truncation=True,
    )

    assert preds is not None
    preds = list(preds)
    num_preds = len(preds)
    # If in the line below is needed so that type checker does not complain.
    pred_labels = [pred.get("label") for pred in preds if isinstance(pred, dict)]
    preds = [
        model.config.label2id[label] if label is not None else None
        for label in pred_labels
    ]
    # Check if nothing was filtered out.
    assert len(preds) == num_preds

    return preds


def compute_attack_results(
    dataset: Optional[Dataset],
    attacked_dataset: Dataset,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device: Optional[str] = None,
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

    hf_pipeline = FilteredEvaluationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=False,
        framework="pt",
    )

    original_preds = None
    if dataset is not None:
        original_preds = _get_predictions(
            hf_pipeline=hf_pipeline,
            dataset=dataset,
            model=model,
            batch_size=batch_size,
        )

    attacked_preds = _get_predictions(
        hf_pipeline=hf_pipeline,
        dataset=attacked_dataset,
        model=model,
        batch_size=batch_size,
    )

    results = AttackResults.from_labels_and_predictions(
        labels=attacked_dataset["label"],
        original_preds=original_preds,
        attacked_preds=attacked_preds,
    )

    return results


def do_adversarial_evaluation(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Optional[Dataset],
    num_generated_examples: Optional[int],
    attack: Attack,
    batch_size: int,
) -> Dict[str, float]:
    """Performs adversarial evaluation and logs the results."""
    # Exactly one of these should be provided.
    assert (dataset is None) != (num_generated_examples is None)

    print("Doing adversarial evaluation...")

    attacked_dataset = attack.get_attacked_dataset(
        dataset=dataset, max_n_outputs=num_generated_examples
    )

    attack_results = compute_attack_results(
        dataset=dataset,
        attacked_dataset=attacked_dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    print(attack_results)

    metrics = attack_results.compute_adversarial_evaluation_metrics()

    # TODO(GH#158): Refactor/unify logging.
    wandb.log(metrics, commit=False)
    print("Adversarial evaluation metrics:")
    print(metrics)

    return metrics
