"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import wandb
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TextClassificationPipeline,
)

from robust_llm.attacks.attack import Attack
from robust_llm.utils import div_maybe_nan


@dataclass
class AttackResults:
    """Results of an attack on a dataset."""

    n_successes: int  # Number of examples that were successfully attacked
    n_failures: int  # Number of examples that were not successfully attacked
    # Number of examples for which the model makes a mistake even before the attack,
    # so for the sake of the metrics we do not care about them; attack might have been
    # skipped for these cases.
    n_mistakes_before_attack: int

    @classmethod
    def from_labels_and_predictions(
        cls,
        labels: Sequence[int],
        original_preds: Optional[Sequence[int]],
        attacked_preds: Sequence[int],
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
        original_maybe_preds: Sequence[Optional[int]]
        if original_preds:
            original_maybe_preds = original_preds
        else:
            original_maybe_preds = [None] * len(labels)
        assert len(labels) == len(original_maybe_preds) == len(attacked_preds)

        n_successes, n_failures, n_mistakes_before_attack = 0, 0, 0
        for label, original_pred, attacked_pred in zip(
            labels, original_maybe_preds, attacked_preds
        ):
            # In this case, we expect that the attack should have been tried, and we
            # check if it was successful or not.
            if original_pred is None or label == original_pred:
                if label == attacked_pred:
                    n_failures += 1
                else:
                    n_successes += 1
            # Prediction was already incorrect, so we expect this example should have
            # been skipped and we do not consider it either a success or a failure.
            else:
                n_mistakes_before_attack += 1

        return cls(
            n_successes=n_successes,
            n_failures=n_failures,
            n_mistakes_before_attack=n_mistakes_before_attack,
        )

    @property
    def n_total(self) -> int:
        """Total number of examples used."""
        return self.n_successes + self.n_failures + self.n_mistakes_before_attack

    @property
    def n_attempted(self) -> int:
        """Total number of examples for which the attack should have been attempted."""
        return self.n_successes + self.n_failures

    def compute_adversarial_evaluation_metrics(self) -> Dict[str, float]:
        """Computes final metrics to report."""

        return {
            "adversarial_eval/pre_attack_accuracy": div_maybe_nan(
                self.n_attempted, self.n_total
            ),
            "adversarial_eval/post_attack_accuracy": div_maybe_nan(
                self.n_failures, self.n_total
            ),
            "adversarial_eval/attack_success_rate": div_maybe_nan(
                self.n_successes, self.n_attempted
            ),
            "adversarial_eval/n_total_examples_used": self.n_total,
        }


def compute_attack_results(
    dataset: Optional[Dataset],
    attacked_dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
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

    hf_pipeline = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, device=device, return_all_scores=False
    )

    original_preds = None
    if dataset is not None:
        original_preds = hf_pipeline(
            dataset["text"],
            batch_size=batch_size,
            padding="max_length",
            truncation=True,
        )
        assert original_preds is not None
        original_preds = [
            model.config.label2id[pred["label"]]  # type: ignore
            for pred in original_preds
        ]

    attacked_preds = hf_pipeline(
        attacked_dataset["text"],
        batch_size=batch_size,
        padding="max_length",
        truncation=True,
    )
    attacked_preds = [
        model.config.label2id[pred["label"]] for pred in attacked_preds  # type: ignore
    ]

    results = AttackResults.from_labels_and_predictions(
        labels=attacked_dataset["label"],
        original_preds=original_preds,
        attacked_preds=attacked_preds,
    )

    return results


def do_adversarial_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
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

    metrics = attack_results.compute_adversarial_evaluation_metrics()

    # TODO(GH#158): Refactor/unify logging.
    wandb.log(metrics, commit=False)
    print("Adversarial evaluation metrics:")
    print(metrics)

    return metrics
