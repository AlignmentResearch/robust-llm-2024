from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from accelerate import Accelerator
from accelerate.utils import gather_object

from robust_llm.utils import auto_repr, div_maybe_nan


@dataclass(frozen=True)
@auto_repr
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

    Please note that the properties of this class are not disjoint. For example,
    a datapoint can be both flagged post-attack and correctly classified post-attack.
    """

    pre_attack_successes: Sequence[bool]
    post_attack_successes: Sequence[bool]

    def __post_init__(self):
        # Because we only attack examples that the model got correect before the attack.
        assert sum(self.pre_attack_successes) == len(self.post_attack_successes)

    @cached_property
    def n_examples(self) -> int:
        """Total number of datapoints in the original dataset.

        Note that this is equal to n_pre_attack_examples because
        we run on all examples in the original dataset.
        """
        return len(self.pre_attack_successes)

    @cached_property
    def n_examples_pre_attack(self) -> int:
        """Total number of datapoints in the pre-attack dataset."""
        return len(self.pre_attack_successes)

    @cached_property
    def n_examples_post_attack(self) -> int:
        """Total number of datapoints in the post-attack dataset.

        Note that this is equal to n_correct_pre_attack and can be less than
        n_examples_pre_attack.
        """
        return len(self.post_attack_successes)

    @cached_property
    def n_correct_pre_attack(self) -> int:
        """Number of examples the model gets correct before the attack."""
        return sum(self.pre_attack_successes)

    @cached_property
    def n_incorrect_pre_attack(self) -> int:
        """Number of examples the model gets incorrect before the attack."""
        return self.n_examples_pre_attack - self.n_correct_pre_attack

    @cached_property
    def n_correct_post_attack(self) -> int:
        """Number of examples the model gets correct after the attack."""
        return sum(self.post_attack_successes)

    @cached_property
    def n_incorrect_post_attack(self) -> int:
        """Number of examples the model gets incorrect after the attack."""
        return self.n_examples_post_attack - self.n_correct_post_attack

    def compute_adversarial_evaluation_metrics(self) -> dict[str, Any]:
        return {
            "adversarial_eval/n_examples": self.n_examples,
            "adversarial_eval/n_correct_pre_attack": self.n_correct_pre_attack,
            "adversarial_eval/n_incorrect_pre_attack": self.n_incorrect_pre_attack,
            "adversarial_eval/pre_attack_accuracy": div_maybe_nan(
                self.n_correct_pre_attack, self.n_examples_pre_attack
            ),
            "adversarial_eval/n_correct_post_attack": self.n_correct_post_attack,
            "adversarial_eval/n_incorrect_post_attack": self.n_incorrect_post_attack,
            "adversarial_eval/attack_success_rate": div_maybe_nan(
                self.n_incorrect_post_attack, self.n_examples_post_attack
            ),
            "adversarial_eval/post_attack_accuracy_on_pre_attack_correct_examples": (
                div_maybe_nan(self.n_correct_post_attack, self.n_correct_pre_attack)
            ),
        }

    def with_defense_flags(
        self,
        pre_attack_flag_values: Sequence[bool],
        post_attack_flag_values: Sequence[bool],
    ) -> DefendedAttackResults:
        return DefendedAttackResults(
            pre_attack_successes=self.pre_attack_successes,
            post_attack_successes=self.post_attack_successes,
            pre_attack_flag_values=pre_attack_flag_values,
            post_attack_flag_values=post_attack_flag_values,
        )


@dataclass(frozen=True)
@auto_repr
class DefendedAttackResults(AttackResults):
    pre_attack_flag_values: Sequence[bool]
    post_attack_flag_values: Sequence[bool]

    @cached_property
    def n_flagged_pre_attack(self) -> int:
        """Number of examples from the original dataset flagged by the defense.

        This is a "false positive" for the defense: none of the original
        dataset has been attacked.
        """
        return sum(self.pre_attack_flag_values)

    @cached_property
    def n_not_flagged_pre_attack(self) -> int:
        """Number of examples from the original dataset not flagged by the defense.

        This is a "true negative" for the defense: none of the original dataset
        has been attacked.
        """
        return self.n_examples_pre_attack - self.n_flagged_pre_attack

    @cached_property
    def n_flagged_post_attack(self) -> int:
        """Number of examples from the attacked dataset flagged by the defense.

        This is a "true positive" for the defense: all of the attacked dataset
        has been attacked.
        """
        return sum(self.post_attack_flag_values)

    @cached_property
    def n_not_flagged_post_attack(self) -> int:
        """Number of examples from the attacked dataset not flagged by the defense.

        This is a "false negative" for the defense: all of the attacked dataset
        has been attacked.
        """
        return self.n_examples_post_attack - self.n_flagged_post_attack

    @cached_property
    def n_flagged_post_attack_then_correct(self) -> int:
        """Number of examples flagged post-attack but the model got correct."""
        flags = self.post_attack_flag_values
        succs = self.post_attack_successes
        flag_and_correct = [flag and succ for flag, succ in zip(flags, succs)]
        return sum(flag_and_correct)

    @cached_property
    def n_flagged_post_attack_then_incorrect(self) -> int:
        """Number of examples flagged post-attack and the model got incorrect."""
        return self.n_flagged_post_attack - self.n_flagged_post_attack_then_correct

    @cached_property
    def n_not_flagged_post_attack_then_correct(self) -> int:
        """Number of examples not flagged post-attack and the model got correct."""
        flags = self.post_attack_flag_values
        succs = self.post_attack_successes
        not_flag_and_correct = [(not flag) and succ for flag, succ in zip(flags, succs)]
        return sum(not_flag_and_correct)

    @cached_property
    def n_not_flagged_post_attack_then_incorrect(self) -> int:
        """Number of examples not flagged post-attack but the model got incorrect."""
        return (
            self.n_not_flagged_post_attack - self.n_not_flagged_post_attack_then_correct
        )

    def compute_adversarial_evaluation_metrics(self) -> dict[str, Any]:
        """Computes final metrics to report."""
        metrics = super().compute_adversarial_evaluation_metrics()
        return metrics | {
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
            # Computed metrics
            "adversarial_eval/pre_attack_flagging_rate": div_maybe_nan(
                self.n_flagged_pre_attack, self.n_examples_pre_attack
            ),
            "adversarial_eval/post_attack_flagging_rate": div_maybe_nan(
                self.n_flagged_post_attack,
                self.n_examples_post_attack,
            ),
            "adversarial_eval/post_attack_accuracy_including_original_mistakes": (
                div_maybe_nan(self.n_correct_post_attack, self.n_examples)
            ),
            "adversarial_eval/post_attack_accuracy_on_not_flagged_examples": (
                div_maybe_nan(
                    self.n_not_flagged_post_attack_then_correct,
                    self.n_not_flagged_post_attack,
                )
            ),
            "adversarial_eval/attack_post_defense_success_rate": div_maybe_nan(
                self.n_not_flagged_post_attack_then_incorrect,
                self.n_examples_post_attack,
            ),
            "adversarial_eval/post_attack_robustness_rate": div_maybe_nan(
                self.n_flagged_post_attack
                + self.n_not_flagged_post_attack_then_correct,
                self.n_examples_post_attack,
            ),
            "adversarial_eval/defense_true_positive_rate": div_maybe_nan(
                self.n_flagged_post_attack, self.n_examples_post_attack
            ),
            "adversarial_eval/defense_true_negative_rate": div_maybe_nan(
                self.n_not_flagged_pre_attack, self.n_examples_pre_attack
            ),
            "adversarial_eval/defense_false_positive_rate": div_maybe_nan(
                self.n_flagged_pre_attack, self.n_examples_pre_attack
            ),
            "adversarial_eval/defense_false_negative_rate": div_maybe_nan(
                self.n_not_flagged_post_attack, self.n_examples_post_attack
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
        }


def assert_same_data_between_processes(
    accelerator: Accelerator, data: Sequence[Any]
) -> None:
    length = len(data)
    # We use 'gather_object' rather than 'gather_for_metrics' because we want to see
    # all the data gathered, especially repeats. (In theory 'gather_for_metrics'
    # should also work here, but we were having issues with flaky tests on CircleCI.)
    data_gathered = gather_object(data)
    for i in range(accelerator.num_processes):
        start = i * length
        end = (i + 1) * length
        assert data_gathered[start:end] == data, (
            f"Data from process {i} does not match original.\n"
            f"Original (len {length}): {data}\n"
            f"Process {i} ({start=}, {end=}): {data_gathered[start:end]}\n"
        )
