import numpy as np

from robust_llm.evaluation import AttackResults
from robust_llm.evaluation_utils import DefendedAttackResults


def test_AttackResults():
    """Test the AttackResults class."""

    pre_attack_successes = [True, False, True, False, True]
    post_attack_successes = [True, True, False]
    attack_results = AttackResults(
        pre_attack_successes=pre_attack_successes,
        post_attack_successes=post_attack_successes,
    )
    assert attack_results.n_examples == 5

    assert attack_results.n_examples_pre_attack == 5
    assert attack_results.n_incorrect_pre_attack == 2
    assert attack_results.n_correct_pre_attack == 3

    assert attack_results.n_examples_post_attack == 3
    assert attack_results.n_correct_post_attack == 2
    assert attack_results.n_incorrect_post_attack == 1


def test_DefendedAttackResults():
    """Test the DefendedAttackResults class."""
    pre_attack_successes = [True, False, True, False, True]
    post_attack_successes = [True, True, False]
    attack_results = AttackResults(
        pre_attack_successes=pre_attack_successes,
        post_attack_successes=post_attack_successes,
    )
    pre_attack_flag_values = [True, True, False, False, True]
    post_attack_flag_values = [True, False, True]
    defended_attack_results = attack_results.with_defense_flags(
        pre_attack_flag_values=pre_attack_flag_values,
        post_attack_flag_values=post_attack_flag_values,
    )
    defended_attack_results_from_scratch = DefendedAttackResults(
        pre_attack_successes=pre_attack_successes,
        post_attack_successes=post_attack_successes,
        pre_attack_flag_values=pre_attack_flag_values,
        post_attack_flag_values=post_attack_flag_values,
    )
    assert defended_attack_results == defended_attack_results_from_scratch

    # Check that the AttackResults properties haven't broken.
    assert attack_results.n_examples == defended_attack_results.n_examples

    assert (
        attack_results.n_examples_pre_attack
        == defended_attack_results.n_examples_pre_attack
    )
    assert (
        attack_results.n_incorrect_pre_attack
        == defended_attack_results.n_incorrect_pre_attack
    )
    assert (
        attack_results.n_correct_pre_attack
        == defended_attack_results.n_correct_pre_attack
    )

    assert (
        attack_results.n_examples_post_attack
        == defended_attack_results.n_examples_post_attack
    )
    assert (
        attack_results.n_correct_post_attack
        == defended_attack_results.n_correct_post_attack
    )
    assert (
        attack_results.n_incorrect_post_attack
        == defended_attack_results.n_incorrect_post_attack
    )

    # Check the defense-related properties.
    assert defended_attack_results.n_flagged_pre_attack == 3
    assert defended_attack_results.n_not_flagged_pre_attack == 2

    assert defended_attack_results.n_flagged_post_attack == 2
    assert defended_attack_results.n_not_flagged_post_attack == 1

    assert defended_attack_results.n_flagged_post_attack_then_correct == 1
    assert defended_attack_results.n_not_flagged_post_attack_then_correct == 1


def test_compute_adversarial_evaluation_metrics():
    # Prepare AttackResults to match old test
    pre_attack_flag_values = [False, False, False, False, False, False, True]
    pre_attack_success = [True, True, True, True, True, False, False]

    post_attack_flag_values = [True, True, True, True, False]
    post_attack_success = [True, True, False, False, True]

    attack_results = DefendedAttackResults(
        pre_attack_successes=pre_attack_success,
        post_attack_successes=post_attack_success,
        pre_attack_flag_values=pre_attack_flag_values,
        post_attack_flag_values=post_attack_flag_values,
    )
    assert attack_results.n_examples == 7
    assert attack_results.n_incorrect_pre_attack == 2
    assert attack_results.n_flagged_pre_attack == 1
    # n attacked = 5
    # n not attacked = 2
    assert attack_results.n_flagged_post_attack == 4
    # n not flagged post attack = 1
    assert attack_results.n_flagged_post_attack_then_correct == 2
    assert attack_results.n_not_flagged_post_attack_then_correct == 1

    # Check metrics
    metrics = attack_results.compute_adversarial_evaluation_metrics()

    assert metrics["adversarial_eval/n_examples"] == 7
    assert np.isclose(metrics["adversarial_eval/pre_attack_flagging_rate"], 1 / 7)
    assert np.isclose(metrics["adversarial_eval/pre_attack_accuracy"], 5 / 7)
    assert np.isclose(metrics["adversarial_eval/post_attack_flagging_rate"], 4 / 5)
    assert np.isclose(
        metrics["adversarial_eval/post_attack_accuracy_including_original_mistakes"],
        3 / 7,
    )
    assert np.isclose(metrics["adversarial_eval/attack_success_rate"], 2 / 5)
    assert np.isclose(
        metrics["adversarial_eval/post_attack_accuracy_on_pre_attack_correct_examples"],
        3 / 5,
    )
    assert np.isclose(
        metrics["adversarial_eval/post_attack_accuracy_on_not_flagged_examples"], 1 / 1
    )
    assert np.isclose(metrics["adversarial_eval/defense_true_positive_rate"], 4 / 5)
    assert np.isclose(metrics["adversarial_eval/defense_true_negative_rate"], 6 / 7)
    assert np.isclose(metrics["adversarial_eval/defense_false_positive_rate"], 1 / 7)
    assert np.isclose(metrics["adversarial_eval/defense_false_negative_rate"], 1 / 5)

    assert np.isclose(
        metrics["adversarial_eval/post_attack_flagged_but_correct_rate"], 2 / 4
    )
    assert np.isclose(
        metrics["adversarial_eval/post_attack_not_flagged_but_incorrect_rate"], 0 / 1
    )
