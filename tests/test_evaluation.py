import numpy as np

from robust_llm.evaluation import AttackResults


def test_attack_results_from_labels_and_predictions():
    """Test the AttackResults.from_labels_and_predictions method."""

    # (original_label, attacked_label, original prediction, attacked prediction)
    data = [
        (0, 0, 0, 1),  # success
        (1, 1, 0, 0),  # mistake before attack
        (0, 0, 0, 0),  # failure
        (1, 1, 1, 1),  # failure
        (1, 1, 1, 0),  # success
        (0, 0, 1, 1),  # mistake before attack
        (1, 1, 1, 1),  # failure
        (0, 0, 0, 0),  # failure
        (1, 1, 0, 1),  # mistake before attack
        (0, 0, None, 0),  # false positive
        (0, 0, 0, None),  # true positive
        (1, 1, 1, None),  # true positive
        (0, 1, 0, 0),  # success, with ground truth changed
        (0, 1, 0, 1),  # failure, with ground truth changed
    ]
    original_labels = []
    attacked_labels = []
    original_preds = []
    attacked_preds = []
    for original_label, attacked_label, original_pred, attacked_pred in data:
        original_labels.append(original_label)
        attacked_labels.append(attacked_label)
        original_preds.append(original_pred)
        attacked_preds.append(attacked_pred)

    attack_results = AttackResults.from_labels_and_predictions(
        original_labels, attacked_labels, original_preds, attacked_preds
    )
    assert attack_results == AttackResults(
        n_successes=3,
        n_failures=5,
        n_mistakes_pre_attack=3,
        n_filtered_out_post_attack=2,
        n_filtered_out_pre_attack=1,
    )


def test_compute_adversarial_evaluation_metrics():
    """Test the AttackResults.compute_adversarial_evaluation_metrics method."""

    attack_results = AttackResults(
        n_successes=2,
        n_failures=4,
        n_mistakes_pre_attack=3,
        n_filtered_out_post_attack=2,
        n_filtered_out_pre_attack=1,
    )
    metrics = attack_results.compute_adversarial_evaluation_metrics()

    assert np.isclose(metrics["adversarial_eval/pre_attack_accuracy"], 8 / 12)
    assert np.isclose(metrics["adversarial_eval/post_attack_accuracy"], 4 / 9)
    assert np.isclose(metrics["adversarial_eval/attack_success_rate"], 2 / 8)
    assert np.isclose(metrics["adversarial_eval/filtering_false_positive_rate"], 1 / 12)
    assert np.isclose(metrics["adversarial_eval/filtering_true_positive_rate"], 2 / 8)
    assert metrics["adversarial_eval/n_total_examples_used"] == 12
