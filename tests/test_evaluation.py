import numpy as np

from robust_llm.evaluation import AttackResults


def test_attack_results_from_labels_and_predictions():
    """Test the AttackResults.from_labels_and_predictions method."""

    # (label, original prediction, attacked prediction)
    data = [
        (0, 0, 1),  # success
        (1, 0, 0),  # mistake before attack
        (0, 0, 0),  # failure
        (1, 1, 1),  # failure
        (1, 1, 0),  # success
        (0, 1, 1),  # mistake before attack
        (1, 1, 1),  # failure
        (0, 0, 0),  # failure
        (1, 0, 1),  # mistake before attack
    ]
    labels = []
    original_preds = []
    attacked_preds = []
    for label, original_pred, attacked_pred in data:
        labels.append(label)
        original_preds.append(original_pred)
        attacked_preds.append(attacked_pred)

    attack_results = AttackResults.from_labels_and_predictions(
        labels, original_preds, attacked_preds
    )
    assert attack_results == AttackResults(
        n_successes=2, n_failures=4, n_mistakes_before_attack=3
    )


def test_compute_adversarial_evaluation_metrics():
    """Test the AttackResults.compute_adversarial_evaluation_metrics method."""

    attack_results = AttackResults(
        n_successes=2, n_failures=4, n_mistakes_before_attack=3
    )
    metrics = attack_results.compute_adversarial_evaluation_metrics()

    assert np.isclose(metrics["adversarial_eval/pre_attack_accuracy"], 6 / 9)
    assert np.isclose(metrics["adversarial_eval/post_attack_accuracy"], 4 / 9)
    assert np.isclose(metrics["adversarial_eval/attack_success_rate"], 2 / 6)
    assert metrics["adversarial_eval/n_total_examples_used"] == 9
