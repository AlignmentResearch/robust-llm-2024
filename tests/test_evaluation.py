import numpy as np

from robust_llm.evaluation import AttackResults


def test_attack_results_from_labels_and_predictions():
    """Test the AttackResults.from_labels_and_predictions method."""

    # (original_label, attacked_label, original prediction, attacked prediction,
    #  attacked logits)

    PROBS_PRED0 = [0.8, 0.2]
    PROBS_PRED1 = [0.2, 0.8]
    LOGITS_PRED0 = np.log(PROBS_PRED0).tolist()
    LOGITS_PRED1 = np.log(PROBS_PRED1).tolist()
    LOSS_FAILURE = -np.log(PROBS_PRED0[0])
    LOSS_SUCCESS = -np.log(PROBS_PRED1[0])
    PROB_CORR_FAILURE = PROBS_PRED0[0]
    PROB_CORR_SUCCESS = PROBS_PRED1[0]

    data = [
        (0, 0, 0, 1, None, LOGITS_PRED1),  # success
        (1, 1, 0, 0, None, LOGITS_PRED0),  # mistake before attack
        (0, 0, 0, 0, None, LOGITS_PRED0),  # failure
        (1, 1, 1, 1, None, LOGITS_PRED1),  # failure
        (1, 1, 1, 0, None, LOGITS_PRED0),  # success
        (0, 0, 1, 1, None, LOGITS_PRED1),  # mistake before attack
        (1, 1, 1, 1, None, LOGITS_PRED1),  # failure
        (0, 0, 0, 0, None, LOGITS_PRED0),  # failure
        (1, 1, 0, 1, None, LOGITS_PRED1),  # mistake before attack
        (0, 0, 0, 0, True, LOGITS_PRED0),  # false positive
        (0, 0, 0, 1, True, LOGITS_PRED1),  # true positive
        (1, 1, 1, 0, True, LOGITS_PRED0),  # true positive
        (0, 1, 0, 0, None, LOGITS_PRED0),  # success, with ground truth changed
        (0, 1, 0, 1, None, LOGITS_PRED1),  # failure, with ground truth changed
    ]
    original_labels = [d[0] for d in data]
    attacked_labels = [d[1] for d in data]
    original_preds = [d[2] for d in data]
    attacked_preds = [d[3] for d in data]
    attacked_filter_values = [d[4] for d in data]
    attacked_logits = [d[5] for d in data]

    attack_results = AttackResults.from_labels_and_predictions(
        original_labels,
        attacked_labels,
        original_preds,
        attacked_preds,
        attacked_filter_values,
        attacked_logits,
    )

    assert attack_results.n_filtered_out_pre_attack == 0
    assert attack_results.n_mistakes_pre_attack == 3
    assert attack_results.n_filtered_out_post_attack == 3
    assert attack_results.n_failures == 5
    assert attack_results.n_successes == 3
    assert np.allclose(
        attack_results.post_attack_losses,
        [
            LOSS_SUCCESS,
            LOSS_FAILURE,
            LOSS_FAILURE,
            LOSS_SUCCESS,
            LOSS_FAILURE,
            LOSS_FAILURE,
            LOSS_SUCCESS,
            LOSS_FAILURE,
        ],
    )
    assert np.allclose(
        attack_results.post_attack_correct_class_probs,
        [
            PROB_CORR_SUCCESS,
            PROB_CORR_FAILURE,
            PROB_CORR_FAILURE,
            PROB_CORR_SUCCESS,
            PROB_CORR_FAILURE,
            PROB_CORR_FAILURE,
            PROB_CORR_SUCCESS,
            PROB_CORR_FAILURE,
        ],
    )


def test_compute_adversarial_evaluation_metrics():
    """Test the AttackResults.compute_adversarial_evaluation_metrics method."""

    attack_results = AttackResults(
        n_successes=2,
        n_failures=4,
        n_mistakes_pre_attack=3,
        n_filtered_out_post_attack=2,
        n_filtered_out_pre_attack=1,
        post_attack_losses=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        post_attack_correct_class_probs=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    )
    metrics = attack_results.compute_adversarial_evaluation_metrics()

    assert np.isclose(metrics["adversarial_eval/pre_attack_accuracy"], 8 / 12)
    assert np.isclose(metrics["adversarial_eval/post_attack_accuracy"], 4 / 9)
    assert np.isclose(metrics["adversarial_eval/attack_success_rate"], 2 / 8)
    assert np.isclose(metrics["adversarial_eval/filtering_false_positive_rate"], 1 / 12)
    assert np.isclose(metrics["adversarial_eval/filtering_true_positive_rate"], 2 / 8)
    assert np.isclose(metrics["adversarial_eval/avg_post_attack_loss"], 0.35)
    assert np.isclose(
        metrics["adversarial_eval/avg_post_attack_correct_class_prob"], 0.55
    )
    assert metrics["adversarial_eval/n_total_examples_used"] == 12
