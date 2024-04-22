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
    LOSS_FAILURE = -np.log(PROBS_PRED0[0])  # attack failed
    LOSS_SUCCESS = -np.log(PROBS_PRED1[0])  # attack succeeded
    PROB_CORR_FAILURE = PROBS_PRED0[0]
    PROB_CORR_SUCCESS = PROBS_PRED1[0]

    data = [
        (0, 0, 1, 1, None, None, LOGITS_PRED1),  # mistake before attack
        (1, 1, 0, 1, None, None, LOGITS_PRED1),  # mistake before attack
        (0, 0, 0, 1, None, None, LOGITS_PRED1),  # attack succeeded
        (1, 1, 1, 0, None, None, LOGITS_PRED0),  # attack succeeded
        (0, 0, 0, 0, None, None, LOGITS_PRED0),  # attack failed
        (1, 1, 1, 1, None, None, LOGITS_PRED1),  # attack failed
        (0, 0, 0, 0, True, None, LOGITS_PRED0),  # defense false positive
        (0, 0, 0, 1, False, True, LOGITS_PRED1),  # defense true positive
        (1, 1, 1, 0, None, True, LOGITS_PRED0),  # defense true positive
        (0, 1, 0, 0, None, None, LOGITS_PRED0),  # attack succeeded, truth changed
        (0, 1, 0, 1, None, None, LOGITS_PRED1),  # attack failed, truth changed
    ]
    original_labels = [d[0] for d in data]
    attacked_labels = [d[1] for d in data]
    original_pred_labels = [d[2] for d in data]
    attacked_pred_labels = [d[3] for d in data]
    original_flag_values = [d[4] for d in data]
    attacked_flag_values = [d[5] for d in data]
    attacked_logits = [d[6] for d in data]

    # TODO(niki): the code this is testing assumes that either all
    # the examples are flagged or none are. So ideally this test
    # would be broken up into two tests: one in which flag values
    # are all True/False, and one in which they are all None.
    attack_results = AttackResults.from_labels_and_predictions(
        original_labels=original_labels,
        original_pred_labels=original_pred_labels,
        original_flag_values=original_flag_values,  # type: ignore
        attacked_labels=attacked_labels,
        attacked_pred_labels=attacked_pred_labels,
        attacked_flag_values=attacked_flag_values,  # type: ignore
        attacked_pred_logits=attacked_logits,
    )

    assert attack_results.n_examples == len(data)
    assert attack_results.n_flagged_pre_attack == 1
    assert attack_results.n_incorrect_pre_attack == 2
    assert attack_results.n_flagged_post_attack == 2
    assert np.allclose(
        attack_results.post_attack_losses,
        [
            LOSS_SUCCESS,
            LOSS_SUCCESS,
            LOSS_FAILURE,
            LOSS_FAILURE,
            LOSS_FAILURE,
            LOSS_SUCCESS,
            LOSS_SUCCESS,
            LOSS_SUCCESS,
            LOSS_FAILURE,
        ],
    )
    assert np.allclose(
        attack_results.post_attack_correct_class_probs,
        [
            PROB_CORR_SUCCESS,
            PROB_CORR_SUCCESS,
            PROB_CORR_FAILURE,
            PROB_CORR_FAILURE,
            PROB_CORR_FAILURE,
            PROB_CORR_SUCCESS,
            PROB_CORR_SUCCESS,
            PROB_CORR_SUCCESS,
            PROB_CORR_FAILURE,
        ],
    )


def test_compute_adversarial_evaluation_metrics():
    """Test the AttackResults.compute_adversarial_evaluation_metrics method."""

    attack_results = AttackResults(evaluation_outputs=[])

    # Manually set the attributes of the AttackResults object
    attack_results.n_examples = 7
    attack_results.n_incorrect_pre_attack = 2
    attack_results.n_flagged_pre_attack = 1
    # n attacked = 5
    # n not attacked = 2
    attack_results.n_flagged_post_attack = 4
    # n not flagged post attack = 1
    attack_results.n_flagged_post_attack_then_correct = 2
    attack_results.n_not_flagged_post_attack_then_correct = 1
    # n correct post attack = 3
    attack_results.post_attack_losses = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    attack_results.post_attack_correct_class_probs = [
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
    ]

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

    assert np.isclose(metrics["adversarial_eval/avg_post_attack_loss"], 0.3)
    assert np.isclose(
        metrics["adversarial_eval/avg_post_attack_correct_class_prob"], 0.6
    )
