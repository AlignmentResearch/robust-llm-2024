from typing import Sequence

import numpy as np


def get_n_adv_tr_rounds() -> list[int]:
    # See https://docs.google.com/spreadsheets/d/1eifxKB_r9IRnVSqm10as-grVEtcOal9zE8B57iD6-Z0  # noqa: E501
    n_adv_tr_rounds = [953, 413, 163, 59, 21, 8, 6, 3, 1, 1]

    # Set a floor of 3 and a ceiling of 250 for adv tr rounds
    n_adv_tr_rounds = [max(5, min(249, x)) for x in n_adv_tr_rounds]

    # Increment all the rounds by 1 because we skip the first round
    # to avoid training on clean data only.
    n_adv_tr_rounds = [i + 1 for i in n_adv_tr_rounds]

    return n_adv_tr_rounds


# Now we want to turn the number of rounds into a list of lists.
# Each list will have the 10 first seeds, the 10 last seeds,
# and 10 seeds spread out evenly over the whole range.
def get_n_rounds_to_evaluate(n_rounds: int) -> list[int]:
    evaluation_rounds = list(
        sorted(
            list(
                set(
                    np.concatenate(
                        [
                            np.arange(11),
                            np.linspace(0, n_rounds, 10).astype(int),
                            np.arange(np.max((0, n_rounds - 6)).astype(int), n_rounds),
                        ]
                    )
                )
            )
        )
    )

    # Some of the models only trained for 5 rounds (as described
    # in the N_ADV_TR_ROUNDS list). But all of the lists in ROUNDS_FOR_EACH
    # at least go up to 10. So for those models which stop before 10,
    # we want to remove any rounds that are beyond the number of rounds
    # that the model was trained for. For example, if the model was trained
    # until round 6, then instead of looking like [0, 1, ..., 6, 7, 8, 9, 10],
    # the list should look like [0, 1, ..., 6].
    return evaluation_rounds[:n_rounds]


def get_all_n_rounds_to_evaluate() -> list[list[int]]:
    n_adv_tr_rounds = get_n_adv_tr_rounds()
    all_the_rounds = []
    for n_rounds in n_adv_tr_rounds:
        all_the_rounds.append(get_n_rounds_to_evaluate(n_rounds))
    return all_the_rounds


def get_start_median_and_end(my_sequence: Sequence) -> list:
    return [my_sequence[0], my_sequence[len(my_sequence) // 2], my_sequence[-1]]


def get_all_minimal_rounds_to_evaluate() -> list[list[int]]:
    # Same idea as above, but we only take the first, median, and last rounds.
    all_n_rounds = get_all_n_rounds_to_evaluate()
    minimal_rounds = []
    for n_rounds in all_n_rounds:
        minimal_rounds.append(get_start_median_and_end(n_rounds))
    return minimal_rounds
