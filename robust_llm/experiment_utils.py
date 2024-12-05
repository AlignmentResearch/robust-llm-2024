import numpy as np

# These Pythia rounds are the result of _get_rounds_to_evaluate_pythia(),
# but we copy them in full here for transparency.
PYTHIA_GCG_EVAL_ROUNDS = {
    "14m": [1, 2, 3, 4, 6, 9, 15, 24, 38, 60],
    "31m": [1, 2, 3, 4, 6, 9, 15, 24, 38, 60],
    "70m": [1, 2, 3, 4, 6, 9, 15, 24, 38, 60],
    "160m": [1, 2, 3, 4, 6, 9, 15, 23, 37, 59],
    "410m": [1, 2, 3, 4, 5, 6, 7, 10, 14, 21],
    "1b": [1, 2, 3, 4, 5, 6, 7, 8],
    "1.4b": [1, 2, 3, 4, 5, 6],
    "2.8b": [1, 2, 3, 4, 5],
    "6.9b": [1, 2, 3, 4, 5],
    "12b": [1, 2, 3, 4, 5],
}

PYTHIA_RT_EVAL_ROUNDS = {
    "14m": [1, 2, 3, 6, 11, 21, 39, 73, 135, 250],
    "31m": [1, 2, 3, 6, 11, 21, 39, 73, 135, 250],
    "70m": [1, 2, 3, 5, 9, 16, 29, 52, 92, 163],
    "160m": [1, 2, 3, 4, 6, 9, 15, 23, 37, 59],
    "410m": [1, 2, 3, 4, 5, 6, 7, 10, 14, 21],
    "1b": [1, 2, 3, 4, 5, 6, 7, 8],
    "1.4b": [1, 2, 3, 4, 5, 6],
    "2.8b": [1, 2, 3, 4, 5],
    "6.9b": [1, 2, 3, 4, 5],
    "12b": [1, 2, 3, 4, 5],
}

QWEN_ROUNDS = {
    "0.5B": 22,
    "1.5B": 10,
    "3B": 10,
    "7B": 10,
    "14B": 10,
}

QWEN_EVAL_ROUNDS = {
    # >>> np.geomspace(1, 22, 10, dtype=int)
    # array([ 1,  1,  1,  2,  3,  5,  7, 11, 15, 22])
    "0.5B": [1, 2, 3, 4, 5, 6, 7, 11, 15, 22],
    "1.5B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "3B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "7B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "14B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


def get_n_adv_tr_rounds(attack: str) -> list[int]:
    assert attack in ("rt", "gcg")
    max_adv_tr_rounds = 250 if attack == "rt" else 60
    n_adv_tr_rounds = [
        np.clip(x, 5, max_adv_tr_rounds) for x in [953, 413, 163, 59, 21, 8, 6, 3, 1, 1]
    ]
    return n_adv_tr_rounds


def _pad_rounds(points: np.ndarray, stop: int, num: int) -> np.ndarray:
    # If we have too few points, add the smallest available integers
    all_possible = np.arange(1, stop + 1, dtype=int)
    if len(points) < num:
        # Get all possible integers in the range
        # Find integers not already in our sequence
        available = np.setdiff1d(all_possible, points)

        # Add the smallest available integers until we have enough points
        num_to_add = min(num - len(points), len(available))
        if len(available) > 0:
            additional = np.sort(available[:num_to_add])
            points = np.sort(np.concatenate([points, additional]))
    if len(points) != min(num, len(all_possible)):
        raise ValueError("Could not generate requested number of unique integers")
    return points


def _get_rounds_to_evaluate_pythia(
    attack: str,
    num_rounds_to_eval: int = 10,
) -> list[list[int]]:
    """Get the adversarial training rounds to evaluate for each Pythia model size.

    Prefer using get_rounds_to_evaluate() instead.

    Args:
        attack: The attack to use. Either "rt" or "gcg".
        num_rounds_to_eval: The number of rounds to evaluate.

    Returns:
        [n_models, n_rounds] list of adversarial training rounds to evaluate.
        For pythia, n_models=10. The number of rounds to evaluate is bounded by
        the number of rounds that were trained and the sum of start_rounds,
        middle_rounds, and end_rounds.
    """
    n_adv_tr_rounds = get_n_adv_tr_rounds(attack)
    return [
        list(
            _pad_rounds(
                np.unique(
                    np.geomspace(
                        start=1,
                        stop=n_rounds,
                        num=num_rounds_to_eval,
                        dtype=int,
                    )
                ),
                stop=n_rounds,
                num=num_rounds_to_eval,
            )
        )
        for n_rounds in n_adv_tr_rounds
    ]


def get_rounds_to_evaluate(
    model_family: str,
    training_attack: str,
) -> dict[str, list[int]]:
    """Get the adversarial training rounds to evaluate for each model size.

    Returns:
        A dictionary mapping model size to the adversarial training rounds to
        evaluate.
    """
    if model_family == "pythia":
        match training_attack:
            case "rt":
                return PYTHIA_RT_EVAL_ROUNDS
            case "gcg":
                return PYTHIA_GCG_EVAL_ROUNDS
            case _:
                raise ValueError(f"Invalid attack: {training_attack}")
    if "qwen" in model_family.lower():
        return QWEN_EVAL_ROUNDS
    raise ValueError(f"Invalid model family: {model_family}")
