import numpy as np


def int_geomspace(start, stop, num=50, endpoint=True) -> np.ndarray:
    """
    Return num unique integers spaced approximately evenly on a log scale.

    If initial geometric spacing produces too few unique integers after rounding,
    fills in with the smallest available integers in the range.

    Parameters
    ----------
    start : int
        The starting value of the sequence.
    stop : int
        The final value of the sequence.
    num : int, optional
        Number of unique samples to generate. Default is 50.
    endpoint : boolean, optional
        If true, stop is the last sample. Default is True.

    Returns
    -------
    ndarray
        Array of num unique integers spaced approximately geometrically.

    Examples
    --------
    >>> int_geomspace(1, 256, num=9)
    array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    >>> int_geomspace(1, 1000, num=4)
    array([1, 10, 100, 1000])
    """
    if not isinstance(start, (int, np.integer)) or not isinstance(
        stop, (int, np.integer)
    ):
        raise ValueError("start and stop must be integers")

    if num == 0:
        return np.array([], dtype=int)
    elif num < 2:
        raise ValueError("num must be at least 2")

    if start < 0 or stop < 0:
        raise ValueError("start and stop must be non-negative")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")

    # Generate initial points, handling start=0 carefully
    points = (
        np.geomspace(start + 1, stop + 1, num=num, endpoint=endpoint, dtype=int) - 1
    )
    points = np.unique(points)

    # If we have too few points, add the smallest available integers
    all_possible = np.arange(start, stop + 1, dtype=int)
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


def get_n_adv_tr_rounds(attack: str) -> list[int]:
    assert attack in ("rt", "gcg")
    max_adv_tr_rounds = 250 if attack == "rt" else 60
    n_adv_tr_rounds = [
        np.clip(x, 5, max_adv_tr_rounds) for x in [953, 413, 163, 59, 21, 8, 6, 3, 1, 1]
    ]
    n_adv_tr_rounds = [x + 1 for x in n_adv_tr_rounds]
    return n_adv_tr_rounds


def get_all_n_rounds_to_evaluate(
    attack: str,
    start_rounds: int = 0,
    middle_rounds: int = 10,
    end_rounds: int = 0,
) -> list[list[int]]:
    """Get the adversarial training rounds to evaluate for each model size.

    Args:
        attack: The attack to use. Either "rt" or "gcg".
        start_rounds: The number of rounds to evaluate at the start.
        middle_rounds: The number of rounds to evaluate in the middle.
        end_rounds: The number of rounds to evaluate at the end.

    Returns:
        [n_models, n_rounds] list of adversarial training rounds to evaluate.
        For pythia, n_models=10. The number of rounds to evaluate is bounded by
        the number of rounds that were trained and the sum of start_rounds,
        middle_rounds, and end_rounds.
    """
    n_adv_tr_rounds = get_n_adv_tr_rounds(attack)

    # If there are insufficient rounds available, take first from the start, then
    # the end, then the middle.
    all_start_rounds = [min(n_rounds, start_rounds) for n_rounds in n_adv_tr_rounds]
    all_end_rounds = [
        min(n_rounds - s_rounds, end_rounds)
        for s_rounds, n_rounds in zip(all_start_rounds, n_adv_tr_rounds, strict=True)
    ]
    all_middle_rounds = [
        min(n_rounds - s_rounds - e_rounds, middle_rounds)
        for s_rounds, e_rounds, n_rounds in zip(
            all_start_rounds, all_end_rounds, n_adv_tr_rounds
        )
    ]

    # Construct lists of rounds for concatenation, splitting up into steps for debugging
    all_start_rounds_list = [np.arange(s_rounds) for s_rounds in all_start_rounds]
    all_middle_rounds_list = [
        int_geomspace(
            start=s_rounds,
            stop=max(n_rounds - 1 - e_rounds, s_rounds),
            num=m_rounds,
        )
        for s_rounds, m_rounds, e_rounds, n_rounds in zip(
            all_start_rounds, all_middle_rounds, all_end_rounds, n_adv_tr_rounds
        )
    ]
    all_end_rounds_list = [
        np.arange(n_rounds - e_rounds, n_rounds)
        for n_rounds, e_rounds in zip(n_adv_tr_rounds, all_end_rounds)
    ]

    concatenated_rounds = [
        sorted(list(set(np.concatenate([start, middle, end]))))
        for start, middle, end in zip(
            all_start_rounds_list,
            all_middle_rounds_list,
            all_end_rounds_list,
            strict=True,
        )
    ]
    return concatenated_rounds
