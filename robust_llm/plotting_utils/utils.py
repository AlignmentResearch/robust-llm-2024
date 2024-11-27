import numpy as np
import pandas as pd
from scipy.stats import norm


def add_model_idx_inplace(
    df: pd.DataFrame,
    reference_col: str,
    exists_ok: bool = False,
) -> pd.DataFrame:
    """Add a model_idx column to a dataframe.

    Notes:
    - Modifies df in place.
    - This should only be used if the df already contains data for all the
    models. It should not be used if it'll later be merged with another df that
    contains data for other model sizes.

    Args:
        df:
           The dataframe to add the model_idx column to.
        reference_col:
            The column to use as a reference for the model_idx.
            If the reference_col (TODO: work out what to do here)
        exists_ok:
            If True, don't raise an error if model_idx already exists in the
            dataframe, instead recalculate it. If False, raise an error.

    Returns:
        The modified dataframe (which is a reference to the original df).

    """
    if "model_idx" in df and not exists_ok:
        raise ValueError("model_idx already in dataframe")

    df["model_idx"] = df[reference_col].rank(method="dense").astype(int) - 1
    return df


def merge_adv_and_train_data(
    adv_data: pd.DataFrame, train_data: pd.DataFrame
) -> pd.DataFrame:
    """Merge adversarial eval data with adversarial training data.

    This lets us have all the data in one df (e.g. ASRs and pretrain compute.
    """
    adv_data = drop_duplicates(
        adv_data, ["model_key", "seed_idx", "adv_training_round"], name="adv_data"
    )

    train_data = drop_duplicates(
        train_data,
        ["model_key", "seed_idx", "adv_training_round"],
        name="train_data",
    )

    adv_data = adv_data.merge(
        train_data,
        on=["model_key", "seed_idx", "adv_training_round"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_train"),
    )
    return adv_data


def drop_duplicates(
    df: pd.DataFrame, keys: list[str], name: str = "df", keep: str = "last"
) -> pd.DataFrame:
    dups = df.duplicated(subset=keys)
    if dups.any():
        print(
            f"\033[91mWARNING: Dropping {len(df[dups]) // 2} duplicates in "
            f"{name} with identical keys={keys}, keeping {keep}\033[0m"
        )
        df = df.drop_duplicates(subset=keys, keep=keep)  # type: ignore
    return df


def get_wilson_score_interval(
    df: pd.DataFrame, successes_col: str, trials_col: str, confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compute the Wilson score interval for a binomial proportion.

    See https://www.statisticshowto.com/wilson-ci/ for reference.
    """
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    # Compute proportions and intermediate values
    n = df[trials_col]
    p_hat = df[successes_col] / n
    q_hat = 1 - p_hat
    second_term = z**2 / (2 * n)
    third_term = z * np.sqrt(p_hat * q_hat / n + z**2 / (4 * n**2))
    denominator = 1 + z**2 / n

    # Compute bounds
    lower_bound = (p_hat + second_term - third_term) / denominator
    upper_bound = (p_hat + second_term + third_term) / denominator

    # Return the DataFrame with bounds as new columns
    result_df = df.copy()
    result_df["lower_bound"] = lower_bound
    result_df["upper_bound"] = upper_bound
    return result_df
