from pathlib import Path

import pandas as pd

from robust_llm.file_utils import compute_repo_path

root = Path(compute_repo_path())


def remove_fake_adv_evals(data: pd.DataFrame) -> pd.DataFrame:
    """Remove fake adversarial evals data.

    niki-170-adv-tr-gcg-spam-small-0038 saved a model with
    revision adv-training-round-0 only (so no adversarial training)
    and we evaluated it in https://wandb.ai/farai/robust-llm/runs/w6insut2/overview
    but recorded no flops data. It's annoying so we remove such cases.
    """
    if data.adv_training_round.eq(0).all():
        # No action needed for finetuned evals
        return data
    data["max_adv_training_round"] = data.groupby(["model_idx", "seed_idx"])[
        "adv_training_round"
    ].transform("max")
    if data.max_adv_training_round.eq(0).any():
        print(
            f"\033[91mWARNING: Dropping {data.max_adv_training_round.eq(0).sum()} "
            f"evals which are not adversarial\033[0m"
        )
        data = data.loc[~data.max_adv_training_round.eq(0)]
    return data


if __name__ == "__main__":
    for path in (root / "cache_csvs/evaluation").iterdir():
        data = pd.read_csv(path)
        orig_len = len(data)
        data = remove_fake_adv_evals(data)
        if len(data) != orig_len:
            print(f"Removed {orig_len - len(data)} fake adversarial evals in {path}")
            data.to_csv(path, index=False)
