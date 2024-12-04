from pathlib import Path

import pandas as pd

from robust_llm.file_utils import compute_repo_path

root = Path(compute_repo_path())


def match_ft_seed_to_adv(model_key: str):
    """Match the fine-tuning seed to the adversarial training seed."""
    # model_key='clf_imdb_pythia-31m_s-0_adv_tr_gcg_t-4'
    adv_seed = model_key.split("_t-")[-1]
    ft_seed = model_key.split("_s-")[1].split("_")[0]
    return model_key.replace(f"_s-{ft_seed}", f"_s-{adv_seed}")


def hack_together_new_evals_old_flops(train_data):
    """Manually edit old flops data to match new evals data.

    Some runs, e.g. https://wandb.ai/farai/robust-llm/runs/7c3lhlpe
    do not have usable flops data. We can manually edit the old flops data
    so that it matches the new evals data. The problem is that we logged
    the flops after each batch, did not log the adversarial training round
    at all, and on top of that we resumed from checkpoint so the flops data is
    spread across wandb runs. The FLOPs should be the same as the old runs
    as it's just a different seed so we use those instead.
    """
    matched_seeds_train_data = train_data.copy()
    matched_seeds_train_data.model_key = matched_seeds_train_data.model_key.apply(
        match_ft_seed_to_adv
    )
    matched_seeds_train_data.training_force_name_to_save = (
        matched_seeds_train_data.model_key
    )
    return pd.concat([train_data, matched_seeds_train_data], ignore_index=True)


GROUPS = [
    "niki_152a_adv_tr_gcg_imdb_small",
    "niki_152_adv_tr_gcg_imdb_small",
]


if __name__ == "__main__":
    for group in GROUPS:
        path = root / "cache_csvs" / "training" / f"{group}.csv"
        data = pd.read_csv(path)
        orig_len = len(data)
        data = hack_together_new_evals_old_flops(data)
        print(f"Added {len(data) - orig_len} new FLOPs rows to {path}")
        data.to_csv(path, index=False)
