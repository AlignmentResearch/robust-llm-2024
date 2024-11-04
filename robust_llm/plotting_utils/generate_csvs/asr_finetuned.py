"""Generate data for attack scaling plots (finetuned models)."""

import pandas as pd
from tqdm import tqdm

from robust_llm.plotting_utils.tools import (
    export_csv_and_metadata,
    get_cached_asr_logprob_data,
    prepare_adv_training_data,
)

GROUPS = [
    "ian_102a_gcg_pythia_harmless",
    "ian_103a_gcg_pythia_helpful",
    "ian_104a_rt_pythia_harmless",
    "ian_105a_rt_pythia_helpful",
    "ian_106_gcg_pythia_imdb",
    "ian_107_gcg_pythia_pm",
    "ian_108_gcg_pythia_wl",
    "ian_109_gcg_pythia_spam",
    "ian_110_rt_pythia_imdb",
    "ian_111_rt_pythia_pm",
    "ian_112_rt_pythia_wl",
    "ian_113_rt_pythia_spam",
]


def save_asr_for_group(
    df: pd.DataFrame,
    group_name: str,
):
    attack = group_name.split("_")[2]
    dataset = group_name.split("_")[-1]

    flop_data = prepare_adv_training_data(
        (group_name,),
        summary_keys=[
            "experiment_yaml.model.name_or_path",
            "experiment_yaml.dataset.n_val",
            "model_size",
            "experiment_yaml.model.revision",
        ],
        metrics=[
            "flops_per_iteration",
        ],
    )

    df = df.merge(
        flop_data,
        on=["model_idx", "seed_idx"],
        how="left",
        validate="m:1",
        suffixes=("", "_flop"),
    )

    export_csv_and_metadata(
        df,
        save_asr_for_group.__name__,
        {
            "group_name": group_name,
        },
        "asr",
        attack,
        dataset,
        "finetuned",
    )


def main():
    for group in tqdm(GROUPS):
        df = get_cached_asr_logprob_data(group)
        if df.empty:
            print(f"Group {group} does not exist")
            continue
        save_asr_for_group(df, group)


if __name__ == "__main__":
    main()
