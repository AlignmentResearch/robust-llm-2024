"""Generate data for attack scaling plots (finetuned models)."""

import pandas as pd
from tqdm import tqdm

from robust_llm.plotting_utils.tools import (
    export_csv_and_metadata,
    get_attack_from_name,
    get_cached_asr_logprob_data,
    get_dataset_from_name,
    get_family_from_name,
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
    "oskar_025a_gcg_eval_qwen25_ft_harmless",
    "oskar_025b_gcg_eval_qwen25_ft_spam",
    "tom_021b_beast_eval_qwen25_ft_harmless",
    "tom_022b_beast_eval_qwen25_ft_spam",
    "tom_019b_beast_eval_pythia_ft_harmless",
]


def save_asr_for_group(
    df: pd.DataFrame,
    group_name: str,
):
    family = get_family_from_name(group_name)
    attack = get_attack_from_name(group_name)
    dataset = get_dataset_from_name(group_name)

    export_csv_and_metadata(
        df,
        save_asr_for_group.__name__,
        {
            "group_name": group_name,
            "model": family,
            "attack": attack,
            "dataset": dataset,
        },
        "asr",
        family,
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
