import json
import os

import pandas as pd
from datasets.utils.logging import disable_progress_bar
from tqdm import tqdm

from robust_llm.plotting_utils.utils import add_model_idx_inplace
from robust_llm.wandb_utils.wandb_api_tools import get_runs_for_group


def maybe_download_asr_data(group: str) -> None:
    """Download ASR data from runs which stored an asr_per_iteration wandb table.

    We want to standardize many runs with different storage methods to be convenient
    CSVs in the repo. This function downloads the runs where we save a wandb table
    (i.e. artifact) called asr_per_iteration and standardizes them to a CSV.
    """
    asr_path = f"cache_csvs/asr_{group}.csv"
    if os.path.exists(asr_path):
        print(f"Skipping {asr_path}")
        return
    runs = get_runs_for_group(group)
    dfs = []
    for run in runs:
        if run.summary.get("really_finished") is None:
            continue
        wandb_run = run.to_wandb()
        asr_tables = [a for a in wandb_run.logged_artifacts() if "asr" in a.name]
        if not asr_tables:
            print(f"No ASR tables found for {wandb_run.name}")
            continue
        if len(asr_tables) > 1:
            print(f"Multiple ASR tables found for {wandb_run.name}")
            continue
        asr_table = asr_tables[0]
        table_dir = asr_table.download()
        table_path = os.path.join(table_dir, os.listdir(table_dir)[0])
        with open(table_path) as file:
            json_dict = json.load(file)
        df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
        df = df.T.reset_index()
        df.columns = ["iteration", "asr"]
        df.iteration = df.iteration.str.replace("asr@", "").astype(int)
        df["model_size"] = run.summary["model_size"]
        df["model_name"] = run.summary["experiment_yaml"]["model"]["name_or_path"]
        df["seed_idx"] = df.model_name.str.split("-").str[-1].astype(int)
        df["revision"] = run.summary["experiment_yaml"]["model"]["revision"]
        df["adv_training_round"] = df.revision.str.replace(
            "main", "adv-training-round-0"
        ).str.extract(r"adv-training-round-(\d+)", expand=False)
        dfs.append(df)
    concat_df = pd.concat(dfs)
    concat_df = add_model_idx_inplace(concat_df, reference_col="model_size")
    concat_df.drop_duplicates(
        inplace=True,
    )
    concat_df.to_csv(f"cache_csvs/asr_{group}.csv", index=False)


def main():
    disable_progress_bar()
    GROUPS = [
        "oskar_025a_gcg_eval_qwen25_ft_harmless",
        "oskar_025b_gcg_eval_qwen25_ft_spam",
    ]
    os.makedirs("outputs", exist_ok=True)
    for group in tqdm(GROUPS):
        maybe_download_asr_data(group)


if __name__ == "__main__":
    main()
