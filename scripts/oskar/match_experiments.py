import os
import re

import pandas as pd
import wandb
import yaml
from tqdm import tqdm

MODEL_DIRECTORY = "robust_llm/hydra_conf/model/Default/clf"
EXPERIMENT_DIRECTORY = "experiments/ian"


def process_yaml_files(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        data = yaml.safe_load(f)
                        name_or_path = data.get("name_or_path", "")
                        if name_or_path:
                            match = re.search(r"_v-([^_]+)_s-", name_or_path)
                            if match:
                                version = match.group(1).replace("-", "_")
                                experiment_name = find_experiment(version)
                                if experiment_name:
                                    results.append((name_or_path, experiment_name))
                    except yaml.YAMLError as e:
                        print(f"Error parsing {file_path}: {e}")
    return results


def find_experiment(version):
    for file in os.listdir(EXPERIMENT_DIRECTORY):
        if file.startswith(version) and file.endswith(".py"):
            return os.path.splitext(file)[0]
    return None


def search_wandb_runs(df):
    api = wandb.Api(timeout=60)
    experiment_groups = df.groupby("experiment_name")
    run_info = {}

    for experiment_name, group in tqdm(experiment_groups):
        runs = api.runs("farai/robust-llm", {"group": experiment_name})
        for run in runs:
            if run.state != "finished":
                continue
            assert run.metadata is not None
            assert run.summary is not None
            training_config = run.summary.get("experiment_yaml", {}).get("training", {})
            name_to_save = training_config.get(
                # Parameter changed from force_name_to_save to save_name
                "force_name_to_save",
                training_config.get("save_name"),
            )
            if name_to_save:
                model_name = "AlignmentResearch/robust_llm_" + name_to_save
                if model_name in group["model_name"].values:
                    gpu_type = run.metadata.get("gpu", "Unknown")
                    gpu_count = run.metadata.get("gpu_count", "Unknown")
                    duration = run.summary.get("_runtime", "Unknown")
                    run_info[model_name] = {
                        "wandb_run_link": run.url,
                        "gpu_type": gpu_type,
                        "gpu_count": gpu_count,
                        "duration_seconds": duration,
                    }

    return run_info


if __name__ == "__main__":
    output_csv = "tmp/model_experiment_mapping.csv"

    results = process_yaml_files(MODEL_DIRECTORY)
    df = pd.DataFrame(
        results, columns=["model_name", "experiment_name"]  # type: ignore
    )
    df["base_model"] = df["model_name"].str.extract(r"_([^_]*)_clf_")
    df["num_parameters"] = (
        df.base_model.str.split("-")
        .str[-1]
        .str.replace("m", "e6")
        .str.replace("b", "e9")
        .astype(float)
    )
    df["dataset"] = df["model_name"].str.extract(r"_clf_([^_]*)_v-")
    df["version"] = df["model_name"].str.extract(r"_v-([^_]*)_s-")
    df["seed"] = df["model_name"].str.extract(r"_s-([^_]*)")
    df["hf_link"] = "huggingface.co/" + df["model_name"]
    df["wandb_group_link"] = (
        "https://wandb.ai/farai/robust-llm/groups/" + df["experiment_name"]
    )

    # Search for matching wandb runs and extract additional information
    run_info = search_wandb_runs(df)

    # Add new columns to the DataFrame
    df["wandb_run_link"] = df["model_name"].map(
        lambda x: run_info.get(x, {}).get("wandb_run_link", None)
    )
    df["gpu_type"] = df["model_name"].map(
        lambda x: run_info.get(x, {}).get("gpu_type", None)
    )
    df["gpu_count"] = (
        df["model_name"]
        .map(lambda x: run_info.get(x, {}).get("gpu_count", None))
        .astype(int)
    )
    df["duration_seconds"] = df["model_name"].map(
        lambda x: run_info.get(x, {}).get("duration_seconds", None)
    )

    # Convert duration from seconds to hours
    df["duration_hours"] = df.duration_seconds.astype(float).div(3600)
    df["h100_hours"] = (
        df.duration_hours * df.gpu_count * df.gpu_type.str.contains("H100")
    )

    df.sort_values(by=["num_parameters", "dataset", "version", "seed"], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' has been created with {len(df)} entries.")
