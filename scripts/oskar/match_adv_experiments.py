import re

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

MODEL_DIRECTORY = "robust_llm/hydra_conf/model/AdvTrained/clf"


def search_wandb_runs():
    api = wandb.Api(timeout=60)
    runs = api.runs(
        "farai/robust-llm",
        filters={"config.hub_model_id": {"$regex": ".*_adv_tr.*"}, "state": "finished"},
    )
    data = []
    for run in tqdm(runs):
        hub_model_id = run.config.get(
            "hub_model_id"
        )  # e.g. "AlignmentResearch/robust_llm_clf_pm_pythia-2.8b_s-0_adv_tr_gcg_t-0"
        match = re.match(
            r"AlignmentResearch/robust_llm_clf_(.*)_pythia-(.*)_s-(.*)_adv_tr_(.*)_t-(.*)",  # noqa: E501
            hub_model_id,
        )
        assert match is not None
        dataset, base_model, ft_seed, attack, adv_seed = match.groups()
        if run.metadata is None:
            print(f"Run {run.id} has no metadata.")
            continue
        gpu_type = run.metadata.get("gpu", "Unknown")
        gpu_count = run.metadata.get("gpu_count", "Unknown")
        duration = run.summary.get("_runtime", "Unknown")
        assert all(a.count("=") == 1 for a in run.metadata["args"])
        args = {
            key.lstrip("+"): value
            for item in run.metadata["args"]
            for key, value in [item.split("=", 1)]
        }
        assert "experiment_name" in args
        adv_training_rounds = (
            run.summary.get("experiment_yaml", {})
            .get("training", {})
            .get("adversarial", {})
            .get("num_adversarial_training_rounds", None)
        )
        if adv_training_rounds is None:
            print(f"Run {run.id} is missing num_adversarial_training_rounds.")
        run_data = {
            "hub_model_id": hub_model_id,
            "wandb_run_id": run.id,
            "wandb_group_link": "https://wandb.ai/farai/robust-llm/groups/"
            + args["experiment_name"],
            "dataset": dataset,
            "base_model": base_model,
            "ft_seed": ft_seed,
            "attack": attack,
            "adv_seed": adv_seed,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "duration_seconds": duration,
            "wandb_run_link": run.url,
            "num_adversarial_training_rounds": adv_training_rounds,
        }
        run_data.update(args)
        data.append(run_data)
    return data


if __name__ == "__main__":
    output_csv = (
        "/home/oskar/projects/robust-llm/outputs/adv_model_experiment_mapping.csv"
    )

    data = search_wandb_runs()
    df = pd.DataFrame(data)
    df["num_parameters"] = (
        df.base_model.str.split("-")
        .str[-1]
        .str.replace("m", "e6")
        .str.replace("b", "e9")
        .astype(float)
    )
    df["duration_hours"] = df.duration_seconds.astype(float).div(3600)
    is_h100 = df.gpu_type.str.contains("H100")
    df["h100_hours"] = df.duration_hours * df.gpu_count * np.where(is_h100, 1, 0.25)
    if df.h100_hours.isnull().any():
        print(
            f"Warning {df.h100_hours.isnull().sum()} experiments are missing cost data."
        )
    df.sort_values(by=["num_parameters", "dataset", "attack", "ft_seed"], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' has been created with {len(df)} entries.")
