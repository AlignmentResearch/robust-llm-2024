import concurrent.futures
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from joblib import Memory
from tqdm import tqdm
from wandb.apis.public.runs import Run as WandbRun
from wandb.apis.public.runs import Runs as WandbRuns

from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.file_utils import compute_repo_path
from robust_llm.wandb_utils.constants import PROJECT_NAME

WANDB_API = wandb.Api(timeout=90)

# Set up the cache directory
cache_dir = compute_repo_path() + "/cache/get-metrics-adv-training"
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


def _extract_adv_round(revision: str) -> int:
    re_match = re.search(r"adv-training-round-(\d+)", revision)
    if re_match:
        return int(re_match.group(1))
    else:
        raise ValueError(f"Invalid revision format: {revision}")


def get_attack_data_tables(
    run: WandbRun, max_workers: int = 4
) -> dict[int, pd.DataFrame]:
    artifacts = run.logged_artifacts()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_artifact = {
            executor.submit(
                download_and_process_attack_data_table, artifact, run.name
            ): artifact
            for artifact in artifacts
        }

        dfs = {}
        for future in concurrent.futures.as_completed(future_to_artifact):
            result = future.result()
            if result:
                index, df = result
                dfs[index] = df

    return dfs


def get_wandb_run(group_name: str, run_index: str):
    runs = WANDB_API.runs(
        path=PROJECT_NAME,
        filters={"group": group_name, "state": "finished"},
        # setting high per_page based on https://github.com/wandb/wandb/issues/6614
        per_page=1000,
        # Setting order avoids duplicates https://github.com/wandb/wandb/issues/6614
        order="+created_at",
    )
    target_run = None
    for run in runs:
        if re.search(rf"-{run_index}$", run.name):
            target_run = run
    if target_run is None:
        raise ValueError(
            f"No finished run called {run_index} found in group {group_name}"
        )
    return target_run


def download_and_process_attack_data_table(artifact: wandb.Artifact, run_name: str):
    """Download an attack data table artifact to /tmp/ and return it as a DataFrame.

    Args:
        artifact: The wandb artifact to download.
        run_name: The name of the run on wandb.

    Returns:
        A tuple of (the index of the example in the dataset, the table as a DataFrame).
        Returns None if the artifact is not an attack data table.
    """
    re_match = re.search(r"attack_dataexample_([\d]+):", artifact.name)
    if not re_match:
        return None

    index = int(re_match.group(1))
    table_path = Path(f"/tmp/{run_name}/attack_data/example_{index}.table.json")

    if not table_path.exists() or table_path.stat().st_size == 0:
        table_dir = artifact.download(root=f"/tmp/{run_name}")
        assert str(table_path).startswith(table_dir)

    with table_path.open("r") as f:
        json_data = json.load(f)

    df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
    return index, df


def get_dataset_config_from_run(run):
    """Get a DatasetConfig object from wandb.

    NOTE: This uses the API of wandb.old.summary. Presumably at some point
    they'll start pointing to wandb.summary, and this will break.
    """
    dataset_subdict = run.summary["experiment_yaml"]["dataset"]
    cfg_dict = {k: v for k, v in dataset_subdict.items()}
    dataset_cfg = DatasetConfig(**cfg_dict)  # type: ignore
    return dataset_cfg


def _get_value_iterative(d: dict, key: str):
    for k in key.split("."):
        if k not in d:
            return None
        d = d[k]
        if d is None:
            return None
    return d


def _get_metrics_single_step(
    group, metrics, summary_keys, filters=None, check_num_runs=None
):
    print("getting metrics for", group)

    if filters is None:
        filters = {}
    filters = deepcopy(filters)
    filters["group"] = group
    filters["state"] = "finished"

    runs = WANDB_API.runs(path=PROJECT_NAME, filters=filters)
    if check_num_runs is not None:
        if type(check_num_runs) is int:
            assert len(runs) == check_num_runs
        elif type(check_num_runs) is list:
            assert check_num_runs[0] <= len(runs) <= check_num_runs[1]
        else:
            assert False, "bad check_num_runs!"

    res = []
    for run in runs:
        history = run.history(keys=metrics)
        if history is None or len(history) == 0:
            continue
        for key in summary_keys:
            history[key.split(".")[-1]] = _get_value_iterative(run.summary, key)
        assert len(history) == 1, "Expected 1 step, got {}".format(len(history))
        res.append(history)

    res = pd.concat(res, ignore_index=True)

    return res


@memory.cache
def _cached_get_metrics_single_step(
    group, metrics, summary_keys, filters=None, check_num_runs=None
):
    return _get_metrics_single_step(
        group, metrics, summary_keys, filters=filters, check_num_runs=check_num_runs
    )


def get_metrics_single_step(*args, use_cache=True, **kwargs):
    if use_cache:
        return _cached_get_metrics_single_step(*args, **kwargs)

    return _get_metrics_single_step(*args, **kwargs)


def _get_metrics_adv_training(
    group,
    metrics,
    summary_keys,
    filters=None,
    check_num_runs=None,
    check_num_data_per_run=None,
    verbose=False,
) -> pd.DataFrame:
    if filters is None:
        filters = {"state": "finished"}
    filters = deepcopy(filters)
    filters["group"] = group

    runs = WANDB_API.runs(path=PROJECT_NAME, filters=filters)
    if verbose:
        print(f"Found {len(runs)} runs")
    if check_num_runs is not None:
        if type(check_num_runs) is int:
            assert len(runs) == check_num_runs
        elif type(check_num_runs) is list:
            assert check_num_runs[0] <= len(runs) <= check_num_runs[1]
        else:
            assert False, "bad check_num_runs!"

    res = []
    run_iterator = tqdm(runs, desc="Processing runs") if len(runs) > 100 else runs
    for run in run_iterator:
        filtered_metrics = [m for m in metrics if run.summary.get(m) is not None]
        if not filtered_metrics:
            print(f"Run {run.id} is missing all metrics. Keys={run.summary.keys()}")
        history = run.history(keys=filtered_metrics)
        if verbose:
            print(f"Run {run.id} has {len(history)} data points")

        if check_num_data_per_run is not None:
            assert len(history) == check_num_data_per_run

        if len(history) == 0 or history is None:
            print(f"Run {run.id} has no data")
            continue

        for key in summary_keys:
            # First, replace '.' with '_' in the key
            # Next, delete the "experiment_yaml_" prefix if present
            new_key = key.replace(".", "_")
            if new_key.startswith("experiment_yaml_"):
                new_key = new_key[len("experiment_yaml_") :]
            history[new_key] = _get_value_iterative(run.summary, key)

        history["run_id"] = run.id
        history["run_state"] = run.state

        # Hack: create 'round' column based on increasing steps.
        history = history.sort_values(by="_step")

        if "revision" in history:
            history["adv_training_round"] = [
                _extract_adv_round(name) for name in history["revision"]
            ]
        elif "model_revision" in history:
            history["adv_training_round"] = [
                _extract_adv_round(name) for name in history["model_revision"]
            ]
        else:
            history["adv_training_round"] = np.arange(len(history))

        res.append(history)

    if len(res) == 0:
        raise ValueError(f"No data found for group {group}")
    res = pd.concat(res, ignore_index=True)
    return res


@memory.cache
def _cached_get_metrics_adv_training(
    group,
    metrics,
    summary_keys,
    filters=None,
    check_num_runs=None,
    check_num_data_per_run=None,
    verbose=False,
):
    return _get_metrics_adv_training(
        group,
        metrics,
        summary_keys,
        filters=filters,
        check_num_runs=check_num_runs,
        check_num_data_per_run=check_num_data_per_run,
        verbose=verbose,
    )


def get_metrics_adv_training(*args, use_cache=True, **kwargs):
    if use_cache:
        return _cached_get_metrics_adv_training(*args, **kwargs)

    return _get_metrics_adv_training(*args, **kwargs)


def parse_run_to_dict(run: WandbRun) -> dict[str, Any]:
    if run.metadata is None:
        print(f"Run {run.id} has no metadata")
        return {}
    assert all(a.count("=") == 1 for a in run.metadata["args"])
    args = {
        key.lstrip("+"): value
        for item in run.metadata["args"]
        for key, value in [item.split("=", 1)]
    }
    assert "experiment_name" in args
    experiment_yaml = run.summary.get("experiment_yaml", {})
    training_yaml = (
        experiment_yaml["training"] if experiment_yaml["training"] is not None else {}
    )
    hub_model_id = run.config.get(
        "hub_model_id", experiment_yaml.get("model", {}).get("name_or_path", None)
    )  # e.g. "AlignmentResearch/robust_llm_clf_pm_pythia-2.8b_s-0_adv_tr_gcg_t-0"
    match = re.match(
        r"AlignmentResearch/robust_llm_clf_(.*)_pythia-(.*)_s-(.*)_adv_tr_(.*)_t-(.*)",  # noqa: E501
        hub_model_id,
    )
    if match is None:
        dataset, base_model, ft_seed, attack, adv_seed = [None] * 5
    else:
        dataset, base_model, ft_seed, attack, adv_seed = match.groups()
    num_parameters = (
        float(base_model.split("-")[-1].replace("m", "e6").replace("b", "e9"))
        if base_model is not None
        else None
    )
    revision = experiment_yaml.get("model", {}).get("revision", None)

    run_data = {
        "wandb_run_link": run.url,
        "wandb_run_id": run.id,
        "wandb_group_link": "https://wandb.ai/farai/robust-llm/groups/"
        + args["experiment_name"],
        "host": run.metadata.get("host", "Unknown"),
        "hub_model_id": hub_model_id,
        "dataset": dataset,
        "base_model": base_model,
        "ft_seed": ft_seed,
        "attack": attack,
        "adv_seed": adv_seed,
        "num_parameters": num_parameters,
        "num_adversarial_training_rounds": training_yaml.get("adversarial", {}).get(
            "num_adversarial_training_rounds", None
        ),
        "eval_iterations": experiment_yaml.get("evaluation", {}).get("num_iterations"),
        "eval_attack": (
            "gcg"
            if experiment_yaml.get("evaluation", {})
            .get("evaluation_attack", {})
            .get("n_candidates_per_it")
            is not None
            else "rt"
        ),
        "eval_round": revision.split("-")[-1] if revision is not None else None,
        "really_finished": run.summary.get("really_finished", False),
        "gpu_type": run.metadata.get("gpu", "Unknown"),
        "gpu_count": run.metadata.get("gpu_count", "Unknown"),
        "duration_seconds": run.summary.get("_runtime", "Unknown"),
    }
    run_data.update(args)
    return run_data


def parse_runs_to_dataframe(runs: WandbRuns) -> pd.DataFrame:
    data = []
    for run in tqdm(runs):
        run_data = parse_run_to_dict(run)
        data.append(run_data)
    df = pd.DataFrame(data)
    df["duration_hours"] = df.duration_seconds.astype(float).div(3600)
    host_df = df.groupby("host").duration_hours.aggregate(["max", "sum"])
    host_df.columns = ["max_duration_hours", "total_duration_hours"]
    df = df.merge(host_df, on="host", how="left")
    df["duration_hours_pro_rata"] = (
        df.duration_hours / df.total_duration_hours
    ) * df.max_duration_hours
    is_h100 = df.gpu_type.str.contains("H100")
    df["h100_hours"] = (
        df.duration_hours_pro_rata * df.gpu_count * np.where(is_h100, 1, 0.25)
    )
    df.sort_values(by=["num_parameters", "dataset", "attack", "ft_seed"], inplace=True)
    if df.num_parameters.isnull().any():
        print(
            f"Warning {df.num_parameters.isnull().sum()} runs are missing model size."
        )
    if df.h100_hours.isnull().any():
        print(f"Warning {df.h100_hours.isnull().sum()} runs are missing cost data.")
    return df
