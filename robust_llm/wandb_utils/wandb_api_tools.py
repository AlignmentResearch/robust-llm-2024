import concurrent.futures
import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from wandb.apis.public.runs import Run as WandbRun

from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.wandb_utils.constants import PROJECT_NAME

WANDB_API = wandb.Api(timeout=90)


def _extract_adv_round(revision: str) -> int:
    re_match = re.search(r"adv-training-round-(\d+)", revision)
    if re_match:
        return int(re_match.group(1))
    else:
        raise ValueError(f"Invalid revision format: {revision}")


def get_attack_data_tables(run: WandbRun) -> dict[int, pd.DataFrame]:
    artifacts = run.logged_artifacts()

    with ThreadPoolExecutor() as executor:
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

    if not table_path.exists():
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


def get_metrics_single_step(
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


def get_metrics_adv_training(
    group,
    metrics,
    summary_keys,
    filters=None,
    check_num_runs=None,
    check_num_data_per_run=None,
    verbose=False,
) -> pd.DataFrame:
    if filters is None:
        filters = {}
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
    for run in runs:
        history = run.history(keys=metrics)
        if verbose:
            print(f"Run {run.id} has {len(history)} data points")

        if check_num_data_per_run is not None:
            assert len(history) == check_num_data_per_run

        if len(history) == 0:
            continue

        if history is None:
            continue

        for key in summary_keys:
            history[key.replace(".", "_")] = _get_value_iterative(run.summary, key)
        history["run_id"] = run.id
        history["run_state"] = run.state

        # Hack: create 'round' column based on increasing steps.
        history = history.sort_values(by="_step")

        if "revision" in history:
            history["adv_training_round"] = [
                _extract_adv_round(name) for name in history["revision"]
            ]
        else:
            history["adv_training_round"] = np.arange(len(history))

        res.append(history)

    if len(res) == 0:
        return pd.DataFrame()
    res = pd.concat(res, ignore_index=True)

    return res
