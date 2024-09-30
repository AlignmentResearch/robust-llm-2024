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


def get_adv_training_round_from_run(run: WandbRun) -> int:
    revision = run.summary.get("experiment_yaml", {}).get("model", {}).get("revision")
    return _extract_adv_round(revision)


def _extract_adv_round(revision: str) -> int:
    if revision == "main":
        # This happens for finetuned models
        return 0
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


def get_wandb_runs(group_name: str) -> WandbRuns:
    runs = WANDB_API.runs(
        path=PROJECT_NAME,
        filters={"group": group_name, "state": "finished"},
        # setting high per_page based on https://github.com/wandb/wandb/issues/6614
        per_page=1000,
        # Setting order avoids duplicates https://github.com/wandb/wandb/issues/6614
        order="+created_at",
    )
    return runs


def get_wandb_runs_by_index(group_name: str) -> dict[str, WandbRun]:
    """Return a dictionary of runs indexed by the run index."""
    runs = get_wandb_runs(group_name)
    run_dict = {}
    for run in runs:
        match = re.search(r"-(\d+)$", run.name)
        assert match is not None, f"Run name {run.name} does not end in '-<index>'"
        run_index = match.group(1)
        run_dict[run_index] = run
    return run_dict


def get_wandb_run(group_name: str, run_index: str):
    runs = get_wandb_runs(group_name)
    target_run = None
    for run in runs:
        if re.search(rf"-{run_index}$", run.name):
            target_run = run
    if target_run is None:
        raise ValueError(
            f"No finished run called {run_index} found in group {group_name}"
        )
    return target_run


def get_model_size_and_seed_from_run(run: WandbRun) -> tuple[int, int]:
    """Get the model size and seed from a wandb run.

    Requires that the model name is of one of these forms:
    - `...s-<seed}` or
    - `...s-<seed>_...t-<seed>` and the seeds match.
    """
    model_size = run.summary["model_size"]
    model_name = run.summary["experiment_yaml"]["model"]["name_or_path"]
    ft_seed_regex = r"^.*s-(\d+)$"
    ft_match = re.match(ft_seed_regex, model_name)
    if ft_match is not None:
        return int(model_size), int(ft_match.group(1))

    adv_seed_regex = r"^.*s-(\d+)_.*t-(\d+)$"
    adv_match = re.match(adv_seed_regex, model_name)
    if adv_match is None or len(adv_match.groups()) != 2:
        raise ValueError(f"Could not extract seed from model name {model_name}")
    ft_seed, adv_seed = adv_match.groups()
    if ft_seed != adv_seed:
        print(
            f"Warning: Seeds do not match: {ft_seed=} != {adv_seed=}."
            " Using adv_seed (i.e. t-<seed>)"
        )
    return int(model_size), int(adv_seed)


def download_and_process_attack_data_table(
    artifact: wandb.Artifact,
    run_name: str,
    cache_dir: str = "~/.cache/rllm",
):
    """Download an attack data table artifact to CACHE_DIR and return it as a DataFrame.

    Args:
        artifact: The wandb artifact to download.
        run_name: The name of the run on wandb.
        cache_dir: The directory to cache the downloaded table.

    Returns:
        A tuple of (the index of the example in the dataset, the table as a DataFrame).
        Returns None if the artifact is not an attack data table.
    """
    re_match = re.search(r"attack_dataexample_([\d]+):", artifact.name)
    if not re_match:
        return None
    index = int(re_match.group(1))

    table_path = download_attack_data_table_if_not_cached(
        artifact,
        run_name,
        cache_dir=cache_dir,
    )

    try:
        with table_path.open("r") as f:
            json_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading {table_path=} for {run_name=}") from e

    df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
    return index, df


def download_attack_data_table_if_not_cached(
    artifact: wandb.Artifact,
    run_name: str,
    cache_dir: str = "~/.cache/rllm",
) -> Path:
    """Download an attack data table artifact to CACHE_DIR if not already cached.

    Args:
        artifact: The wandb artifact to download.
        run_name: The name of the run on wandb.
        cache_dir: The directory to cache the downloaded table.

    Returns:
        The path to the download in the cache, or None.
    """
    re_match = re.search(r"attack_dataexample_([\d]+):", artifact.name)
    if not re_match:
        raise ValueError(f"{artifact.name = } does not match expected pattern")

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    index = int(re_match.group(1))
    table_path = cache_path / f"{run_name}/attack_data/example_{index}.table.json"

    if not table_path.exists() or table_path.stat().st_size == 0:
        table_dir = artifact.download(root=str(cache_path / run_name))
        assert str(table_path).startswith(table_dir)

    return table_path


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
    group,
    metrics,
    summary_keys,
    filters=None,
    check_num_runs=None,
    invalidate_cache=False,
):
    if invalidate_cache:
        result = _get_metrics_single_step.call_and_shelve(  # type: ignore
            group, metrics, summary_keys, filters, check_num_runs
        )
        result.clear()  # type: ignore
    return _get_metrics_single_step(
        group, metrics, summary_keys, filters, check_num_runs  # type: ignore
    )


@memory.cache
def _get_metrics_single_step(
    group,
    metrics,
    summary_keys,
    filters=None,
    check_num_runs=None,
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
    *args,
    invalidate_cache=False,
    **kwargs,
) -> pd.DataFrame:
    if invalidate_cache:
        result = _get_metrics_adv_training.call_and_shelve(  # type: ignore
            *args, **kwargs
        )
        result.clear()  # type: ignore
    return _get_metrics_adv_training(*args, **kwargs)  # type: ignore


@memory.cache
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
        history["run_created_at"] = run.created_at

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
            assert run.summary.get("experiment_yaml", {}).get("training") is not None, (
                f"Could not infer adversarial training round for run {run.id}. "
                "The run has no training config and no revision was found using "
                f"keys={summary_keys}."
            )
            history["adv_training_round"] = np.arange(len(history))

        res.append(history)

    if len(res) == 0:
        raise ValueError(f"No data found for group {group}")
    res = pd.concat(res, ignore_index=True)
    return res


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
        # Handle model misnaming issue (GH #921)
        match = re.match(
            r"AlignmentResearch/clf_(.*)_pythia-(.*)_s-(.*)_adv_tr_(.*)_t-(.*)",  # noqa: E501
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
