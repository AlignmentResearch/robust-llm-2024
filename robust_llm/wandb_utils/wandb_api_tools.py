from __future__ import annotations

import concurrent.futures
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from wandb.apis.public.runs import Run as WandbRun
from wandb.apis.public.runs import Runs as WandbRuns

from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.file_utils import ATTACK_DATA_NAME, compute_repo_path
from robust_llm.wandb_utils.constants import PROJECT_NAME

WANDB_API = wandb.Api(timeout=90)


def get_cache_root() -> Path:
    root = Path("/robust_llm_data")
    if not root.exists():
        root = Path(compute_repo_path())
    path = root / "cache"
    if not path.exists():
        path.mkdir(parents=True)
    return path


def try_get_run_from_id(run_id: str, retries: int = 5, backoff: int = 5) -> WandbRun:
    for attempt in range(retries):
        try:
            return WANDB_API.run(f"{PROJECT_NAME}/{run_id}")
        except Exception as e:
            print(f"Error getting run on attempt {attempt}: {e}")
            time.sleep(backoff * 2**attempt)
    raise ValueError(f"Failed to get run {run_id}")


def _get_max_adv_training_round(run: WandbRun) -> int:
    return run.summary.get(
        "adv_training_round", run.summary.get("adversarial_training_round", 0)
    )


def try_get_runs_from_wandb(
    group_name: str, retries: int = 5, backoff: int = 5
) -> list[WandbRun]:
    for attempt in range(retries):
        try:
            # setting per_page and order based on
            # https://github.com/wandb/wandb/issues/6614
            runs = WANDB_API.runs(
                path=PROJECT_NAME,
                filters={"group": group_name},
                per_page=1000,
                order="+created_at",
            )
            name_to_run = dict()
            for run in tqdm(runs, desc=f"Getting runs for {group_name}"):
                # runs which have `really_finished`` are always good to use
                # otherwise we take the run ID per run name with the most
                # adversarial training rounds
                if run.summary.get("really_finished") == 1:
                    name_to_run[run.name] = run
                    continue
                max_round = _get_max_adv_training_round(run)
                prev_furthest_run = name_to_run.get(run.name)
                prev_furthest_round = (
                    _get_max_adv_training_round(prev_furthest_run)
                    if prev_furthest_run is not None
                    else 0
                )
                if max_round > prev_furthest_round:
                    name_to_run[run.name] = run
            run_list = list(name_to_run.values())
            return run_list
        except Exception as e:
            print(f"Error getting runs on attempt {attempt}: {e}")
            time.sleep(backoff * 2**attempt)
    raise ValueError(f"Failed to get runs for group {group_name}")


@dataclass
class RunInfo:
    id: str
    name: str
    group: str
    state: str
    created_at: str
    summary: dict[str, Any]

    def to_json(self, path: Path) -> None:
        with path.open("w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_json(cls, path: Path) -> RunInfo:
        with path.open("r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_wandb(cls, run: WandbRun) -> RunInfo:
        return cls(
            id=run.id,
            name=run.name,
            group=run.group,
            state=run.state,
            created_at=run.created_at,
            summary=run.summary._json_dict,
        )

    def to_wandb(self) -> WandbRun:
        return try_get_run_from_id(self.id)


def get_adv_training_round_from_eval_run(run: RunInfo) -> int:
    revision = run.summary.get("experiment_yaml", {}).get("model", {}).get("revision")
    return _extract_adv_round_from_revision(revision)


def _extract_adv_round_from_revision(revision: str) -> int:
    if revision == "main":
        # This happens for finetuned models
        return 0
    re_match = re.search(r"adv-training-round-(\d+)", revision)
    if re_match:
        return int(re_match.group(1))
    else:
        raise ValueError(f"Invalid revision format: {revision}")


def get_attack_data_tables(
    run: RunInfo, max_workers: int = 4
) -> dict[int, pd.DataFrame]:
    wandb_run = run.to_wandb()
    dfs = _maybe_get_attack_data_from_storage(wandb_run)
    if dfs is not None:
        return dfs
    dfs = _maybe_get_attack_data_from_artifacts(wandb_run)
    if dfs is not None:
        return dfs

    # Fallback to the old way of downloading the attack data tables
    artifacts = wandb_run.logged_artifacts()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_artifact = {
            executor.submit(
                download_and_process_attack_data_table, artifact, wandb_run.name
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


def get_summary_cache_path_for_group(group: str) -> Path:
    return get_cache_root() / "wandb-run-summaries" / group


def _get_attack_data_from_csv(
    path: Path | str,
) -> dict[int, pd.DataFrame]:
    concat_df = pd.read_csv(path, quoting=csv.QUOTE_ALL, escapechar="\\")
    dfs = {}
    for index, df in concat_df.groupby("example_idx"):
        dfs[index] = df.drop(columns="example_idx")
    return dfs


def _maybe_get_attack_data_from_storage(
    run: WandbRun,
) -> dict[int, pd.DataFrame] | None:
    local_files_path = run.summary.get("local_files_path")
    if local_files_path is None:
        return None
    path = Path(local_files_path) / ATTACK_DATA_NAME
    if not path.exists():
        print(f"Warning: {path} does not exist for run {run.name}")
        return None
    return _get_attack_data_from_csv(path)


def _maybe_get_attack_data_from_artifacts(
    run: WandbRun,
) -> dict[int, pd.DataFrame] | None:
    for artifact in run.logged_artifacts():
        if artifact.name.endswith("attack_data:v0"):
            root = artifact.download()
            return _get_attack_data_from_csv(Path(root) / ATTACK_DATA_NAME)
    return None


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


def get_summary_cache_path(group: str, run_id: str) -> Path:
    root = get_summary_cache_path_for_group(group)
    if not root.exists():
        root.mkdir(parents=True)
    return root / f"{run_id}.json"


def get_history_cache_path(group: str, run_id: str) -> Path:
    root = get_cache_root() / "wandb-run-histories" / group
    if not root.exists():
        root.mkdir(parents=True)
    return root / f"{run_id}.csv"


def cache_run_and_return_info(run: WandbRun) -> RunInfo:
    cache_path = get_summary_cache_path(run.group, run.id)
    run_info = RunInfo.from_wandb(run)
    run_info.to_json(cache_path)
    return run_info


def get_runs_from_cache(group_name: str) -> list[RunInfo]:
    cache_path = get_summary_cache_path_for_group(group_name)
    run_ids = [path.stem for path in cache_path.iterdir()]
    return [RunInfo.from_json(cache_path / f"{run_id}.json") for run_id in run_ids]


def get_runs_for_group(group_name: str, use_cache: bool = True) -> list[RunInfo]:
    cache_path = get_summary_cache_path_for_group(group_name)
    if not use_cache or not cache_path.exists():
        runs = try_get_runs_from_wandb(group_name)
        return [cache_run_and_return_info(run) for run in runs]
    return get_runs_from_cache(group_name)


def get_summary_from_id(run_id: str) -> RunInfo:
    cache_path = get_cache_root() / "wandb-run-summaries" / f"{run_id}.json"
    if not cache_path.exists():
        run = try_get_run_from_id(run_id)
        return cache_run_and_return_info(run)
    return RunInfo.from_json(cache_path)


def get_wandb_runs_by_index(
    group_name: str, use_cache: bool = True
) -> dict[str, RunInfo]:
    """Return a dictionary of runs indexed by the run index."""
    runs = get_runs_for_group(group_name, use_cache=use_cache)
    run_dict = {}
    for run in runs:
        match = re.search(r"-(\d+)$", run.name)
        assert match is not None, f"Run name {run.name} does not end in '-<index>'"
        run_index = match.group(1)
        run_dict[run_index] = run
    return run_dict


def get_run_from_index(
    group_name: str, run_index: str, use_cache: bool = True
) -> RunInfo:
    runs = get_runs_for_group(group_name, use_cache=use_cache)
    target_run = None
    for run in runs:
        if re.search(rf"-{run_index}$", run.name):
            target_run = run
    if target_run is None:
        raise ValueError(
            f"No finished run called {run_index} found in group {group_name}"
        )
    return target_run


def get_model_size_and_seed_from_run(run: RunInfo) -> tuple[int, int]:
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
):
    """Download an attack data table artifact to CACHE_DIR and return it as a DataFrame.

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

    table_path = download_attack_data_table_if_not_cached(
        artifact,
        run_name,
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
) -> Path:
    """Download an attack data table artifact to CACHE_DIR if not already cached.

    Args:
        artifact: The wandb artifact to download.
        run_name: The name of the run on wandb.

    Returns:
        The path to the download in the cache, or None.
    """
    re_match = re.search(r"attack_dataexample_([\d]+):", artifact.name)
    if not re_match:
        raise ValueError(f"{artifact.name=} does not match expected pattern")

    index = int(re_match.group(1))
    cache_path = get_cache_root() / "attack_tables" / run_name
    cache_path.mkdir(parents=True, exist_ok=True)
    table_path = cache_path / f"attack_data/example_{index}.table.json"

    if not table_path.exists() or table_path.stat().st_size == 0:
        table_dir = artifact.download(root=str(cache_path))
        assert str(table_path).startswith(table_dir)

    return table_path


def get_dataset_config_from_run(run: RunInfo) -> DatasetConfig:
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


def _get_history_for_run(run: RunInfo) -> pd.DataFrame:
    wandb_run = run.to_wandb()
    data = []
    for row in wandb_run.scan_history():
        data.append(row)
    history = pd.DataFrame(data)
    assert isinstance(history, pd.DataFrame)
    return history


def _safe_get_history(
    run: RunInfo, max_retries: int = 5, backoff: int = 2
) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            # Get the full history here (we will filter later).
            return _get_history_for_run(run)
        except Exception as e:
            print(f"Error getting history on attempt {attempt} for run {run.id}: {e}")
            time.sleep(backoff * 2**attempt)
    raise ValueError(f"Failed to get history for run {run.id}")


def _get_full_history(run: RunInfo) -> pd.DataFrame:
    cache_path = get_history_cache_path(run.group, run.id)
    if not cache_path.exists():
        history = _safe_get_history(run)
        assert not history.empty, f"Raw history is empty for run {run.id}"
        history.to_csv(cache_path, index=False)
        return history
    return pd.read_csv(cache_path)


def _fix_deprecated_adversarial_training_round(history: pd.DataFrame):
    if "adv_training_round" not in history and "adversarial_training_round" in history:
        history["adv_training_round"] = history["train/total_flops"].notnull().cumsum()
    history.adv_training_round = history.adv_training_round.where(
        history.adv_training_round.notnull(),
        (history.adv_training_round.bfill() - 1).clip(0, None),
    )
    assert history.adv_training_round.min() == 0, "adv_training_round is not 0-indexed"


def _fix_round_0_flops(history: pd.DataFrame):
    if "train/total_flops" in history and "adv_training_round" in history:
        history.loc[
            history.adv_training_round.eq(0) & history["train/total_flops"].isnull(),
            "train/total_flops",
        ] = 0


def _filter_for_metrics(history: pd.DataFrame, metrics: list[str]):
    if "_step" not in metrics:
        metrics = ["_step"] + metrics
    filtered_metrics = [m for m in metrics if m in history.columns]
    return history.loc[
        history[filtered_metrics].notnull().all(axis=1), filtered_metrics
    ]


def _get_filtered_history(run: RunInfo, metrics: list[str]) -> pd.DataFrame:
    history = _get_full_history(run)
    assert not history.empty, f"Full history is empty for run {run.id}"
    _fix_deprecated_adversarial_training_round(history)
    _fix_round_0_flops(history)
    return _filter_for_metrics(history, metrics)


def get_enriched_history(
    run: RunInfo,
    metrics: list[str] | None = None,
    summary_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Main entrypoint to get wandb logs for a single run."""
    if metrics is None:
        metrics = []
    if summary_keys is None:
        summary_keys = []
    history = _get_filtered_history(run, metrics)
    assert not history.empty, (
        f"Filtered history is empty for run={run.id}, "
        f"metrics={metrics}, summary_keys={summary_keys}"
    )
    for key in summary_keys:
        new_key = key.replace(".", "_")
        if new_key.startswith("experiment_yaml_"):
            new_key = new_key[len("experiment_yaml_") :]
        history[new_key] = _get_value_iterative(run.summary, key)
    history["run_id"] = run.id
    history["run_state"] = run.state
    history["run_created_at"] = run.created_at
    history = history.sort_values(by="_step")
    if "adv_training_round" in history:
        history.adv_training_round = history.adv_training_round.astype(int)
    elif "revision" in history:
        history["adv_training_round"] = [
            _extract_adv_round_from_revision(name) for name in history["revision"]
        ]
    elif "model_revision" in history:
        history["adv_training_round"] = [
            _extract_adv_round_from_revision(name) for name in history["model_revision"]
        ]
    return history


def get_group_enriched_history(
    group_name: str,
    metrics: list[str] | None = None,
    summary_keys: list[str] | None = None,
    use_group_cache: bool = True,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Main entrypoint to get wandb logs for a group of runs."""
    runs = get_runs_for_group(group_name, use_cache=use_group_cache)

    def process_run(run):
        cache_path = get_history_cache_path(run.group, run.id)
        if cache_path.exists():
            # If cached, process synchronously
            return get_enriched_history(run, metrics, summary_keys)
        else:
            # If not cached, return the run to be processed asynchronously
            return run

    # First, process all runs that have cache
    dfs = []
    runs_to_fetch = []
    for run in tqdm(runs, desc=f"Processing cached runs for {group_name}"):
        result = process_run(run)
        if isinstance(result, pd.DataFrame):
            dfs.append(result)
        else:
            runs_to_fetch.append(result)

    # Then, use multithreading for runs that need fetching
    if runs_to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(get_enriched_history, run, metrics, summary_keys)
                for run in runs_to_fetch
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(runs_to_fetch),
                desc=f"Fetching uncached runs for {group_name}",
            ):
                try:
                    df = future.result()
                    dfs.append(df)
                except Exception as e:
                    print(f"An error occurred while processing a run: {e}")

    return pd.concat(dfs, ignore_index=True)


def _get_tracking_data_for_run(run: WandbRun) -> dict[str, Any]:
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


def get_tracking_data_for_runs(runs: WandbRuns) -> pd.DataFrame:
    """Create a dataframe for GSheet export to track runs."""
    data = []
    for run in tqdm(runs, desc="Getting tracking data"):
        run_data = _get_tracking_data_for_run(run)
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
