from __future__ import annotations

import concurrent.futures
import csv
import json
import re
import shutil
import time
import uuid
import zipfile
from collections import defaultdict
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
from robust_llm.file_utils import ATTACK_DATA_NAME, get_save_root
from robust_llm.wandb_utils.constants import PROJECT_NAME

WANDB_API = wandb.Api(timeout=90)


def get_attack_tables_cache_root() -> Path:
    path = Path(get_save_root()) / "attack_tables"
    if not path.exists():
        path.mkdir(parents=True)
    return path


def get_attack_tables_index_path() -> Path:
    return get_attack_tables_cache_root() / "index.json"


def try_get_run_from_id(run_id: str, retries: int = 5, backoff: int = 5) -> WandbRun:
    for attempt in range(retries):
        try:
            return WANDB_API.run(f"{PROJECT_NAME}/{run_id}")
        except Exception as e:
            print(f"Error getting run on attempt {attempt}: {e}")
            time.sleep(backoff * 2**attempt)
    raise ValueError(f"Failed to get run {run_id}")


def try_get_runs_from_wandb(
    group_name: str, retries: int = 5, backoff: int = 5
) -> list[WandbRun]:
    for attempt in range(retries):
        try:
            # setting per_page and order based on
            # https://github.com/wandb/wandb/issues/6614
            return [
                run
                for run in WANDB_API.runs(
                    path=PROJECT_NAME,
                    filters={"group": group_name},
                    per_page=1000,
                    order="+created_at",
                )
            ]
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
    maybe_unzip_attack_data_tables(wandb_run.name)
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
    return Path(get_save_root()) / "wandb-run-summaries" / group


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
    root = Path(get_save_root()) / "wandb-run-histories" / group
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
    cache_path = Path(get_save_root()) / "wandb-run-summaries" / f"{run_id}.json"
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
        if re.search(rf"-{run_index}$", run.name) and run.summary.get(
            "really_finished", False
        ):
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

    adv_seed_regex = r"^.*s-(\d+)_.*t-(\d+)"
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


def zip_attack_data_tables(only_prefix: str | None = None):
    """Zip JSON files from cache/attack_tables to save disk space and remove originals.

    We iterate through `cache/attack_tables` for directories which have names
    like `ian-113-rt-pythia-spam-0048`. We zip these directories based on the
    common prefix `ian-113-rt-pythia-spam` and write what we have done to `index.json`
    in the same directory. After successful zipping, original directories are removed.

    Arguments:
        only_prefix: If set, only zip directories with this
            prefix. This is useful if a group is actively in use.
    """

    tables_dir = get_attack_tables_cache_root()
    index_path = get_attack_tables_index_path()
    prefix_groups = defaultdict(list)

    # Group directories by prefix
    for path in tables_dir.iterdir():
        if not path.is_dir():
            continue
        prefix = "-".join(path.name.split("-")[:-1])
        if only_prefix is None or prefix == only_prefix:
            prefix_groups[prefix].append(path)

    index = {}
    for prefix, paths in prefix_groups.items():
        if len(paths) == 1:
            continue

        uuid_str = str(uuid.uuid4())
        zip_path = tables_dir / f"{prefix}-{uuid_str}.zip"
        print(f"Zipping {len(paths)} dirs to {zip_path}")

        try:
            # Create zip file
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for path in paths:
                    zipf.write(path, arcname=path.name)

            # Verify zip file integrity
            with zipfile.ZipFile(zip_path, "r") as zipf:
                if zipf.testzip() is not None:
                    raise zipfile.BadZipFile("Zip file verification failed")

            # If zip is valid, remove original directories
            for path in paths:
                try:
                    shutil.rmtree(path)
                    print(f"Removed original directory: {path}")
                except Exception as e:
                    print(f"Warning: Failed to remove directory {path}: {e}")

            # Update index
            index[zip_path.name] = [p.name for p in paths]

        except (Exception, KeyboardInterrupt) as e:
            print(f"Error processing {prefix}: {e}")
            # If zip creation fails, remove the partial zip file
            if zip_path.exists():
                zip_path.unlink()
            continue

    # Update index file
    if index_path.exists():
        with open(index_path) as f:
            index.update(json.load(f))

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def maybe_unzip_attack_data_tables(run_name: str) -> bool:
    """Unzip attack data tables containing the specified run_name and remove the zip.

    Args:
        run_name: Name of the run to extract

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    index_path = get_attack_tables_cache_root() / "index.json"
    if not index_path.exists():
        return False

    try:
        with open(index_path, "r") as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading index file: {e}")
        return False

    for name, paths in index.items():
        if run_name in paths:
            zip_path = get_attack_tables_cache_root() / name
            if not zip_path.exists():
                print(f"Warning: Zip file {zip_path} listed in index but not found")
                continue

            try:
                # First verify zip file integrity
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    if zipf.testzip() is not None:
                        print(f"Warning: Zip file {zip_path} is corrupted")
                        continue

                # Extract files
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    zipf.extractall(get_attack_tables_cache_root())
                print(f"Unzipped {zip_path}")

                # Verify all files were extracted
                all_files_exist = all(
                    (get_attack_tables_cache_root() / path).exists() for path in paths
                )

                if all_files_exist:
                    # Remove zip file after successful extraction
                    zip_path.unlink()
                    print(f"Removed zip file {zip_path}")

                    # Update index file to remove the entry
                    del index[name]
                    with open(index_path, "w") as f:
                        json.dump(index, f, indent=2)

                    return True
                else:
                    print("Warning: Not all files were extracted successfully")
                    return False

            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file")
            except PermissionError:
                print(f"Error: Permission denied when accessing {zip_path}")
            except Exception as e:
                print(f"Error processing {zip_path}: {e}")
            return False

    print(f"No zip file found containing {run_name}")
    return False


def download_artifact_with_retry(
    artifact: wandb.Artifact,
    root: str,
    retries: int = 5,
    backoff: int = 5,
):
    for attempt in range(retries):
        try:
            return artifact.download(root=root)
        except Exception as e:
            print(f"Error downloading artifact on attempt {attempt}: {e}")
            time.sleep(backoff * 2**attempt)
    raise ValueError(f"Failed to download artifact {artifact.name}")


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
    cache_path = get_attack_tables_cache_root() / run_name
    cache_path.mkdir(parents=True, exist_ok=True)
    table_path = cache_path / f"attack_data/example_{index}.table.json"

    if not table_path.exists() or table_path.stat().st_size == 0:
        table_dir = download_artifact_with_retry(artifact, str(cache_path))
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
        experiment_yaml.get("training", {})
        if experiment_yaml.get("training", {}) is not None
        else {}
    )
    hub_model_id = run.config.get(
        "hub_model_id", experiment_yaml.get("model", {}).get("name_or_path", None)
    )  # e.g. "AlignmentResearch/robust_llm_clf_pm_pythia-2.8b_s-0_adv_tr_gcg_t-0"
    match = (
        re.match(
            r"AlignmentResearch/robust_llm_clf_(.*)_pythia-(.*)_s-(.*)_adv_tr_(.*)_t-(.*)",  # noqa: E501
            hub_model_id,
        )
        if hub_model_id is not None
        else None
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
    adversarial_config = (
        training_yaml.get("adversarial", {})
        if training_yaml.get("adversarial", {}) is not None
        else {}
    )
    evaluation_config = (
        experiment_yaml.get("evaluation", {})
        if experiment_yaml.get("evaluation", {}) is not None
        else {}
    )

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
        "num_adversarial_training_rounds": (
            adversarial_config.get("num_adversarial_training_rounds", None)
        ),
        "eval_iterations": (evaluation_config.get("num_iterations")),
        "eval_attack": (
            "gcg"
            if evaluation_config.get("evaluation_attack", {}).get("n_candidates_per_it")
            is not None
            else "rt"
        ),
        "eval_round": revision.split("-")[-1] if revision is not None else None,
        "really_finished": run.summary.get("really_finished", False),
        "finish_reason": run.summary.get("finish_reason", None),
        "gpu_type": run.metadata.get("gpu", "Unknown"),
        "gpu_count": run.metadata.get("gpu_count", np.nan),
        "duration_seconds": run.summary.get("_runtime", np.nan),
        "created_at": run.created_at,
        "state": run.state,
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
    assert not df.empty
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
    df.really_finished = df.really_finished.astype(bool)
    return df
