"""Deprecated tools to fetch wandb logs for old runs"""

import concurrent.futures
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from robust_llm.wandb_utils.wandb_api_tools import (
    RunInfo,
    _extract_adv_round_from_revision,
    get_history_cache_path,
    get_runs_for_group,
)


def _get_value_iterative(d: dict, key: str):
    for k in key.split("."):
        if k not in d:
            return None
        d = d[k]
        if d is None:
            return None
    return d


def _maybe_hack_run_missing_flops(history: pd.DataFrame, run: RunInfo) -> None:
    """Hacky way to fix runs which died before committing flops to wandb."""
    if run.id == "e6g2egro":
        # This run https://wandb.ai/farai/robust-llm/runs/e6g2egro did not commit the
        # flops before dying
        history.loc[history.adversarial_training_round.eq(0), "train/flops"] = (
            # Interpolated from https://wandb.ai/farai/robust-llm/runs/sy4lvytt
            # which has 455e15 for the second round
            227e15
        )
    elif run.id == "cq4yc0cb":
        # This run https://wandb.ai/farai/robust-llm/runs/cq4yc0cb seems to be missing
        # round 9 flops as the next run https://wandb.ai/farai/robust-llm/runs/xrlr9r2p
        # seemed to only log round 10 flops. We interpolate the missing value.
        history.loc[
            history.adversarial_training_round.shift(1) == 8,
            "adversarial_training_round",
        ] = 9
        history.loc[history.adversarial_training_round.shift(1) == 8, "train/flops"] = (
            3.5e19
        )
    elif run.id == "0nj11l12":
        # https://wandb.ai/farai/robust-llm/runs/0nj11l12 seems to be missing round 21
        # Next run https://wandb.ai/farai/robust-llm/runs/osmh69ad
        history.loc[
            history.adversarial_training_round.shift(1) == 20,
            "adversarial_training_round",
        ] = 21
        history.loc[
            history.adversarial_training_round.shift(1) == 20, "train/flops"
        ] = 6.3e18
    elif run.id == "sy4lvytt":
        # https://wandb.ai/farai/robust-llm/runs/sy4lvytt ss missing round 6
        # Next run https://wandb.ai/farai/robust-llm/runs/7wgctcxd
        history.loc[
            history.adversarial_training_round.shift(1) == 5,
            "adversarial_training_round",
        ] = 6
        history.loc[history.adversarial_training_round.shift(1) == 5, "train/flops"] = (
            7.85e18
        )
    elif run.id == "osmh69ad":
        assert history.adversarial_training_round.max() == 21
        assert history["train/flops"].notnull().sum() == 1
        assert bool(history["train/flops"].notnull().iloc[-1])
        assert bool(history.adversarial_training_round.isnull().iloc[-1])
        history.loc[
            history["train/flops"].notnull()
            & history.adversarial_training_round.isnull(),
            "adversarial_training_round",
        ] = 22
    elif run.id == "d5yw2nv7":
        # https://wandb.ai/farai/robust-llm/runs/d5yw2nv7 missing round 8
        # also not present in https://wandb.ai/farai/robust-llm/runs/2hm2r2ue
        # Interpolate (8.9+13.9)/2=11.4e18
        history.loc[
            history.adversarial_training_round.shift(1) == 7,
            "adversarial_training_round",
        ] = 8
        history.loc[history.adversarial_training_round.shift(1) == 7, "train/flops"] = (
            11.4e18
        )


def _get_history_for_run(run: RunInfo) -> pd.DataFrame:
    wandb_run = run.to_wandb()
    data = []
    for row in wandb_run.scan_history():
        data.append(row)
    history = pd.DataFrame(data)
    _maybe_hack_run_missing_flops(history, run)
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
        history.to_csv(cache_path, index=False)
        return history
    try:
        return pd.read_csv(cache_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fix_flops_round_mapping(history: pd.DataFrame, run: RunInfo) -> pd.DataFrame:
    if "train/flops" in history and "train/total_flops" not in history:
        # In the training pipeline refactor, `train/total_flops` was renamed to
        # `train/flops`. We also change the way we log metrics to wandb.
        # Also we need to get the flops from the previous adversarial training round.
        history = history.rename(columns={"train/flops": "train/total_flops"})
        history.adversarial_training_round = history.adversarial_training_round.ffill()
        history = history.loc[history["train/total_flops"].notnull()]
        dupe_rounds = history.adversarial_training_round.duplicated(keep="first")
        if dupe_rounds.any():
            # In https://wandb.ai/farai/robust-llm/runs/lv0vo7z the last round isn't
            # incremented for some reason so we do it manually.
            history.loc[dupe_rounds, "adversarial_training_round"] += 1
            history["adv_training_round"] = history.adversarial_training_round
        history["adv_training_round"] = history.adversarial_training_round
        return history

    if "adv_training_round" in history and "adversarial_training_round" not in history:
        # We don't need to do anything in the case of an evaluation run
        return history
    elif "adv_training_round" in history:
        # In older training runs, we had duplicate columns for adv training round
        history.loc[history.adv_training_round.eq(0), "train/total_flops"] = 0
        return history
    elif (
        "adversarial_training_round" not in history
        or "train/total_flops" not in history
    ):
        # We didn't get far enough into the run to log anything useful
        # e.g. https://wandb.ai/farai/robust-llm/runs/383zxn9j/overview
        # or https://wandb.ai/farai/robust-llm/runs/cunozfct/overview
        return pd.DataFrame()
    has_dummy_round = (history.adversarial_training_round.min() == 0) and (
        run.summary.get("experiment_yaml", {})
        .get("training", {})
        .get("adversarial", {})
        .get("skip_first_training_round", False)
    )
    if has_dummy_round:
        history.loc[
            np.where(history["train/total_flops"].isnull())[0][-1],
            "train/total_flops",
        ] = 0
    if bool(history["train/total_flops"].isnull().any()):
        # We have a crash in the middle of training so no data for the last round
        history["adv_training_round"] = history.adversarial_training_round.shift(1)
        history = history.iloc[1:]
    else:
        history["adv_training_round"] = history.adversarial_training_round
    assert history.adv_training_round.notnull().all()
    return history


def _filter_for_metrics(history: pd.DataFrame, metrics: list[str]):
    if "_step" not in metrics:
        metrics = ["_step"] + metrics
    filtered_metrics = [m for m in metrics if m in history.columns]
    return history.loc[
        history[filtered_metrics].notnull().all(axis=1), filtered_metrics
    ]


def _get_filtered_history(run: RunInfo, metrics: list[str]) -> pd.DataFrame:
    history = _get_full_history(run)
    if history.empty:
        return history
    assert not history.empty, f"Full history is empty for run {run.id}"
    history = _fix_flops_round_mapping(history, run)
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
    if history.empty:
        return history
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

    df = pd.concat(dfs, ignore_index=True)
    if "run_name" in df:
        round_col = (
            "adv_training_round"
            if "adv_training_round" in df
            else "adversarial_training_round"
        )
        df = df.drop_duplicates(subset=["run_name", round_col], keep="last")
    return df
