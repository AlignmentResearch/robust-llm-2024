"""
Iterations for success (IFS) metric:

The minimum number of iterations required to achieve a certain attack success rate (ASR)
over the whole dataset. We use ASR=0%, 10%, 20%, ..., 100% as thresholds.
"""

import argparse
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from robust_llm.attacks.attack import AttackOutput
from robust_llm.metrics.asr_per_iteration import (
    ASRMetricResults,
    compute_asr_per_iteration_from_logits,
    compute_asr_per_iteration_from_text,
)
from robust_llm.metrics.metric_utils import get_attack_output_from_wandb_run
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.plotting_utils.utils import add_model_idx_inplace
from robust_llm.scoring_callbacks.scoring_callback_utils import BinaryCallback
from robust_llm.wandb_utils.wandb_api_tools import (
    RunInfo,
    get_adv_training_round_from_eval_run,
    get_model_size_and_seed_from_run,
    get_run_from_index,
    get_wandb_runs_by_index,
)


@dataclass(frozen=True)
class IFSMetricResults:
    """Results of computing robustness metrics on a dataset.

    NOTE: iterations are 1-indexed. This is because the first iteration is the
    original un-attacked dataset.
    """

    asr_per_iteration: list[float]
    ifs_per_decile: list[int | None]


def compute_iterations_for_success(
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> IFSMetricResults:
    """Computes the robustness metric for the attack.

    Args:
        attack_out: The AttackOutput object from the attack.
        success_callback: The callback to use to evaluate the attack.
        model: The model to evaluate the attack on.

    Returns:
        An object containing the ASRs for each iteration of the attack, and the
        iteration number at which the ASR crosses each decile. The decile is None
        if the ASR never crosses that threshold.
    """
    dataset = attack_out.dataset
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None
    logits = attack_out.attack_data.logits  # [n_val, n_its, n_labels]
    if dataset.inference_type == InferenceType.CLASSIFICATION and logits is not None:
        results = compute_iterations_for_success_from_logits(
            attack_out=attack_out,
        )
    else:
        results = compute_iterations_for_success_from_text(
            attack_out=attack_out,
            success_callback=success_callback,
            model=model,
        )
    return results


def compute_iterations_for_success_from_text(
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> IFSMetricResults:
    asrs = compute_asr_per_iteration_from_text(
        attack_out=attack_out,
        success_callback=success_callback,
        model=model,
    )
    results = compute_iterations_for_success_from_asrs(asrs)
    return results


def compute_iterations_for_success_from_logits(
    attack_out: AttackOutput,
) -> IFSMetricResults:
    assert attack_out.attack_data is not None
    asrs = compute_asr_per_iteration_from_logits(attack_out)
    results = compute_iterations_for_success_from_asrs(asrs)
    return results


def compute_iterations_for_success_from_asrs(
    asrs: ASRMetricResults,
) -> IFSMetricResults:
    robustness_metric_deciles: list[int | None] = [None] * 11

    # Special handling for 'iteration' 0, which is the un-attacked inputs.
    robustness_metric_deciles[0] = 0

    for iteration, asr in enumerate(asrs.asr_per_iteration):
        for decile in range(11):
            if robustness_metric_deciles[decile] is None and asr >= decile / 10:
                # We don't have to add 1 any longer because the ASR computation
                # handles that.
                robustness_metric_deciles[decile] = iteration

    results = IFSMetricResults(
        asr_per_iteration=asrs.asr_per_iteration,
        ifs_per_decile=robustness_metric_deciles,
    )
    return results


def compute_ifs_metric_from_wandb(group_name: str, run_index: str) -> IFSMetricResults:
    run = get_run_from_index(group_name, run_index)
    return compute_ifs_metric_from_wandb_run(run)


def compute_ifs_metric_from_wandb_run(run: RunInfo) -> IFSMetricResults:
    attack_output = get_attack_output_from_wandb_run(run)

    tic = time.perf_counter()
    results = compute_iterations_for_success_from_logits(attack_output)
    toc = time.perf_counter()
    print(f"Computed Iterations For Success metric in {toc - tic:.2f} seconds")
    return results


def compute_ifs_in_thread(run: RunInfo) -> tuple[IFSMetricResults, int, int, int]:
    """Callable for ThreadPoolExecutor to compute ifs in parallel.

    Args:
        run: The WandbRun object

    Returns:
        A tuple containing the IFS results, the model size, the ft/adv
        training seed, and the adv training round.

    """
    ifs = compute_ifs_metric_from_wandb_run(run)
    model_size, seed = get_model_size_and_seed_from_run(run)
    adv_training_round = get_adv_training_round_from_eval_run(run)
    return ifs, model_size, seed, adv_training_round


def compute_all_ifs_metrics(
    group_name, max_workers: int = 2, debug_n_runs: int = -1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the IFS for all runs in a group.

    Args:
        group_name: The wandb group name.
        max_workers: The number of threads to use.
        debug_n_runs: If >= 0, only process this many runs.

    Returns:
        Dataframes with columns for model index, model size, seed index, adv
        training round, and (decile, IFS) for IFS, (iteration, ASR) for ASR.
    """
    wandb_runs_by_index = get_wandb_runs_by_index(group_name)
    # Sort runs by name (which implicitly sorts by index)
    wandb_runs_by_index_tuples = sorted(wandb_runs_by_index.items())
    if debug_n_runs >= 0:
        # DEBUG: just trying a few.
        wandb_runs_by_index_tuples = wandb_runs_by_index_tuples[:debug_n_runs]
    asr_data = []
    ifs_data = []
    asr_cache_dir = Path("~/.cache/rllm/asr_csvs").expanduser()
    ifs_cache_dir = Path("~/.cache/rllm/ifs_csvs").expanduser()
    asr_cache_dir.mkdir(parents=True, exist_ok=True)
    ifs_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load cached results first
    debug_suffix = f"_debug_{debug_n_runs}" if debug_n_runs >= 0 else ""
    already_computed = []
    for run_index, run in wandb_runs_by_index_tuples:
        asr_cache_path = asr_cache_dir / f"{group_name}_{run_index}{debug_suffix}.csv"
        ifs_cache_path = ifs_cache_dir / f"{group_name}_{run_index}{debug_suffix}.csv"
        if asr_cache_path.exists():
            asr_df = pd.read_csv(asr_cache_path)
            asr_data.append(asr_df)
            ifs_df = pd.read_csv(ifs_cache_path)
            ifs_data.append(ifs_df)
            already_computed.append(run_index)

    wandb_runs_by_index_tuples = [
        (run_index, run)
        for run_index, run in wandb_runs_by_index_tuples
        if run_index not in already_computed
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(compute_ifs_in_thread, run): run_index
            for run_index, run in wandb_runs_by_index_tuples
        }

        for future in concurrent.futures.as_completed(future_to_index):
            print(f"Processing future for {group_name}_{future_to_index[future]}")
            future.exception()
            try:
                ifs_results, model_size, seed, adv_training_round = future.result()
            except Exception as e:
                print(f"Exception: {e}")
                # Cancel remaining futures and exit as gracefully as possible
                for potential_future in future_to_index:
                    potential_future.cancel()
                raise e
            ifs_df = pd.DataFrame(
                {
                    "ifs": ifs_results.ifs_per_decile,
                    "decile": [i for i in range(len(ifs_results.ifs_per_decile))],
                    "model_size": model_size,
                    "seed_idx": seed,
                    "adv_training_round": adv_training_round,
                }
            )
            asr_df = pd.DataFrame(
                {
                    "asr": ifs_results.asr_per_iteration,
                    "iteration": [i for i in range(len(ifs_results.asr_per_iteration))],
                    "model_size": model_size,
                    "seed_idx": seed,
                    "adv_training_round": adv_training_round,
                }
            )

            # Cache the results
            run_index = future_to_index[future]
            with asr_cache_dir / f"{group_name}_{run_index}{debug_suffix}.csv" as f:
                asr_df.to_csv(f, index=False)
            with ifs_cache_dir / f"{group_name}_{run_index}{debug_suffix}.csv" as f:
                ifs_df.to_csv(f, index=False)
            print(f"Cached IFS for {group_name}_{future_to_index[future]}")

            asr_data.append(asr_df)
            ifs_data.append(ifs_df)

    asr_df = pd.concat(asr_data)
    ifs_df = pd.concat(ifs_data)
    # Add model index based on model size
    asr_df = add_model_idx_inplace(asr_df, reference_col="model_size")
    ifs_df = add_model_idx_inplace(ifs_df, reference_col="model_size")
    return asr_df, ifs_df


def main():
    parser = argparse.ArgumentParser(description="Iterations for Success metric")
    parser.add_argument("group_name", type=str, help="wandb group name")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of threads to use.",
    )
    args = parser.parse_args()

    tic = time.perf_counter()
    asr_df, ifs_df = compute_all_ifs_metrics(args.group_name, args.max_workers)
    toc = time.perf_counter()
    print(f"Computed Iterations for Success metric in {toc - tic:.2f} seconds")

    ifs_path = f"outputs/ifs_{args.group_name}.csv"
    print(f"Saving to {ifs_path}")
    ifs_df.to_csv(ifs_path, index=False)

    asr_path = f"outputs/asr_{args.group_name}.csv"
    print(f"Saving to {asr_path}")
    asr_df.to_csv(asr_path, index=False)


if __name__ == "__main__":
    main()
