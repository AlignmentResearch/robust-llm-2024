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
from itertools import product

import pandas as pd

from robust_llm.attacks.attack import AttackOutput
from robust_llm.metrics.asr_per_iteration import (
    ASRMetricResults,
    compute_asr_per_iteration_from_logits,
    compute_asr_per_iteration_from_text,
)
from robust_llm.metrics.metric_utils import get_attack_output_from_wandb
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks.scoring_callback_utils import BinaryCallback


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
    attack_output = get_attack_output_from_wandb(group_name, run_index)

    tic = time.perf_counter()
    results = compute_iterations_for_success_from_logits(attack_output)
    toc = time.perf_counter()
    print(results)
    print(f"Computed Iterations For Success metric in {toc - tic:.2f} seconds")
    return results


def compute_ifs_in_thread(group_name: str, seed_first_run: int, model_idx):
    """Callable for ThreadPoolExecutor to compute ifs in parallel.

    Args:
        group_name: The wandb group name.
        seed_first_run: The index of the first run for this seed
            (i.e. seed * n_models).
        model_idx: The index of the model.

    Runs are indexed such that runs for a seed are consecutive and
    runs for a model are separated by the number of models.
    e.g. [s0m0, s0m1, ... s0m9, s1m0, ...]
    """
    run_index = str(seed_first_run + model_idx)
    ifs = compute_ifs_metric_from_wandb(group_name, run_index.zfill(4))
    return seed_first_run, model_idx, ifs


def compute_all_ifs_metrics(
    group_name, n_models: int, n_seeds: int, max_workers: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the IFS for all runs in a group.

    Args:
        group_name: The wandb group name.
        n_models: The number of models.
        n_seeds: The number of seeds.
        max_workers: The number of threads to use.

    Returns:
        A dataframe with columns for model index, seed index, decile, and IFS.
    """
    model_indices = list(range(n_models))
    seed_first_runs = [seed_index * n_models for seed_index in range(n_seeds)]

    asr_data = []
    ifs_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(
                compute_ifs_in_thread, group_name, seed_first_run, model_idx
            ): (
                seed_first_run,
                model_idx,
            )
            for seed_first_run, model_idx in product(seed_first_runs, model_indices)
        }

        for future in concurrent.futures.as_completed(future_to_params):
            seed_first_run, model_idx, ifs_results = future.result()
            ifs_df = pd.DataFrame(
                {
                    "model_idx": model_idx,
                    "seed_idx": seed_first_runs.index(seed_first_run),
                    "ifs": ifs_results.ifs_per_decile,
                    "decile": [i for i in range(len(ifs_results.ifs_per_decile))],
                }
            )
            asr_df = pd.DataFrame(
                {
                    "model_idx": model_idx,
                    "seed_idx": seed_first_runs.index(seed_first_run),
                    "asr": ifs_results.asr_per_iteration,
                    "iteration": [i for i in range(len(ifs_results.asr_per_iteration))],
                }
            )
            asr_data.append(asr_df)
            ifs_data.append(ifs_df)

    return pd.concat(asr_data), pd.concat(ifs_data)


def main():
    parser = argparse.ArgumentParser(description="Iterations for Success metric")
    parser.add_argument("group_name", type=str, help="wandb group name")
    parser.add_argument(
        "--n_models", type=int, default=10, help="Number of models (must be accurate)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds (can set artificially small)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of threads to use.",
    )
    args = parser.parse_args()

    tic = time.perf_counter()
    asr_df, ifs_df = compute_all_ifs_metrics(
        args.group_name, args.n_models, args.n_seeds, args.max_workers
    )
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
