import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

from robust_llm import logger
from robust_llm.attacks.attack import AttackOutput
from robust_llm.metrics.metric_utils import (
    _dataset_for_iteration,
    get_attack_output_from_wandb_run,
)
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.plotting_utils.utils import add_model_idx_inplace
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    BinaryCallback,
    CallbackInput,
)
from robust_llm.utils import print_time
from robust_llm.wandb_utils.wandb_api_tools import (
    RunInfo,
    get_adv_training_round_from_eval_run,
    get_model_size_and_seed_from_run,
    get_run_from_index,
    get_save_root,
    get_wandb_runs_by_index,
)


@dataclass(frozen=True)
class LogProbMetricResults:
    mean_log_probs: list[float]
    log_mean_probs: list[float]
    mean_logits: list[float]


def get_logits_for_iteration(
    iteration: int,
    ds: Dataset,
    logits: torch.Tensor | None,
    original_input_data: list[str] | None,
    success_callback: BinaryCallback | None = None,
    model: WrappedModel | None = None,
) -> torch.Tensor:
    """Get the logits for a specific iteration.

    Handles the two cases where we either have the logits stored in the attack_out
    object, or we need to compute them from the attacked data using the
    success_callback.

    Args:
        iteration: The iteration for which to get the logits.
        ds: The dataset for the iteration.
        logits: The [n_val, n_its, n_labels] logits stored in the attack_out object, if
            available.
        original_input_data: The [n_val] original input data, if we need to compute the
            logits.
        success_callback: The callback to use to get logits from the attacked data if
            we didn't store the logits.
        model: The model to use to get logits on the attacked data if we didn't store
            the logits.

    Returns:
        The logits for the iteration, with shape [n_val, n_labels].
    """

    if logits is not None:
        iteration_logits = logits[:, iteration, :]  # [n_val, n_labels]
    else:
        assert model is not None and success_callback is not None
        iteration_in = CallbackInput(
            input_data=ds["text"],
            original_input_data=original_input_data,
            clf_label_data=ds["clf_label"],
            gen_target_data=ds["gen_target"],
        )
        iteration_out = success_callback(model, iteration_in)
        assert isinstance(iteration_out.info.get("logits"), torch.Tensor)
        iteration_logits = iteration_out.info["logits"]  # [n_val, n_labels]
    return iteration_logits


def compute_log_prob_metrics(
    attack_out: AttackOutput,
    success_callback: BinaryCallback | None = None,
    model: WrappedModel | None = None,
) -> LogProbMetricResults:
    """Compute log prob metrics from raw logits on attacked data.

    Case 1: If the logits are stored in the attack_out object, we use them.
    These are stored for every datapoint at every iteration, so have
    shape [n_val, n_its, n_labels].

    Case 2: If the logits are not stored, we use the success_callback and
    model to compute the logits from the attacked data.

    In either case, once we have the logits, we want to compute *three* metrics
    for each iteration, returning three lists of length [n_its].

    1) Mean log prob: index into the correct label, and then take the mean over
    n_val logprobs. This is the average log probability of the correct label.

    2) Log mean prob: index into the correct label, and then take the log of the
    mean over n_val probabilities. This is the log of the average probability
    of the correct label.

    3) Mean logit: take the difference in logprobs between the correct and
    incorrect labels (this is the "logit" function), and then take the mean over
    n_val. This is the average logit.

    Args:
        attack_out: An object containing either the logits or failing that,
            the attacked dataset from which we can recompute the logits.
        success_callback: The callback to use to get logits from the
            attacked data if we didn't store the logits.
        model: The model to use to get logits on the attacked data if
            we didn't store the logits.
    """
    assert (
        attack_out.dataset.inference_type == InferenceType.CLASSIFICATION
    ), "TODO(Oskar): Implement log prob metrics for generative tasks"
    device = "cpu"
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None

    # Prepare logits, or failing that, the raw inputs
    if attack_out.attack_data.logits is not None:
        logits = torch.tensor(
            attack_out.attack_data.logits, device=device
        )  # [n_val, n_its, n_labels]
        original_input_data = None
    else:
        assert model is not None
        logits = None
        original_input_data = model.maybe_apply_user_template(
            attack_out.dataset.ds["text"]
        )

    # Somewhat hacky way to get the number of iterations
    n_its = len(attack_out.attack_data.iteration_texts[0])

    mean_log_probs = []
    log_mean_probs = []
    mean_logits = []

    for iteration in range(n_its):
        ds = _dataset_for_iteration(attack_out, iteration)
        clf_labels = torch.tensor(ds["clf_label"], device=device)
        iteration_logits = get_logits_for_iteration(
            iteration,
            ds,
            logits,
            original_input_data,
            success_callback,
            model,
        )
        iteration_probs = iteration_logits.softmax(dim=-1)
        iteration_logprobs = iteration_logits.log_softmax(dim=-1)

        gathered_probs = torch.gather(
            iteration_probs, dim=1, index=clf_labels.view(-1, 1)
        ).squeeze()
        gathered_logprobs = torch.gather(
            iteration_logprobs, dim=1, index=clf_labels.view(-1, 1)
        ).squeeze()
        complement_logprobs = torch.gather(
            iteration_logprobs, dim=1, index=(1 - clf_labels).view(-1, 1)
        ).squeeze()

        mean_log_prob = gathered_logprobs.mean().item()
        log_mean_prob = torch.log(gathered_probs.mean()).item()
        mean_logit = (gathered_logprobs - complement_logprobs).mean().item()

        log_mean_probs.append(log_mean_prob)
        mean_log_probs.append(mean_log_prob)
        mean_logits.append(mean_logit)

        # Print the metrics for this iteration every roughly 10% of the way
        logging_step = max(1, n_its // 10)
        if iteration % logging_step == 0:
            logger.info(
                f"Log prob metrics for iteration {iteration+1}: "
                f"{mean_log_prob=} {log_mean_prob=} {mean_logit=}"
            )

    assert len(mean_log_probs) == len(log_mean_probs) == n_its
    results = LogProbMetricResults(
        mean_log_probs=mean_log_probs,
        log_mean_probs=log_mean_probs,
        mean_logits=mean_logits,
    )
    return results


@print_time()
def compute_logprob_metric_from_wandb_run(run: RunInfo) -> LogProbMetricResults:
    attack_output = get_attack_output_from_wandb_run(run)

    results = compute_log_prob_metrics(attack_output)
    return results


def compute_logprob_metric_from_wandb(
    group_name: str, run_index: str
) -> LogProbMetricResults:
    run = get_run_from_index(group_name, run_index)
    return compute_logprob_metric_from_wandb_run(run)


def compute_logprob_in_thread(
    run: RunInfo,
) -> tuple[LogProbMetricResults, int, int, int]:
    """Callable for ThreadPoolExecutor to compute logprobs in parallel.

    Args:
        run: The WandbRun object

    Returns:
        A tuple containing the logprob results, the model size, the ft/adv
        training seed, and the adv training round.

    """
    logprobs = compute_logprob_metric_from_wandb_run(run)
    model_size, seed = get_model_size_and_seed_from_run(run)
    adv_training_round = get_adv_training_round_from_eval_run(run)
    return logprobs, model_size, seed, adv_training_round


@print_time()
def compute_all_logprob_metrics(
    group_name, max_workers: int = 2, debug_n_runs: int = -1
) -> pd.DataFrame:
    """Compute the mean log prob for all runs in a group.

    Args:
        group_name: The wandb group name.
        max_workers: The number of threads to use.
        debug_n_runs: If >= 0, only process this many runs.

    Returns:
        Dataframes with columns for model index, model size, seed index, adv
        training round, log mean prob and mean log prob.
    """
    wandb_runs_by_index = get_wandb_runs_by_index(group_name)
    wandb_runs_by_index = {
        i: run
        for i, run in wandb_runs_by_index.items()
        if run.summary.get("really_finished")
    }
    # Sort runs by name (which implicitly sorts by index)
    wandb_runs_by_index_tuples = sorted(wandb_runs_by_index.items())
    if debug_n_runs >= 0:
        # DEBUG: just trying a few.
        wandb_runs_by_index_tuples = wandb_runs_by_index_tuples[:debug_n_runs]
    logprob_data = []
    logprob_cache_dir = Path(get_save_root()) / "logprob_csvs"
    logprob_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load cached results first
    debug_suffix = f"_debug_{debug_n_runs}" if debug_n_runs >= 0 else ""
    already_computed = []
    for run_index, run in wandb_runs_by_index_tuples:
        logprob_cache_path = (
            logprob_cache_dir / f"{group_name}_{run_index}{debug_suffix}.csv"
        )
        if logprob_cache_path.exists():
            logprob_df = pd.read_csv(logprob_cache_path)
            logprob_data.append(logprob_df)
            already_computed.append(run_index)

    wandb_runs_by_index_tuples = [
        (run_index, run)
        for run_index, run in wandb_runs_by_index_tuples
        if run_index not in already_computed
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(compute_logprob_in_thread, run): run_index
            for run_index, run in wandb_runs_by_index_tuples
        }

        for future in concurrent.futures.as_completed(future_to_index):
            logger.debug(
                f"Processing future for {group_name}_{future_to_index[future]}"
            )
            future.exception()
            try:
                logprob_results, model_size, seed, adv_training_round = future.result()
            except Exception as e:
                logger.error(f"Exception: {e}")
                # Cancel remaining futures and exit as gracefully as possible
                for potential_future in future_to_index:
                    potential_future.cancel()
                raise e
            logprob_df = pd.DataFrame(
                {
                    "log_mean_prob": logprob_results.log_mean_probs,
                    "mean_log_prob": logprob_results.mean_log_probs,
                    "mean_logit": logprob_results.mean_logits,
                    "iteration": [
                        i for i in range(len(logprob_results.log_mean_probs))
                    ],
                    "model_size": model_size,
                    "seed_idx": seed,
                    "adv_training_round": adv_training_round,
                }
            )

            # Cache the results
            run_index = future_to_index[future]
            with logprob_cache_dir / f"{group_name}_{run_index}{debug_suffix}.csv" as f:
                logprob_df.to_csv(f, index=False)
            logger.debug(f"Cached logprobs for {group_name}_{future_to_index[future]}")

            logprob_data.append(logprob_df)

    logprob_df = pd.concat(logprob_data)
    # Add model index based on model size
    logprob_df = add_model_idx_inplace(logprob_df, reference_col="model_size")
    return logprob_df


def main():
    parser = argparse.ArgumentParser(description="Log Prob metric")
    parser.add_argument("group_name", type=str, help="wandb group name")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of threads to use.",
    )
    parser.add_argument(
        "--debug_n_runs",
        type=int,
        default=-1,
        help="If >= 0, only process this many runs.",
    )
    args = parser.parse_args()

    logprob_df = compute_all_logprob_metrics(
        args.group_name, args.max_workers, debug_n_runs=args.debug_n_runs
    )

    logprob_path = f"outputs/logprob_{args.group_name}.csv"
    logger.info(f"Saving to {logprob_path}")
    logprob_df.to_csv(logprob_path, index=False)
