import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd
import torch

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
    get_cache_root,
    get_model_size_and_seed_from_run,
    get_run_from_index,
    get_wandb_runs_by_index,
)


@dataclass(frozen=True)
class LogProbMetricResults:
    mean_log_probs: list[float]
    log_mean_probs: list[float]


def compute_log_prob_metrics(
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> LogProbMetricResults:
    """Computes the smooth log-prob metrics for the attack.

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
        results = compute_log_prob_metrics_from_logits(
            attack_out=attack_out,
        )
    else:
        results = compute_log_prob_metrics_from_text(
            attack_out=attack_out,
            success_callback=success_callback,
            model=model,
        )
    return results


def compute_log_prob_metrics_from_logits(
    attack_out: AttackOutput,
) -> LogProbMetricResults:
    device = "cpu"
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None
    assert attack_out.attack_data.logits is not None
    logits = torch.tensor(
        attack_out.attack_data.logits, device=device
    )  # [n_val, n_its, n_labels]

    # Somewhat hacky way to get the number of iterations
    n_its = len(attack_out.attack_data.iteration_texts[0])

    mean_log_probs = []
    log_mean_probs = []

    for iteration in range(n_its):
        ds = _dataset_for_iteration(attack_out, iteration)
        clf_labels = torch.tensor(ds["clf_label"], device=device)
        iteration_logits = logits[:, iteration, :]  # [n_val, n_labels]
        iteration_probs = iteration_logits.softmax(dim=-1)
        iteration_logprobs = iteration_logits.log_softmax(dim=-1)

        gathered_probs = torch.gather(
            iteration_probs, dim=1, index=clf_labels.view(-1, 1)
        ).squeeze()
        gathered_logprobs = torch.gather(
            iteration_logprobs, dim=1, index=clf_labels.view(-1, 1)
        ).squeeze()

        mean_log_prob = gathered_logprobs.mean().item()
        log_mean_prob = torch.log(gathered_probs.mean()).item()
        log_mean_probs.append(log_mean_prob)
        mean_log_probs.append(mean_log_prob)

        # Print the metrics for this iteration every roughly 10% of the way
        logging_step = max(1, n_its // 10)
        if iteration % logging_step == 0:
            logger.info(
                f"Mean log prob for iteration {iteration+1}: {mean_log_prob}. "
                f"Log mean prob for iteration {iteration+1}: {log_mean_prob}"
            )

    assert len(mean_log_probs) == len(log_mean_probs) == n_its
    results = LogProbMetricResults(
        mean_log_probs=mean_log_probs, log_mean_probs=log_mean_probs
    )
    return results


def compute_log_prob_metrics_from_text(
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> LogProbMetricResults:
    device = "cpu"
    dataset = attack_out.dataset
    # We use dataset_to_attack so that we use the same examples as in attacked_dataset
    original_input_data = model.maybe_apply_user_template(dataset.ds["text"])
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None

    # Somewhat hacky way to get the number of iterations
    n_its = len(attack_out.attack_data.iteration_texts[0])

    mean_log_probs = []
    log_mean_probs = []
    # TODO(Oskar): prepend the metrics on the clean data here

    for iteration in range(n_its):
        ds = _dataset_for_iteration(attack_out, iteration)
        clf_labels = torch.tensor(ds["clf_label"], device=device)

        iteration_in = CallbackInput(
            input_data=ds["text"],
            original_input_data=original_input_data,
            clf_label_data=ds["clf_label"],
            gen_target_data=ds["gen_target"],
        )
        iteration_out = success_callback(model, iteration_in)

        iteration_logits = iteration_out.info.get("logits")  # [n_val, n_labels]
        assert iteration_logits is not None

        iteration_probs = iteration_logits.softmax(dim=-1)
        iteration_logprobs = iteration_logits.log_softmax(dim=-1)

        gathered_probs = torch.gather(
            iteration_probs, dim=1, index=clf_labels.view(-1, 1)
        ).squeeze()
        gathered_logprobs = torch.gather(
            iteration_logprobs, dim=1, index=clf_labels.view(-1, 1)
        ).squeeze()

        mean_log_prob = gathered_logprobs.mean().item()
        log_mean_prob = torch.log(gathered_probs.mean()).item()
        log_mean_probs.append(log_mean_prob)
        mean_log_probs.append(mean_log_prob)

        # Print the metrics for this iteration every roughly 10% of the way
        logging_step = max(1, n_its // 10)
        if iteration % logging_step == 0:
            logger.info(
                f"Mean log prob for iteration {iteration+1}: {mean_log_prob}. "
                f"Log mean prob for iteration {iteration+1}: {log_mean_prob}"
            )

    assert len(mean_log_probs) == len(log_mean_probs) == n_its
    results = LogProbMetricResults(
        mean_log_probs=mean_log_probs, log_mean_probs=log_mean_probs
    )
    return results


@print_time()
def compute_logprob_metric_from_wandb_run(run: RunInfo) -> LogProbMetricResults:
    attack_output = get_attack_output_from_wandb_run(run)

    results = compute_log_prob_metrics_from_logits(attack_output)
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
    logprob_cache_dir = get_cache_root() / "logprob_csvs"
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
