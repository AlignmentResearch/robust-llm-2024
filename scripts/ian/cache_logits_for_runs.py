"""Helper script to cache logits from wandb runs for later use."""

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from datasets.utils.logging import disable_progress_bar
from tqdm import tqdm
from wandb.apis.public.runs import Run as WandbRun

from robust_llm.metrics.metric_utils import get_attack_output_from_wandb_run
from robust_llm.wandb_utils.wandb_api_tools import get_wandb_runs


def get_run_index(run: WandbRun) -> str:
    return run.name.split("-")[-1]


# def download_attack_data_tables(run: WandbRun, max_workers: int = 4) -> None:
#     """Downloads attack data tables.

#     This is like wandb_api_tools.get_attack_data_tables, but it doesn't load the
#     data into memory.
#     """
#     artifacts = run.logged_artifacts()
#     filtered_artifacts = []
#     for artifact in artifacts:
#         re_match = re.search(r"attack_dataexample_([\d]+):", artifact.name)
#         if not re_match:
#             continue
#         filtered_artifacts.append(artifact)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_artifact = {
#             executor.submit(
#                 download_attack_data_table_if_not_cached, artifact, run.name
#             ): artifact
#             for artifact in filtered_artifacts
#         }

#         for future in concurrent.futures.as_completed(future_to_artifact):
#             result = future.result()


def main(
    group_names: list[str],
    starting_index: str | None,
    max_threads: int,
    max_subthreads_per_thread: int,
) -> None:
    for group_name in tqdm(group_names):
        unsorted_runs = get_wandb_runs(group_name)
        # Sort runs by name (which implicitly sorts by index)
        runs = sorted(unsorted_runs, key=lambda run: get_run_index(run))
        if starting_index is not None:
            runs = [run for run in runs if get_run_index(run) >= starting_index]

        print(f"Processing {len(runs)} runs for group {group_name}")
        with ThreadPoolExecutor(max_threads) as executor:
            future_to_run = {
                executor.submit(
                    # get_attack_data_tables, *(run, max_subthreads_per_thread)
                    get_attack_output_from_wandb_run,
                    *(run, max_subthreads_per_thread),
                ): run
                for run in runs
            }
        with tqdm(
            total=len(runs),
            desc="Processing runs",
            unit="Run",
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_run):
                future.result()
                print(f"Processed run: {future_to_run[future].name}")
                pbar.update(1)


if __name__ == "__main__":
    disable_progress_bar()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group_names",
        type=str,
        help="The wandb group name to cache logits for, comma separated.",
    )
    parser.add_argument(
        "--starting_index",
        type=str,
        help="(if passing a single run): The run index start from.",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=2,
        help="The number of threads to use to process runs in parallel.",
    )
    parser.add_argument(
        "--max_subthreads_per_thread",
        type=int,
        default=5,
        help="The number of threads to use for downloading in each thread.",
    )
    args = parser.parse_args()
    group_names = [name.strip() for name in args.group_names.split(",")]
    if len(group_names) > 1:
        assert (
            args.starting_index is None
        ), "If passing multiple group names, do not pass starting index"
        starting_index = None
    else:
        starting_index = args.starting_index
    main(group_names, starting_index, args.max_threads, args.max_subthreads_per_thread)
