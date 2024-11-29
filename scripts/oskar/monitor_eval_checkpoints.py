"""Monitor checkpoints using `/robust_llm_data/` mount"""

import argparse
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_PATH = "/robust_llm_data/checkpoints"

# Paths look like /robust_llm_data/checkpoints/<group>/<run>/<hash>/attack_states/


@lru_cache(maxsize=1024)
def list_dir(path: str) -> list[str]:
    """Cache directory listings to avoid repeated filesystem access."""
    try:
        return sorted(os.listdir(path), reverse=True)
    except (FileNotFoundError, PermissionError):
        return []


def get_subdirs(path: str) -> list[str]:
    """Get full paths of all subdirectories in a directory."""
    contents = list_dir(path)
    return [
        os.path.join(path, d) for d in contents if os.path.isdir(os.path.join(path, d))
    ]


def find_latest_state(config_dir: str) -> str:
    """Find the latest state file by checking attack states in descending order."""
    last_attack = None
    attack_states_dir = os.path.join(config_dir, "attack_states")

    for example_dir in list_dir(attack_states_dir):
        if example_dir.startswith("example_"):
            example_file = os.path.join(
                attack_states_dir, example_dir, "examples_0.pkl"
            )
            if os.path.isfile(example_file) and last_attack is None:
                return example_file

    return ""


def analyze_checkpoints(group: str) -> pd.DataFrame:
    """Analyze checkpoints for a specific group with optimized filesystem access."""
    results = []
    group_path = os.path.join(BASE_PATH, group)

    if not os.path.exists(group_path):
        return pd.DataFrame()

    # Get all run directories in one go
    run_prefix = group.replace("_", "-")
    run_dirs = [
        d for d in get_subdirs(group_path) if os.path.basename(d).startswith(run_prefix)
    ]

    for run_dir in tqdm(run_dirs):
        run_num = run_dir.split("-")[-1]
        config_dirs = get_subdirs(run_dir)

        for config_dir in config_dirs:
            config_hash = os.path.basename(config_dir)

            # Find latest state file and attack states efficiently
            latest_attack_file = find_latest_state(config_dir)
            if not latest_attack_file:
                continue
            attack_examples = re.search(r"example_(\d+)", latest_attack_file)

            # Get modification times efficiently
            latest_time = os.path.getmtime(latest_attack_file)

            latest_time_str = datetime.fromtimestamp(latest_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            results.append(
                {
                    "group": group,
                    "run": run_num,
                    "config_hash": config_hash,
                    "last_modified": latest_time_str,
                    "attacked_examples": attack_examples,
                    "modified_ts": latest_time,
                }
            )

    return (
        pd.DataFrame(results).sort_values(["run", "config_hash"])
        if results
        else pd.DataFrame()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Monitor checkpoint progress for specified groups"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[
            "oskar_026a_gcg_eval_qwen25_adv_spam",
            "oskar_026b_gcg_eval_qwen25_adv_harmless",
        ],
        help="List of group names to analyze (space-separated)",
    )
    args = parser.parse_args()

    all_results = []
    for group in args.groups:
        results_df = analyze_checkpoints(group)
        if not results_df.empty:
            all_results.append(results_df)
        else:
            print(f"Failed to find checkpoints for {group}.")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df = combined_df.sort_values(["group", "run", "config_hash"])

        print("\nCombined Checkpoint Analysis Results:")
        reference_ts = datetime.now().timestamp()
        combined_df["reference_ts"] = reference_ts
        path = Path(f"outputs/checkpoint_analysis/{reference_ts}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {path}")
        combined_df.to_csv(path, index=False)
    else:
        print("No results found for any group.")


if __name__ == "__main__":
    main()
