"""Monitor checkpoints using `/robust_llm_data/` mount"""

import argparse
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_PATH = "/robust_llm_data/checkpoints"

# Paths look like /robust_llm_data/checkpoints/<group>/<run>/<hash>/epoch_####


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


def find_latest_state(config_dir: str) -> tuple[str, str]:
    """Find the latest state file by checking epochs in descending order."""
    # Get all epoch directories and sort them in reverse order
    entries = list_dir(config_dir)
    epoch_dirs = [e for e in entries if e.startswith("epoch_")]

    # Check each epoch directory in descending order
    last_state = None
    last_attack = None
    for epoch_dir in epoch_dirs:
        int(epoch_dir.split("_")[-1])
        full_epoch_path = os.path.join(config_dir, epoch_dir)
        state_file = os.path.join(full_epoch_path, "state.json")
        attack_states_dir = os.path.join(full_epoch_path, "attack_states")

        if os.path.isdir(attack_states_dir):
            for example_dir in list_dir(attack_states_dir):
                if example_dir.startswith("example_"):
                    example_file = os.path.join(
                        attack_states_dir, example_dir, "examples_0.pkl"
                    )
                    if os.path.isfile(example_file) and last_attack is None:
                        last_attack = example_file
                        break

        if os.path.isfile(state_file) and last_state is None:
            last_state = state_file

        if last_attack is not None and last_state is not None:
            return last_state, last_attack

    return "", ""


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
            latest_state_file, latest_attack_file = find_latest_state(config_dir)
            if not latest_state_file:
                continue
            model_epoch_re = re.search(r"epoch_(\d+)", latest_state_file)
            attack_epoch_re = re.search(r"epoch_(\d+)", latest_attack_file)
            assert (
                model_epoch_re is not None
            ), f"Failed to parse epoch from {latest_state_file}"
            assert (
                attack_epoch_re is not None
            ), f"Failed to parse epoch from {latest_attack_file}"
            attack_epoch = int(attack_epoch_re.group(1))
            model_epoch = int(model_epoch_re.group(1))
            attack_examples = re.search(r"example_(\d+)", latest_attack_file)
            current_examples = int(attack_examples.group(1)) if attack_examples else 0

            try:
                with open(latest_state_file, "r") as f:
                    state_data = json.load(f)

                config_dict = json.loads(state_data["config"])
                total_epochs = config_dict["training"]["adversarial"][
                    "num_adversarial_training_rounds"
                ]
                round_examples = config_dict["training"]["adversarial"][
                    "num_examples_to_generate_each_round"
                ]
                model_name = config_dict["training"]["save_name"]

                # Get modification times efficiently
                latest_time = max(
                    os.path.getmtime(latest_state_file),
                    os.path.getmtime(latest_attack_file),
                )

                latest_time_str = datetime.fromtimestamp(latest_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                completed_epoch_fraction = model_epoch / total_epochs
                total_examples = (total_epochs - 1) * round_examples
                previous_examples = max(0, (attack_epoch - 1) * round_examples)
                cumulative_examples = previous_examples + current_examples
                attacked_fraction = cumulative_examples / total_examples
                progress = max(completed_epoch_fraction, attacked_fraction)

                results.append(
                    {
                        "group": group,
                        "run": run_num,
                        "config_hash": config_hash,
                        "last_modified": latest_time_str,
                        "model_epoch": model_epoch,
                        "attack_epoch": attack_epoch,
                        "total_epochs": total_epochs,
                        "attacked_examples": cumulative_examples,
                        "total_examples": total_examples,
                        "model_name": model_name,
                        "progress": progress,
                        "modified_ts": latest_time,
                    }
                )

            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                print(f"Error processing {latest_state_file}: {str(e)}")
                continue

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
            "oskar_024a_adv_tr_qwen25_gcg_harmless",
            "oskar_024b_adv_tr_qwen25_gcg_spam",
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
        overall_progress = (
            combined_df["attacked_examples"].sum() / combined_df["total_examples"].sum()
        )
        print(f"Overall Progress: {overall_progress:.1%}")
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
