"""Prunes old profile files corresponding to deleted or failed W&B runs."""

import argparse
import logging
from pathlib import Path

import wandb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def should_delete_profile(run_id: str) -> bool:
    try:
        api = wandb.Api()
        run = api.run(f"farai/robust-llm/{run_id}")
        return run.state in ["crashed", "failed"]
    except wandb.errors.CommError as e:  # type: ignore[attr-defined]
        return e.message.startswith("Could not find run")
    except Exception as e:
        logging.error(f"Error checking W&B status for run {run_id}: {e}")
        return False  # In case of other errors, don't delete the profile


def process_profiles(profile_dir: Path, dry_run: bool) -> None:
    for file in profile_dir.iterdir():
        if file.name.endswith(".speedscope.json"):
            run_id = file.name.split("-")[-2]  # Run ID is in the filename
            assert len(run_id) == 8 and run_id.isalnum(), f"Invalid run ID: {run_id}"

            if should_delete_profile(run_id):
                try:
                    if dry_run:
                        logging.info(f"Would delete profile: {file}")
                        continue
                    file.unlink()
                    logging.info(f"Deleted profile: {file}")
                except Exception as e:
                    logging.error(f"Error deleting {file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/robust_llm_data/profiles/"),
        help="Directory containing the profile data files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be deleted without actually deleting them",
    )

    args = parser.parse_args()
    process_profiles(args.directory, args.dry_run)
