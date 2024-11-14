"""Uploads models saved on disk to the Hugging Face Hub and deletes them.

The script processes all models stored as
`/robust_llm_data/models/<model_name>/<revision>/<version>/`. Upload or
deletion is skipped if the directory contains a file named `skip-upload` or
`skip-delete`.

Text files logging the uploads/deletions are stored in
/robust_llm_data/upload-info/<timestamp>/.
"""

import argparse
import datetime
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm
from transformers.utils.logging import disable_progress_bar

from robust_llm.dist_utils import dist_rmtree
from robust_llm.models.model_disk_utils import (
    get_model_versions_by_recency,
    get_now_timestamp_str,
    timestamp_str_to_datetime,
)

disable_progress_bar()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/robust_llm_data")
MODELS_DIR = PROJECT_DIR / "models"
RECORD_DIR = PROJECT_DIR / "upload-info"

PROJECT_HF_PREFIX = "AlignmentResearch/robust_llm_"
WRITE_TIMESTAMP_FILE = "disk-write-timestamp.txt"
UPLOAD_TIMESTAMP_FILE = "hf-upload-timestamp.txt"

# Avoid overwriting data from other threads.
FILE_LOCKS = {
    # The contents are mostly just lists of file paths, but failure_upload.txt
    # contains (path, error message) tuples.
    "delete_revision_dir.txt": threading.Lock(),
    "delete_model_dir.txt": threading.Lock(),
    "success_upload.txt": threading.Lock(),
    "failure_upload.txt": threading.Lock(),
    "failure_untracked.txt": threading.Lock(),
    "interrupted.txt": threading.Lock(),
}


def write_to_file(record_dir: Path, file_name: str, content: str):
    with FILE_LOCKS[file_name]:
        with (record_dir / f"{file_name}").open("a") as f:
            f.write(f"{content}\n")


def is_already_on_hub(api: HfApi, repo_id: str, revision: str) -> bool:
    return api.file_exists(repo_id, "config.json", revision=revision)


def get_hf_timestamp(api: HfApi, repo_id: str, revision: str) -> datetime.datetime:
    """Gets timestamp of a model revision that is on HF."""
    # For our purposes we're interested in getting the disk-write timestamp of
    # the model. If that doesn't exist, then we fall back to the latest commit
    # time.
    if api.file_exists(repo_id, WRITE_TIMESTAMP_FILE, revision=revision):
        timestamp_file = Path(
            api.hf_hub_download(
                repo_id,
                WRITE_TIMESTAMP_FILE,
                revision=revision,
                cache_dir="/tmp",
                local_dir="/tmp",
            )
        )
        return timestamp_str_to_datetime(timestamp_file.read_text())
    timestamp = api.repo_info(repo_id, revision=revision).last_modified
    assert timestamp is not None
    return timestamp


def is_already_uploaded(version_path) -> bool:
    return (version_path / UPLOAD_TIMESTAMP_FILE).exists()


def should_version_be_uploaded(version_path: Path) -> bool:
    """Returns true if a version of a model should be uploaded."""
    return (
        (version_path / "done-saving").exists()
        and (not is_already_uploaded(version_path))
        and (not (version_path / "skip-upload").exists())
    )


def should_version_be_deleted(version_path: Path) -> bool:
    """Returns true if a version of a model should be deleted."""
    return (not (version_path / "skip-delete").exists()) and is_already_uploaded(
        version_path
    )


def upload_one_model_version(
    api: HfApi, repo_id: str, revision: str, version_path: Path
) -> None:
    """Upload one version of a model revision to HF Hub."""
    RETRIES = 1
    for i in range(RETRIES + 1):
        try:
            # api.create_repo() throws an exception if two threads processing
            # revisions of the same model both try to create a new repo for the
            # model concurrently.
            api.create_repo(repo_id, exist_ok=True)
            api.create_branch(repo_id, branch=revision, exist_ok=True)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=version_path,
                commit_message=f"Upload from {version_path} by {__file__}",
                revision=revision,
                create_pr=False,
            )
            with open(version_path / UPLOAD_TIMESTAMP_FILE, "w") as f:
                f.write(get_now_timestamp_str())
            return
        except Exception as e:
            if i < RETRIES:
                time.sleep(15 * (2**i))
            else:
                raise e


def process_revision(
    record_dir: Path,
    delete_empty_dirs: bool,
    dry_run: bool,
    revision_path: Path,
) -> bool:
    """Uploads versions of a model revision to HF Hub and deletes them from disk.

    Raises:
        KeyboardInterrupt: Re-raised if a keyboard interrupt is detected.
    """
    repo_id = PROJECT_HF_PREFIX + revision_path.parent.name
    revision = revision_path.name

    api = HfApi()
    success = True
    try:
        # We upload versions in chronological order so that the most recent
        # commit is the most recent version of the model.
        versions = reversed(list(get_model_versions_by_recency(revision_path)))
        for version in versions:
            version_path_str = str(version)

            if is_already_on_hub(api, repo_id, revision):
                hf_last_modified = get_hf_timestamp(api, repo_id, revision)
                version_timestamp = timestamp_str_to_datetime(
                    (version / WRITE_TIMESTAMP_FILE).read_text()
                )
                if hf_last_modified >= version_timestamp:
                    # We do not upload the model in this case since it will mess up
                    # the order of commits on HF Hub. This model revision will
                    # require manual intervention to resolve.
                    error_message = (
                        f"HF Hub version is newer than disk version {version}"
                        f" ({hf_last_modified} >= {version_timestamp})"
                    )
                    logger.error(error_message)
                    write_to_file(
                        record_dir,
                        "failure_upload.txt",
                        f"{version_path_str},{error_message}",
                    )
                    success = False
                    break

            if should_version_be_uploaded(version):
                try:
                    if not dry_run:
                        upload_one_model_version(api, repo_id, revision, version)
                    write_to_file(
                        record_dir, "success_upload.txt", str(version_path_str)
                    )
                except Exception as e:
                    logger.error(f"Failed to upload {version}: {e}")
                    write_to_file(
                        record_dir, "failure_upload.txt", f"{version_path_str},{e}"
                    )
                    success = False
                    # If we fail to upload a version then we should not try to
                    # upload further versions because that will mess up the order of
                    # commits on HF Hub.
                    break

            if should_version_be_deleted(version):
                if not dry_run:
                    # Deleting the directory is a race if some job is currently
                    # loading it from disk. Hopefully the job just throws an
                    # exception and restarts in this case.
                    dist_rmtree(version)

        if delete_empty_dirs:
            if not any(revision_path.iterdir()):
                if not dry_run:
                    revision_path.rmdir()
                write_to_file(record_dir, "delete_revision_dir.txt", str(revision_path))

    except KeyboardInterrupt:
        write_to_file(record_dir, "interrupted.txt", str(revision_path))
        raise
    except Exception as e:
        logger.error(
            f"An error occurred when uploading or deleting {revision_path}: {e}"
        )
        write_to_file(record_dir, "failure_untracked.txt", str(revision_path))
        success = False

    return success


def get_revisions(dirs_file: Path | None) -> list[Path]:
    """Returns a list of directories that look like model revisions."""
    if dirs_file is not None:
        return [Path(line.strip()) for line in open(dirs_file, "r").readlines()]
    # Expected path is of models to upload MODELS_DIR/model/revision/, as given
    # by model_disk_utils.generate_model_save_path()
    dirs = []
    for model in MODELS_DIR.iterdir():
        if not model.is_dir():
            continue
        for revision in model.iterdir():
            if not revision.is_dir():
                continue
            dirs.append(revision)
    return dirs


def main(
    max_workers: int,
    record_dir: Path,
    dirs_file: Path | None,
    delete_empty_dirs: bool,
    dry_run: bool,
):
    num_successes = 0
    num_failures = 0
    revisions = get_revisions(dirs_file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {
            executor.submit(
                process_revision,
                record_dir=record_dir,
                delete_empty_dirs=delete_empty_dirs,
                dry_run=dry_run,
                revision_path=revision,
            ): revision
            for revision in revisions
        }

        try:
            with tqdm(
                total=len(revisions), desc="Processing revisions", unit="revision"
            ) as pbar:
                for future in as_completed(future_to_dir):
                    try:
                        success = future.result()
                        if success:
                            num_successes += 1
                        else:
                            num_failures += 1
                    except Exception as e:
                        logger.error(f"An error occurred: {e}")
                        num_failures += 1

                    pbar.update(1)
                    pbar.set_postfix(successful=num_successes, failed=num_failures)

        except KeyboardInterrupt:
            logger.error("\nKeyboard interrupt detected. Stopping uploads...")

        finally:
            for future in future_to_dir:
                future.cancel()

    if delete_empty_dirs:
        # Delete empty parent directories outside of the thread pool because it's a
        # race if two threads are trying to delete the same directory.
        models = set(revision.parent for revision in revisions)
        for model in tqdm(models, desc="Deleting empty directories", unit="model"):
            if model.exists() and not any(model.iterdir()):
                if not dry_run:
                    model.rmdir()
                write_to_file(record_dir, "delete_model_dir.txt", str(model))

    logger.info(f"Successfully processed {num_successes} models")
    logger.info(f"Failed to process {num_failures} models")

    # Log how many uploads and deletions were done.
    for file_name in FILE_LOCKS:
        with FILE_LOCKS[file_name]:
            path = record_dir / f"{file_name}"
            if not path.exists():
                continue
            num_lines = sum(1 for line in path.open())
            logger.info(f"{file_name}: {num_lines} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--dirs-file",
        type=Path,
        help="File listing paths to revisions that should be processed",
    )
    parser.add_argument(
        "--delete-empty-dirs",
        action="store_true",
        # By default we don't delete empty parent directories because it's a
        # race if a concurrent training run is adding a new subdirectory.
        help=(
            "Delete empty parent directories after deleting models"
            " (off by default to avoid races with concurrent jobs)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually upload or delete anything",
    )
    args = parser.parse_args()

    record_dir = (
        RECORD_DIR / f"{get_now_timestamp_str()}{'-dry-run' if args.dry_run else ''}"
    )
    record_dir.mkdir(parents=True)
    main(
        max_workers=args.max_workers,
        record_dir=record_dir,
        dirs_file=args.dirs_file,
        delete_empty_dirs=args.delete_empty_dirs,
        dry_run=args.dry_run,
    )
