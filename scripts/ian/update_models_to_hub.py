import argparse
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
from transformers.utils.logging import disable_progress_bar

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel

disable_progress_bar()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

RECORD_DIR = Path("/robust_llm_data/update-info")
RECORD_DIR.mkdir(exist_ok=True, parents=True)


class NFSToHFError(Exception):
    pass


class LogitsMismatchError(NFSToHFError):
    pass


class ModelLoadError(NFSToHFError):
    pass


@dataclass
class FileDetails:
    local_file_path: Path
    repo_id: str
    revision: str
    last_modified: datetime


# Avoid overwriting data from other threads
file_locks = {
    "not_present": threading.Lock(),
    "success_update": threading.Lock(),
    "success_older": threading.Lock(),
    "processed": threading.Lock(),
    "failure_load": threading.Lock(),
    "failure_upload": threading.Lock(),
}


def write_to_file(file_name: str, content: str):
    with file_locks[file_name]:
        with (RECORD_DIR / f"{file_name}.csv").open("a") as f:
            f.write(content)


def update_model_to_hub(
    file_details: FileDetails,
) -> tuple[str, str, str, str, str]:
    """Push a model to the hub if it's more recent than the one on the hub."""

    local_file_path = file_details.local_file_path
    repo_id = file_details.repo_id
    revision = file_details.revision
    last_modified = file_details.last_modified

    api = HfApi()

    line_to_write = f"{str(local_file_path)},{repo_id},{revision}\n"
    status = "success"
    message = ""

    # Check if the branch already exists and contains a config.json before
    # trying to upload again.
    try:
        already_on_hub = api.file_exists(repo_id, "config.json", revision=revision)
        if not already_on_hub:
            write_to_file("not_present", line_to_write)
            raise NFSToHFError("Model not present on HFHub")

        # Only upload if the new model is more recent than the one on the hub
        last_modified_on_hub = api.repo_info(repo_id, revision=revision).last_modified
        assert last_modified_on_hub is not None
        if last_modified_on_hub < last_modified:
            print(
                f"Uploading {local_file_path} to HFHub because"
                " it is more recent than the existing upload."
            )
            input_ids = torch.tensor([[1000, 1001]], requires_grad=False)
            local_logits = verify_model_loading(local_file_path, input_ids)
            if local_logits is None:
                write_to_file("failure_load", line_to_write)
                raise ModelLoadError("Model failed to load")
            upload_model(api, repo_id, revision, local_file_path)
            logits_match = check_model_upload(
                repo_id, revision, input_ids, local_logits
            )
            if not logits_match:
                write_to_file("failure_upload", line_to_write)
                raise LogitsMismatchError(
                    "Logits mismatch between local and uploaded model"
                )

            status = "success"
            message = (
                "Uploaded model successfully:"
                f" {last_modified_on_hub} <= {last_modified}"
            )
            write_to_file("success_update", line_to_write)

        else:
            status = "success"
            message = (
                "Model on HFHub is more recent than local model:"
                f" {last_modified_on_hub} > {last_modified}"
            )
            write_to_file("success_older", line_to_write)

    except KeyboardInterrupt:
        status = "interrupted"
        message = "Upload interrupted by user"
        write_to_file("interrupted", line_to_write)
        raise

    except Exception as e:
        status = "failure"
        message = str(e)
        print("======")
        print(f"Failed to upload {local_file_path} to {repo_id} at {revision}:")
        print(f"{type(e).__name__}: {e}")
        print("======")
        if not isinstance(e, NFSToHFError):
            write_to_file("failure_untracked", line_to_write)

    # Record that we've processed this dir, one way or another
    write_to_file("processed", line_to_write)
    return str(local_file_path), repo_id, revision, status, message


def check_model_upload(
    repo_id: str, revision: str, input_ids, local_logits: torch.Tensor
) -> bool:
    """Download an uploaded model and check that its logits match the local model."""
    # First download the repo to a temp dir and *then* load it, to take
    # advantage of faster download.
    temp_dir = f"/tmp/{repo_id}/{revision}"
    snapshot_download(repo_id, revision=revision, local_dir=temp_dir)
    model = load_model(temp_dir)
    model.eval()
    shutil.rmtree(temp_dir)
    with torch.no_grad():
        remote_logits = model(input_ids=input_ids).logits
    del model
    return torch.allclose(remote_logits, local_logits)


def upload_model(
    api: HfApi, repo_id: str, revision: str, local_file_path: Path
) -> None:
    """Upload a local model to huggingface."""
    api.create_repo(repo_id, exist_ok=True)

    try:
        api.create_branch(repo_id, branch=revision)
    except Exception as e:
        print(f"Can't create branch {revision}: {e}")

    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_file_path,
        commit_message=f"Upload from {local_file_path}",
        revision=revision,
        create_pr=False,
    )


def verify_model_loading(
    local_file_path, input_ids: torch.Tensor
) -> torch.Tensor | None:
    """Try loading a model into memory and getting logits.

    Returns:
        Logits if model loaded and ran successfully, None otherwise
    """
    try:
        model = load_model(local_file_path)
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        del model
        return logits
    except Exception as e:
        print(f"Failed to load {local_file_path}: {e}")
        return None


def load_model(model_name_or_path: Path | str, revision: str = "main") -> WrappedModel:
    """Check if a model loads without error.

    Assumes the model is a "pythia" model.
    """
    model_config = ModelConfig(
        name_or_path=str(model_name_or_path),
        family="pythia",
        revision=revision,
        inference_type="classification",
        env_minibatch_multiplier=1.0,
    )

    return WrappedModel.from_config(model_config, accelerator=None, num_classes=2)


def construct_file_details(model_dir: Path, match: re.Match):
    dataset = match.group("dataset")
    base_model_name = match.group("base_model_name")
    attack = match.group("attack")
    ft_seed = match.group("ft_seed")
    adv_seed = match.group("adv_seed")
    model_name = (
        f"clf_{dataset}_{base_model_name}_s-{ft_seed}_adv_tr_{attack}_t-{adv_seed}"
    )

    repo_id = f"AlignmentResearch/{model_name}"
    revision = f"adv-training-round-{match.group('round')}"
    last_modified = get_latest_modification_time(model_dir)

    return FileDetails(model_dir, repo_id, revision, last_modified)


def get_latest_modification_time(directory: Path) -> datetime:
    latest_time = directory.stat().st_mtime
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            latest_time = max(latest_time, file_path.stat().st_mtime)
    return datetime.fromtimestamp(latest_time, tz=timezone.utc)


def get_dirs_to_skip() -> set[str]:
    """Get the directories that we should skip.

    If retry_failures is off, we only try previously unprocessed directories.
    If retry_failures is on, we try all directories that aren't in
    'success_new.csv' or 'success_exists.csv'.

    If retry_failures is on and skip_logit_mismatch is on, we also skip
    directories that are in 'failure_match.csv'.

    If retry_failures is on and skip_load_failures is on, we also skip
    directories that are in 'failure_load.csv'.
    """
    return get_already_processed_dirs()


def get_dirs_from_file(file_name: Path) -> set[str]:
    dirs = set()
    with file_name.open("r") as f:
        for line in f:
            local_file_path, repo_id, revision = line.strip().split(",")
            dirs.add(local_file_path)
    return dirs


def get_already_processed_dirs() -> set[str]:
    processed_path = RECORD_DIR / "processed.csv"
    with file_locks["processed"]:
        if not processed_path.exists():
            return set()

        return get_dirs_from_file(processed_path)


def get_dirs_to_push(dirs_file: str | None) -> list[FileDetails]:
    base_dir = Path("/robust_llm_data/models")
    if dirs_file is not None:
        dirs = [Path(dir) for dir in get_dirs_from_file(Path(dirs_file))]
    else:
        dirs = [dir for dir in base_dir.iterdir()]
    # Run a regex on directories in base_dir to get ones that match
    # clf_{dataset}_{model_name}_s-{ft_seed}_adv_tr_{attack}_{adv_seed}{-?}adv-training-round-{round}
    dir_pattern = r"""
^clf_
(?P<dataset>[a-zA-Z0-9]+)_
(?P<base_model_name>[a-zA-Z0-9-.]+)_
s-(?P<ft_seed>\d+)_
adv_tr_
(?P<attack>[a-zA-Z0-9]+)_
t-(?P<adv_seed>\d+)
(?P<optional_hyphen>-?)
adv-training-round-
(?P<round>\d+)$
"""
    dir_regex = re.compile(dir_pattern, re.VERBOSE)

    push = []
    no_push = []
    for model_dir in dirs:
        match = dir_regex.match(model_dir.name)
        if match is None:
            no_push.append(model_dir)
            continue
        file_details = construct_file_details(model_dir, match)
        push.append(file_details)
    print(f"Will push {len(push)} models and not push {len(no_push)} models")
    return push


def main(
    max_workers: int,
    num_to_process: int | None,
    dirs_file: str | None,
):
    successes = []
    failures = []
    dirs_to_push = get_dirs_to_push(dirs_file)
    if dirs_file is not None:
        dirs_to_skip = set()
    else:
        dirs_to_skip = get_dirs_to_skip()

    still_to_push = [
        details
        for details in dirs_to_push
        if str(details.local_file_path) not in dirs_to_skip
    ]
    print(f"Skipping {len(dirs_to_skip)} directories")
    # Sorting to track progress more easily
    still_to_push = sorted(still_to_push, key=lambda x: str(x.local_file_path))

    if num_to_process is not None:
        still_to_push = still_to_push[:num_to_process]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {
            executor.submit(update_model_to_hub, details): details.local_file_path
            for details in still_to_push
        }

        try:
            with tqdm(
                total=len(still_to_push), desc="Uploading models", unit="model"
            ) as pbar:
                for future in as_completed(future_to_dir):
                    try:
                        local_file_path, repo_id, revision, status, msg = (
                            future.result()
                        )
                        if status == "success":
                            successes.append((local_file_path, repo_id, revision, msg))
                        else:
                            failures.append((local_file_path, msg))

                    except Exception as e:
                        print(f"An error occurred: {e}")
                        failures.append((str(future_to_dir[future]), str(e)))
                        # Update the progress bar

                    pbar.update(1)
                    pbar.set_postfix(successful=len(successes), failed=len(failures))

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping uploads...")

        finally:
            # Cancel any remaining futures
            for future in future_to_dir:
                future.cancel()

    print(f"Successfully uploaded {len(successes)} models")
    print(f"Failed to upload {len(failures)} models")
    print("====")


if __name__ == "__main__":
    # Add basic arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--num-to-process", type=int, default=None)
    parser.add_argument(
        "--dirs-file",
        type=str,
        default=None,
        help="File to get directories from, rather than searching for them.",
    )
    args = parser.parse_args()

    main(
        max_workers=args.max_workers,
        num_to_process=args.num_to_process,
        dirs_file=args.dirs_file,
    )
