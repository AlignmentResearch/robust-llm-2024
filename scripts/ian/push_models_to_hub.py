import argparse
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

RECORD_DIR = Path("/robust_llm_data/upload-info")

# Avoid overwriting data from other threads
file_locks = {
    "success_new": threading.Lock(),
    "success_exists": threading.Lock(),
    "failure_load": threading.Lock(),
    "failure_upload": threading.Lock(),
    "failure_match": threading.Lock(),
    "failure_untracked": threading.Lock(),
    "processed": threading.Lock(),
    "interrupted": threading.Lock(),
}


def write_to_file(file_name: str, content: str):
    with file_locks[file_name]:
        with (RECORD_DIR / f"{file_name}.csv").open("a") as f:
            f.write(content)


def push_model_to_hub(
    local_file_path: Path, repo_id: str, revision: str
) -> tuple[str, str, str, str, str]:
    """Push a model to the hub given its path and revision.

    Steps involved:
    1. Check whether the repo and revision already exist on HFHub.
        a. If they do, check that the huggingface version matches the local version?
    1. Verify that all necessary files are present in the local_file_path.
    2. Verify that the model can be loaded into memory.
    3. Upload the model.

    This function should also record the results to one of four files:
    1. success_new.txt - if the model was successfully uploaded.
    2. success_exists.txt - if the model already existed on the hub.
    3. failure_new.txt - if the model failed to load or upload.
    4. failure_exists.txt - if the model already existed on the hub but does not match.
    """
    api = HfApi()
    line_to_write = f"{str(local_file_path)},{repo_id},{revision}\n"
    status = "success"
    message = ""

    try:
        # Get some proof that the local checkpoint is not corrupted.
        input_ids = torch.tensor([[1000, 1001]])
        local_logits = verify_model_loading(local_file_path, input_ids)
        if local_logits is None:
            write_to_file("failure_load", line_to_write)
            raise ValueError(f"Failed to load model from {local_file_path}")

        # Check if the branch already exists first, before trying to upload again.
        already_on_hub = revision_exists(api, repo_id, revision)
        if not already_on_hub:
            upload_model(api, repo_id, revision, local_file_path)

        logits_match = check_model_upload(repo_id, revision, input_ids, local_logits)
        if not logits_match:
            if already_on_hub:
                write_to_file("failure_match", line_to_write)
            else:
                write_to_file("failure_upload", line_to_write)
            raise ValueError("Logits did not match")
        if already_on_hub:
            write_to_file("success_exists", line_to_write)
        else:
            write_to_file("success_new", line_to_write)

    except KeyboardInterrupt:
        status = "interrupted"
        message = "Upload interrupted by user"
        write_to_file("interrupted", line_to_write)
        raise

    except Exception as e:
        status = "failure"
        message = str(e)
        write_to_file("failure_untracked", line_to_write)

    # Record that we've processed this dir, one way or another
    write_to_file("processed", line_to_write)
    return str(local_file_path), repo_id, revision, status, message


def revision_exists(api, repo_id: str, revision: str) -> bool:
    try:
        api.repo_info(repo_id, revision=revision)
        return True
    except Exception:
        return False


def check_model_upload(
    repo_id: str, revision: str, input_ids, local_logits: torch.Tensor
) -> bool:
    """Download an uploaded model and check that its logits match the local model."""
    # First download the repo to a temp dir and *then* load it, to take
    # advantage of faster download.
    temp_dir = f"/tmp/{repo_id}/{revision}"
    snapshot_download(repo_id, revision=revision, local_dir=temp_dir)
    model = load_model(temp_dir)
    shutil.rmtree(temp_dir)
    remote_logits = model(input_ids=input_ids).logits
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


def construct_file_details(model_dir: Path, match: re.Match, base_dir: Path):
    dataset = match.group("dataset")
    base_model_name = match.group("base_model_name")
    attack = match.group("attack")
    ft_seed = match.group("ft_seed")
    adv_seed = match.group("adv_seed")
    model_name = (
        f"clf_{dataset}_{base_model_name}_s-{ft_seed}_adv_tr_{attack}_t-{adv_seed}"
    )
    revision = f"adv-training-round-{match.group('round')}"
    repo_id = f"AlignmentResearch/{model_name}"
    local_file_path = base_dir / model_dir
    return local_file_path, repo_id, revision


def get_already_processed_dirs() -> set[str]:
    processed_dirs = set()
    processed_path = RECORD_DIR / "processed.csv"
    if not processed_path.exists():
        return set()

    with processed_path.open("r") as f:
        for line in f:
            local_file_path, repo_id, revision = line.strip().split(",")
            processed_dirs.add(local_file_path)
    return processed_dirs


def get_dirs_to_push() -> list[tuple[Path, str, str]]:
    base_dir = Path("/robust_llm_data/models")
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
    for model_dir in base_dir.iterdir():
        match = dir_regex.match(model_dir.name)
        if match is None:
            no_push.append(model_dir)
            continue
        local_file_path, repo_id, revision = construct_file_details(
            model_dir, match, base_dir
        )
        push.append((local_file_path, repo_id, revision))
    print(f"Will push {len(push)} models and not push {len(no_push)} models")
    return push


def main(max_workers: int, num_to_process: int | None):
    successes = []
    failures = []
    dirs_to_push = get_dirs_to_push()
    already_processed_dirs = get_already_processed_dirs()
    still_to_push = [
        args for args in dirs_to_push if str(args[0]) not in already_processed_dirs
    ]
    print(f"Skipping {len(already_processed_dirs)} already processed directories")
    # Sorting to track progress more easily
    still_to_push = sorted(still_to_push, key=lambda x: str(x[0]))

    if num_to_process is not None:
        still_to_push = still_to_push[:num_to_process]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {
            executor.submit(push_model_to_hub, *args): args[0] for args in still_to_push
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
    args = parser.parse_args()

    main(max_workers=args.max_workers, num_to_process=args.num_to_process)
