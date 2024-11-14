from pathlib import Path

import git.repo

ATTACK_DATA_NAME = "attack_data.csv"
SHARED_DATA_DIR = "/robust_llm_data"


def get_git_repo() -> git.repo.base.Repo:
    return git.repo.Repo(".", search_parent_directories=True)


def compute_repo_path() -> str:
    return str(get_git_repo().working_dir)


def get_current_git_commit_hash() -> str:
    return get_git_repo().head.commit.hexsha


def compute_dataset_path() -> str:
    path_to_repo = compute_repo_path()
    return f"{path_to_repo}/robust_llm/local_datasets"


def compute_dataset_management_path() -> str:
    path_to_repo = compute_repo_path()
    return f"{path_to_repo}/robust_llm/dataset_management"


def check_mount_health(path: Path) -> bool:
    """Check if a path exists and appears mounted correctly.

    We had cases where the shared data directory was not mounted correctly, such
    that the path existed but all of the sub-directories were missing and you could
    not write to it. To avoid the complexity of writing and deleting a directory
    to be absolutely sure that the path is mounted, we check for the existence of
    sub-directories.
    """
    return (
        path.exists()
        and path.is_dir()
        and path.is_mount()
        and any([p.is_dir() for p in path.iterdir()])
    )


def get_save_root() -> str:
    try:
        if check_mount_health(Path(SHARED_DATA_DIR)):
            return SHARED_DATA_DIR
    except OSError as e:
        raise OSError("Error checking for shared dir, maybe storage is down?") from e
    path = Path(compute_repo_path()) / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
