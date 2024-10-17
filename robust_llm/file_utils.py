import git.repo


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
