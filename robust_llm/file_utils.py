import git.repo


def compute_repo_path() -> str:
    repo = git.repo.Repo(".", search_parent_directories=True)
    return str(repo.working_dir)


def compute_dataset_path() -> str:
    path_to_repo = compute_repo_path()
    return f"{path_to_repo}/robust_llm/local_datasets"


def compute_dataset_management_path() -> str:
    path_to_repo = compute_repo_path()
    return f"{path_to_repo}/robust_llm/dataset_management"
