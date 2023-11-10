import git.repo


def compute_dataset_path() -> str:
    repo = git.repo.Repo(".", search_parent_directories=True)
    path_to_repo = repo.working_dir
    return f"{path_to_repo}/robust_llm/datasets"
