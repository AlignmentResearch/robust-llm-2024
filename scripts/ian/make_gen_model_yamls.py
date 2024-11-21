from pathlib import Path

from robust_llm.file_utils import compute_repo_path

seeds = [0]  # Only one seed for these test gen models
sizes = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b"]

dataset_path_sizes_combos = [
    ("pm", "AlignmentResearch/robust_llm_pythia-{size}-pm_1.2.0-gen-ian-nd-v2", sizes),
    ("imdb", "AlignmentResearch/robust_llm_pythia-{size}-imdb-gen-ian-nd", sizes),
]

template = """
defaults:
- /model/EleutherAI/pythia-{size}@_here_
- /model/Default/gen/base@_here_
- _self_

name_or_path: {model_path}
"""  # noqa: E501


def create_yaml_file(size, dataset, model_path, seed):
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/Default/gen/{dataset}/pythia-{size}-s{seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(size=size, model_path=model_path)

    with open(filename, "w") as file:
        file.write(content)

    print(f"Created file: {filename}")


def main():
    for dataset, path_template, sizes in dataset_path_sizes_combos:
        for size in sizes:
            for seed in seeds:
                create_yaml_file(size, dataset, path_template.format(size=size), seed)


if __name__ == "__main__":
    main()
