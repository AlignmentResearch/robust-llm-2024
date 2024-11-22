from pathlib import Path

from robust_llm.file_utils import compute_repo_path

seeds = range(3)
small_sizes = [
    "0.5B",
    "1.5B",
    "3B",
]
large_sizes = ["7B", "14B"]

dataset_version_size_combos = [
    ("harmless", "022a", small_sizes),
    ("helpful", "022b", small_sizes),
    ("spam", "022c", small_sizes),
    ("imdb", "022d", small_sizes),
    ("pm", "022e", small_sizes),
    ("wl", "022f", small_sizes),
    ("harmless", "023a", large_sizes),
    ("spam", "023b", large_sizes),
]

template = """
defaults:
- /model/Qwen/Qwen2.5-{size}@_here_
- /model/Default/clf/base@_here_
- _self_

name_or_path: "AlignmentResearch/robust_llm_Qwen2.5-{size}_clf_{dataset}_v-oskar-{version}_s-{seed}"
"""  # noqa: E501


def create_yaml_file(size, dataset, version, seed):
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/Default/clf/{dataset}/Qwen2.5-{size}-s{seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(size=size, dataset=dataset, version=version, seed=seed)

    with open(filename, "w") as file:
        file.write(content)

    print(f"Created file: {filename}")


def main():
    for dataset, version, sizes in dataset_version_size_combos:
        for size in sizes:
            for seed in seeds:
                create_yaml_file(size, dataset, version, seed)


if __name__ == "__main__":
    main()
