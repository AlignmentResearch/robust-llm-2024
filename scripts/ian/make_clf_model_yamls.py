from pathlib import Path

from robust_llm.file_utils import compute_repo_path

seeds = range(5)  # 0 to 4
hh_sizes_small = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b"]
hh_sizes_large = ["6.9b", "12b"]
other_sizes_small = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b"]
other_sizes_large = ["2.8b", "6.9b", "12b"]

dataset_version_size_combos = [
    ("harmless", "100", hh_sizes_small),
    ("harmless", "135c", hh_sizes_large),
    ("helpful", "101", hh_sizes_small),
    ("helpful", "136", hh_sizes_large),
    ("imdb", "079", other_sizes_small),
    ("imdb", "137", other_sizes_large),
    ("pm", "080", other_sizes_small),
    ("pm", "138", other_sizes_large),
    ("spam", "082", other_sizes_small),
    ("spam", "139", other_sizes_large),
    ("wl", "081", other_sizes_small),
    ("wl", "140", other_sizes_large),
]

template = """
defaults:
- /model/EleutherAI/pythia-{size}@_here_
- /model/Default/clf/base@_here_
- _self_

name_or_path: "AlignmentResearch/robust_llm_pythia-{size}_clf_{dataset}_v-ian-{version}_s-{seed}"
"""  # noqa: E501


def create_yaml_file(size, dataset, version, seed):
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/Default/clf/{dataset}/pythia-{size}-s{seed}.yaml"  # noqa: E501
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
