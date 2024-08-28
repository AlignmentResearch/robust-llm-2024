from pathlib import Path

model_sizes = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
seeds = range(5)  # 0 to 4
dataset_version_combos = [
    ("imdb", "079"),
    ("pm", "080"),
    ("wl", "081"),
    ("spam", "082"),
    ("harmless", "100"),
    ("helpful", "101"),
]

template = """
defaults:
- /model/EleutherAI/pythia-{size}@_here_
- /model/Default/clf/base@_here_
- _self_

name_or_path: "AlignmentResearch/robust_llm_pythia-{size}_clf_{dataset}_v-ian-{version}_s-{seed}"
"""  # noqa: E501


def create_yaml_file(size, dataset, version, seed):
    filename = Path(
        f"/Users/ian/code/farai/robust-llm/robust_llm/hydra_conf/model/Default/clf/{dataset}/pythia-{size}-s{seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(size=size, dataset=dataset, version=version, seed=seed)

    with open(filename, "w") as file:
        file.write(content)

    print(f"Created file: {filename}")


def main():
    for dataset, version in dataset_version_combos:
        for size in model_sizes:
            for seed in seeds:
                create_yaml_file(size, dataset, version, seed)


if __name__ == "__main__":
    main()
