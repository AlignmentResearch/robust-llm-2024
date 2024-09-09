from pathlib import Path

from robust_llm.file_utils import compute_repo_path

model_sizes = ["14m", "31m", "70m", "160m", "410m"]
finetune_seeds = [0]
pretrain_seeds = range(1, 10)
dataset_version_combos = [
    ("imdb", "114"),
    ("pm", "117"),
    ("wl", "120"),
    ("spam", "123"),
    # ("harmless", "100"),
    # ("helpful", "101"),
]

template = """
defaults:
- /model/EleutherAI/pythia-{size}@_here_
- /model/Default/clf/base@_here_
- _self_

name_or_path: "AlignmentResearch/robust_llm_pythia-{size}-seed{pretrain_seed}_clf_{dataset}_v-ian-{version}_s-{finetune_seed}"
"""  # noqa: E501


def create_yaml_file(size, dataset, version, pretrain_seed, finetune_seed):
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/Default/clf/{dataset}/pythia-{size}-seed{pretrain_seed}-s{finetune_seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(
        size=size,
        dataset=dataset,
        version=version,
        pretrain_seed=pretrain_seed,
        finetune_seed=finetune_seed,
    )

    with open(filename, "w") as file:
        file.write(content)
    print(f"Created file: {filename}")


def main():
    for dataset, version in dataset_version_combos:
        for size in model_sizes:
            for finetune_seed in finetune_seeds:
                for pretrain_seed in pretrain_seeds:
                    create_yaml_file(
                        size=size,
                        dataset=dataset,
                        version=version,
                        pretrain_seed=pretrain_seed,
                        finetune_seed=finetune_seed,
                    )


if __name__ == "__main__":
    main()
