from pathlib import Path

from robust_llm.file_utils import compute_repo_path

model_sizes = ["14m", "31m", "70m", "160m", "410m"]
pretrain_seeds = range(1, 10)  # 1 to 9

template = """
defaults:
- /model/EleutherAI/pythia-{size}@_here_
- _self_

name_or_path: "EleutherAI/pythia-{size}-seed{pretrain_seed}"
# 'main' is empty for these models, 143k steps is the last checkpoint.
revision: "step143000"
"""  # noqa: E501


def create_yaml_file(size, pretrain_seed):
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/EleutherAI/pythia-{size}-seed{pretrain_seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(size=size, pretrain_seed=pretrain_seed)

    with open(filename, "w") as file:
        file.write(content)

    print(f"Created file: {filename}")


def main():
    for size in model_sizes:
        for pretrain_seed in pretrain_seeds:
            create_yaml_file(size, pretrain_seed)


if __name__ == "__main__":
    main()
