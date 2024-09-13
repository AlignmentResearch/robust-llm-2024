from pathlib import Path

from robust_llm.file_utils import compute_repo_path

model_sizes = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
datasets = ["imdb", "pm", "wl", "spam", "harmless", "helpful"]
seeds = range(5)  # 0 to 4
attacks = ["gcg", "rt"]
template = """
defaults:
- /model/EleutherAI/pythia-{size}@_here_
- /model/Default/clf/base@_here_
- _self_

name_or_path: "AlignmentResearch/robust_llm__clf_{dataset}_pythia-{size}_s-{seed}_adv_tr_{attack}_t-{seed}"
"""  # noqa: E501


def create_yaml_file(size, attack, dataset, seed):
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/AdvTrained/clf/{attack}/{dataset}/pythia-{size}-s{seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(size=size, dataset=dataset, seed=seed, attack=attack)

    with open(filename, "w") as file:
        file.write(content)

    print(f"Created file: {filename}")


def main():
    for attack in attacks:
        for dataset in datasets:
            for size in model_sizes:
                for seed in seeds:
                    create_yaml_file(
                        size=size, dataset=dataset, seed=seed, attack=attack
                    )


if __name__ == "__main__":
    main()
