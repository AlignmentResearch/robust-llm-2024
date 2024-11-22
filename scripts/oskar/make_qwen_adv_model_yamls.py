from pathlib import Path

from robust_llm.file_utils import compute_repo_path
from robust_llm.wandb_utils.constants import QWEN_MODEL_NAMES

datasets = [
    "spam",
    "harmless",
]
seeds = range(3)
attacks = [
    "gcg",
]
template = """
defaults:
- /model/Qwen/Qwen2.5-{size}@_here_
- /model/Default/clf/base@_here_
- _self_

name_or_path: "AlignmentResearch/robust_llm_clf_{version}_{dataset}_Qwen2.5-{size}_s-{seed}_adv_tr_{attack}_t-{seed}"
"""  # noqa: E501

CANONICAL_VERSIONS = {
    "gcg": {
        "spam": dict.fromkeys(QWEN_MODEL_NAMES, "oskar-024b"),
        "harmless": dict.fromkeys(QWEN_MODEL_NAMES, "oskar-024a"),
    }
}


def create_yaml_file(size, attack, dataset, seed):
    version = CANONICAL_VERSIONS.get(attack, {}).get(dataset, {}).get(size)
    if version is None:
        return
    repo_path = compute_repo_path()
    filename = Path(
        f"{repo_path}/robust_llm/hydra_conf/model/AdvTrained/clf/{attack}/{dataset}/Qwen2.5-{size}-s{seed}.yaml"  # noqa: E501
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    content = template.format(
        version=version,
        dataset=dataset,
        size=size,
        seed=seed,
        attack=attack,
    )

    with open(filename, "w") as file:
        file.write(content)

    print(f"Created file: {filename}")


def main():
    for attack in attacks:
        for dataset in datasets:
            for size in QWEN_MODEL_NAMES:
                for seed in seeds:
                    create_yaml_file(
                        size=size, dataset=dataset, seed=seed, attack=attack
                    )


if __name__ == "__main__":
    main()
