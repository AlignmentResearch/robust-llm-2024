from pathlib import Path

from tqdm import tqdm

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.constants import RUN_NAMES
from robust_llm.plotting_utils.tools import load_flops_data


def get_adv_training_groups():
    groups = []
    for family, family_dict in RUN_NAMES.items():
        assert isinstance(family_dict, dict)
        for attack, attack_dict in family_dict.items():
            assert isinstance(attack_dict, dict)
            for dataset, dataset_dict in attack_dict.items():
                assert isinstance(dataset_dict, dict)
                runs = dataset_dict["merge_runs"]
                if isinstance(runs, str):
                    runs = [runs]
                groups += runs
    return groups


if __name__ == "__main__":
    root = Path(compute_repo_path())
    groups = get_adv_training_groups()
    for group in tqdm(groups):
        flops = load_flops_data(group)
        path = root / "cache_csvs" / "training" / f"{group}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        flops.to_csv(path, index=False)
