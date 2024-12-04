from pathlib import Path

import pandas as pd

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.tools import prepare_adv_training_data
from robust_llm.plotting_utils.utils import drop_duplicates

SUMMARY_KEYS = [
    "experiment_yaml.dataset.n_val",
    "model_family",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
    "experiment_yaml.evaluation.evaluation_attack.seed",
]

METRICS = [
    "flops_per_iteration",
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/n_correct_pre_attack",
    "adversarial_eval/n_incorrect_pre_attack",
    "adversarial_eval/n_examples",
    "adversarial_eval/n_correct_post_attack",
    "adversarial_eval/n_incorrect_post_attack",
    "adversarial_eval/post_attack_accuracy",
]

root = Path(compute_repo_path())


def remove_unnamed_cols(df: pd.DataFrame) -> None:
    unnamed_cols = df.columns[df.columns.str.contains("^Unnamed")].tolist()
    if len(unnamed_cols) > 0:
        print(f"Removing columns {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)


if __name__ == "__main__":
    for path in (root / "old_cache_csvs/evaluation").iterdir():
        if not path.stem.startswith("asr_") or path.stem.endswith("-test"):
            continue
        new_path = str(path).replace("asr_", "").replace("old_", "")
        if Path(new_path).exists():
            print(f"Skipping {path}")
            continue
        asr_data = pd.read_csv(path)
        remove_unnamed_cols(asr_data)
        if "flops_per_iteration" in asr_data:
            assert asr_data.flops_per_iteration.notnull().all()
            print(f"Skipping {path}")
            continue
        group = path.stem.replace("asr_", "").replace(".csv", "")
        logprob_path = str(path).replace("asr_", "logprob_")
        adv_data = prepare_adv_training_data(
            group, metrics=METRICS, summary_keys=SUMMARY_KEYS
        )
        remove_unnamed_cols(adv_data)
        adv_data = drop_duplicates(
            adv_data,
            [
                "model_name_or_path",
                "model_revision",
                "adv_training_round",
                "evaluation_evaluation_attack_seed",
            ],
            name="adv_data",
        )
        if Path(logprob_path).exists():
            logprob_data = pd.read_csv(logprob_path)
            remove_unnamed_cols(logprob_data)
            # Avoid off-by-one error between ASR and logprob data
            logprob_data.iteration += 1
            asr_data = asr_data.merge(
                logprob_data,
                on=[
                    "model_idx",
                    "seed_idx",
                    "iteration",
                    "adv_training_round",
                    "model_size",
                ],
                how="left",
                validate="1:1",
                suffixes=("", "_logprob"),
            )
        data = asr_data.merge(
            adv_data,
            on=["model_idx", "seed_idx", "adv_training_round"],
            how="left",
            validate="m:1",
            suffixes=("", "_adv"),
        )
        assert data.flops_per_iteration.notnull().all()
        data.to_csv(new_path, index=False)
