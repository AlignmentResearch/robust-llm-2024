"""Generate data for finetuned-style plots on adversarially trained models"""

from robust_llm.plotting_utils.constants import get_run_names
from robust_llm.plotting_utils.tools import prepare_adv_training_data

summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
    "experiment_yaml.dataset.n_val",
    "model_size",
]
metrics = [
    "adversarial_eval/attack_success_rate",
    "metrics/asr@12",
    "metrics/asr@60",
    "metrics/asr@72",
    "metrics/asr@128",
]


def main():
    attack = "gcg_gcg"
    for dataset in ("imdb", "spam", "wl", "pm"):
        prepare_adv_training_data(
            summary_keys=summary_keys,
            metrics=metrics,
            save_as=("post_adv_training", attack, dataset),
            **get_run_names(attack, dataset),
        )


if __name__ == "__main__":
    main()
