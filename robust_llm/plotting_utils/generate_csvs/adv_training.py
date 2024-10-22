"""Generate data for adversarial training plots (robustness over time)"""

from robust_llm.plotting_utils.constants import get_run_names
from robust_llm.plotting_utils.tools import save_adv_training_data

summary_keys = [
    "experiment_yaml.dataset.n_val",
    "model_size",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
]
METRICS = [
    "metrics/asr@12",
    "metrics/asr@60",
    "metrics/asr@72",
    "metrics/asr@120",
    "metrics/asr@128",
]


def main():
    attack = "gcg_gcg"
    for dataset in (
        "imdb",
        "spam",
        "wl",
        "pm",
    ):
        save_adv_training_data(
            attack=attack,
            dataset=dataset,
            summary_keys=summary_keys,
            metrics=METRICS,
            use_group_cache=True,
            **get_run_names(attack, dataset),
        )


if __name__ == "__main__":
    main()
