"""Generate data for adversarial training transfer plots."""

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
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/n_correct_post_attack",
    "adversarial_eval/n_examples",
]


def main():
    for attack, dataset in (
        ("rt_gcg", "imdb"),
        ("rt_gcg", "spam"),
        ("rt_gcg", "wl"),
        ("rt_gcg", "pm"),
        ("gcg_gcg_infix90_match_seed", "imdb"),
        ("gcg_gcg_infix90", "imdb"),
        ("gcg_gcg_infix90", "spam"),
        ("gcg_gcg_prefix", "imdb"),
        ("gcg_gcg_prefix", "spam"),
        ("gcg_no_ramp_gcg", "imdb"),
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
