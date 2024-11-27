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
    for family, attack, dataset in (
        ("pythia", "rt_gcg", "imdb"),
        ("pythia", "rt_gcg", "spam"),
        ("pythia", "rt_gcg", "wl"),
        ("pythia", "rt_gcg", "pm"),
        ("pythia", "gcg_gcg_infix90_match_seed", "imdb"),
        ("pythia", "gcg_gcg_infix90", "imdb"),
        ("pythia", "gcg_gcg_infix90", "spam"),
        ("pythia", "gcg_gcg_prefix", "imdb"),
        ("pythia", "gcg_gcg_prefix", "spam"),
        ("pythia", "gcg_no_ramp_gcg", "imdb"),
    ):
        save_adv_training_data(
            family=family,
            attack=attack,
            dataset=dataset,
            summary_keys=summary_keys,
            metrics=METRICS,
            use_group_cache=True,
            **get_run_names(family, attack, dataset),
        )


if __name__ == "__main__":
    main()
