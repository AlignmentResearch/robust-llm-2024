"""Get attack scaling data from Tom's transfer evals."""

from robust_llm.plotting_utils.constants import get_run_names
from robust_llm.plotting_utils.tools import save_asr_data

# Include the name for easier debugging
summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.dataset.n_val",
    "model_size",
    "experiment_yaml.model.revision",
]
METRICS = [
    "adversarial_eval/attack_success_rate",
    "flops_per_iteration",
    "attack_flops",
] + [f"metrics/asr@{i}" for i in list(range(0, 128, 12)) + [128]]
ROUNDS = [0, 1e-4, 1e-3, 5e-3, -1]


def main():
    for attack, dataset in [
        ("rt_gcg", "imdb"),
        ("rt_gcg", "spam"),
        ("gcg_gcg", "imdb"),
        ("gcg_gcg", "spam"),
        ("gcg_gcg", "wl"),
        ("gcg_gcg", "pm"),
        ("gcg_gcg_infix90", "imdb"),
        ("gcg_gcg_infix90", "spam"),
        ("gcg_gcg_prefix", "imdb"),
        ("gcg_gcg_prefix", "spam"),
    ]:
        save_asr_data(
            summary_keys=summary_keys,
            metrics=METRICS,
            attack=attack,
            dataset=dataset,
            n_models=10,
            n_seeds=5,
            check_seeds=False,
            n_iterations=128,
            rounds=ROUNDS,
            **get_run_names(attack, dataset),
        )


if __name__ == "__main__":
    main()
