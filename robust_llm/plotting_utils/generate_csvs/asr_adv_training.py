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
    for family, attack, dataset in [
        ("pythia", "rt_gcg", "imdb"),
        ("pythia", "rt_gcg", "spam"),
        ("pythia", "gcg_gcg", "imdb"),
        ("pythia", "gcg_gcg", "spam"),
        ("pythia", "gcg_gcg", "wl"),
        ("pythia", "gcg_gcg", "pm"),
        ("pythia", "gcg_gcg_infix90", "imdb"),
        ("pythia", "gcg_gcg_infix90", "spam"),
        ("pythia", "gcg_gcg_prefix", "imdb"),
        ("pythia", "gcg_gcg_prefix", "spam"),
        ("pythia", "gcg_no_ramp_gcg", "imdb"),
    ]:
        save_asr_data(
            summary_keys=summary_keys,
            metrics=METRICS,
            family=family,
            attack=attack,
            dataset=dataset,
            n_models=10,
            n_seeds=5,
            check_seeds=False,
            n_iterations=64 if "infix" in attack else 128,
            rounds=ROUNDS,
            **get_run_names(family, attack, dataset),
        )


if __name__ == "__main__":
    main()
