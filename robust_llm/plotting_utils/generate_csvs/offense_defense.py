"""Generate data for offense-defense plots"""

from robust_llm.plotting_utils.constants import get_run_names
from robust_llm.plotting_utils.tools import prepare_offense_defense_data

# Include the name for easier debugging
summary_keys = [
    "experiment_yaml.dataset.n_val",
    "model_size",
    "flops_per_iteration",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
]
# We have to get at least one metric because
# of the way the data is loaded.
METRICS = [
    "metrics/asr@12",
]


def main():
    for attack, dataset in [
        ("rt_gcg", "imdb"),
        ("rt_gcg", "spam"),
        ("gcg_gcg", "imdb"),
        ("gcg_gcg_infix90", "imdb"),
        ("gcg_gcg", "spam"),
        ("gcg_gcg_infix90", "spam"),
    ]:
        for target_asr in [5]:
            prepare_offense_defense_data(
                attack=attack,
                dataset=dataset,
                summary_keys=summary_keys,
                target_asr=target_asr,
                metrics=METRICS,
                **get_run_names(attack, dataset),
            )


if __name__ == "__main__":
    main()
