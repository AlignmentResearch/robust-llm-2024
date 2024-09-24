"""Figure 3 plots (adversarial training).

Based on robust_llm/plotting_utils/experiments/niki/adv_training/2024-05-20-adv-round.py
"""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots
from robust_llm.wandb_utils.constants import SUMMARY_KEYS

# Set the plot style
set_plot_style("paper")

# Include the name for easier debugging
summary_keys = SUMMARY_KEYS + [
    "experiment_yaml.training.force_name_to_save",
    "experiment_yaml.training.seed",
]
for x_data_name in ("n_parameter_updates", "adv_training_round"):
    for num_rounds in (10, 30):
        xlim = (1, num_rounds)
        for legend in (True, False):
            load_and_plot_adv_training_plots(
                run_names=(
                    "niki_052_long_adversarial_training_gcg_imdb",
                    "niki_052a_long_adversarial_training_gcg_imdb",
                ),
                title="IMDB, GCG attack",
                save_as=(f"{num_rounds}-rounds", "imdb", "gcg", x_data_name),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                color_data_name="num_params",
                xlim=xlim,
                legend=legend,
                check_seeds=3,
            )
            load_and_plot_adv_training_plots(
                run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
                title="Spam, GCG attack",
                save_as=(f"{num_rounds}-rounds", "spam", "gcg", x_data_name),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                color_data_name="num_params",
                xlim=xlim,
                legend=legend,
                check_seeds=3,
            )
