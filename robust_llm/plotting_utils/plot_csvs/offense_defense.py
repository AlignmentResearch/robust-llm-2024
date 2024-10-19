"""Generate offense-defense plots for the paper."""

import pandas as pd

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_offense_defense_plots

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 255)
# Set the plot style
set_plot_style("paper")

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
    for x_data_name, xscale, y_data_name, yscale in (
        (
            "train_total_flops",
            "log",
            "interpolated_iteration_for_{target_asr}_percent_flops",
            "log",
        ),
        (
            "defense_flops_fraction_pretrain",
            "log",
            "interpolated_iteration_for_{target_asr}_percent_flops_fraction_pretrain",
            "log",
        ),
    ):
        for target_asr in [5]:
            x_data_name = x_data_name.format(target_asr=target_asr)
            y_data_name = y_data_name.format(target_asr=target_asr)
            add_parity_line = False
            if x_data_name.endswith("pretrain") and y_data_name.endswith("pretrain"):
                diagonal_gridlines = True
            elif x_data_name.endswith("flops") and y_data_name.endswith("flops"):
                diagonal_gridlines = True
            else:
                diagonal_gridlines = False

            for legend in (True, False):
                for attack, dataset in [
                    ("rt_gcg", "imdb"),
                    ("rt_gcg", "spam"),
                    ("gcg_gcg", "imdb"),
                    ("gcg_gcg_infix90", "imdb"),
                    ("gcg_gcg", "spam"),
                    ("gcg_gcg_infix90", "spam"),
                ]:
                    load_and_plot_offense_defense_plots(
                        attack=attack,
                        dataset=dataset,
                        x_data_name=x_data_name,
                        y_data_name=y_data_name,
                        color_data_name="num_params",
                        legend=legend,
                        xscale=xscale,
                        yscale=yscale,
                        add_parity_line=add_parity_line,
                        diagonal_gridlines=diagonal_gridlines,
                    )


if __name__ == "__main__":
    main()
