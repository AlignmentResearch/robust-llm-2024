"""Generate offense-defense plots for the paper."""

import argparse

import pandas as pd

from robust_llm.plotting_utils.style import set_style
from robust_llm.plotting_utils.tools import load_and_plot_offense_defense_plots

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 255)

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


def main(style):
    # Set the plot style
    set_style(style)

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
                for family, attack, dataset in [
                    ("pythia", "rt_gcg", "imdb"),
                    ("pythia", "rt_gcg", "spam"),
                    ("pythia", "gcg_gcg", "imdb"),
                    ("pythia", "gcg_gcg_infix90", "imdb"),
                    ("pythia", "gcg_gcg", "spam"),
                    ("pythia", "gcg_gcg_infix90", "spam"),
                ]:
                    load_and_plot_offense_defense_plots(
                        family=family,
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
                        style=style,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate offense-defense plots for the paper."
    )
    parser.add_argument("--style", type=str, default="paper", help="Plot style to use")
    args = parser.parse_args()
    main(args.style)
