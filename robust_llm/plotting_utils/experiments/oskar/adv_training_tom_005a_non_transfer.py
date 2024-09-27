"""Figure 3 plots (adversarial training) on new data."""

import pandas as pd

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 255)
# Set the plot style
set_plot_style("paper")

# Include the name for easier debugging
summary_keys = [
    "experiment_yaml.dataset.n_val",
    "model_size",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
]
METRICS = [
    "metrics/asr@12",
    "metrics/asr@60",
    "metrics/asr@120",
    "metrics/asr@128",
]
for x_data_name in (
    "train_total_flops",
    "flops_percent_pretrain",
    "adv_training_round",
    "n_parameter_updates",
):
    for legend in (True, False):
        for iteration in (12, 60):
            load_and_plot_adv_training_plots(
                run_names=("tom_007_eval_niki_152_gcg", "tom_007_eval_niki_152a_gcg"),
                merge_runs=(
                    "niki_152a_adv_tr_gcg_imdb_small",
                    "niki_152_adv_tr_gcg_imdb_small",
                ),
                title="IMDB, GCG",
                save_as=("imdb", f"gcg_vs_gcg{iteration}"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                color_data_name="num_params",
                legend=legend,
                y_data_name=f"metrics_asr_at_{iteration}",
                metrics=METRICS,
            )
            load_and_plot_adv_training_plots(
                run_names=("tom_009_eval_niki_170_gcg",),
                merge_runs="niki_170_adv_tr_gcg_spam_small",
                title="Spam, GCG",
                save_as=("spam", f"gcg_vs_gcg{iteration}"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                color_data_name="num_params",
                legend=legend,
                y_data_name=f"metrics_asr_at_{iteration}",
                metrics=METRICS,
            )
