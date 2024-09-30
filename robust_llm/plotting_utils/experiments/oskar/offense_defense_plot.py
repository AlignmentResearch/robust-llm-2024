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
METRICS = [
    "metrics/asr@12",
    "metrics/asr@60",
    "metrics/asr@120",
    "metrics/asr@128",
]
for x_data_name in (
    "train_total_flops",
    # "flops_percent_pretrain",
    # "adv_training_round",
    # "n_parameter_updates",
):
    for y_data_name in (
        "interpolated_iteration_for_0.1_percent_flops",
        "interpolated_iteration_for_1_percent",
        "interpolated_iteration_for_1_percent_flops",
        "interpolated_iteration_for_5_percent",
        "interpolated_iteration_for_5_percent_flops",
        "interpolated_iteration_for_10_percent",
        "interpolated_iteration_for_10_percent_flops",
        "interpolated_iteration_for_50_percent",
        "interpolated_iteration_for_50_percent_flops",
    ):
        for legend in (True, False):
            load_and_plot_offense_defense_plots(
                group_names="tom_005a_eval_niki_149_gcg",
                merge_runs="niki_149_adv_tr_rt_imdb_small",
                title="IMDB, RT -> GCG",
                save_as=("offense_defense", "imdb", "rt_to_gcg"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                color_data_name="num_params",
                legend=legend,
                metrics=METRICS,
            )
            load_and_plot_offense_defense_plots(
                group_names="tom_006_eval_niki_150_gcg",
                merge_runs="niki_150_adv_tr_rt_spam_small",
                title="Spam, RT -> GCG",
                save_as=("offense_defense", "spam", "rt_to_gcg"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                color_data_name="num_params",
                legend=legend,
                metrics=METRICS,
            )
            load_and_plot_offense_defense_plots(
                group_names=("tom_007_eval_niki_152_gcg", "tom_007_eval_niki_152a_gcg"),
                merge_runs=(
                    "niki_152a_adv_tr_gcg_imdb_small",
                    "niki_152_adv_tr_gcg_imdb_small",
                ),
                title="IMDB, GCG -> GCG",
                save_as=("offense_defense", "imdb", "gcg_to_gcg"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                color_data_name="num_params",
                legend=legend,
                metrics=METRICS,
            )
            load_and_plot_offense_defense_plots(
                group_names=(
                    "tom_008_eval_niki_152_gcg_infix90",
                    "tom_008_eval_niki_152a_gcg_infix90",
                ),
                merge_runs=(
                    "niki_152a_adv_tr_gcg_imdb_small",
                    "niki_152_adv_tr_gcg_imdb_small",
                ),
                title="IMDB, GCG -> 90%-infix GCG",
                save_as=("offense_defense", "imdb", "gcg_to_gcg_infix"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                color_data_name="num_params",
                legend=legend,
                metrics=METRICS,
            )
            load_and_plot_offense_defense_plots(
                group_names=("tom_009_eval_niki_170_gcg"),
                merge_runs=("niki_170_adv_tr_gcg_spam_small"),
                title="Spam, GCG -> GCG",
                save_as=("offense_defense", "spam", "gcg_to_gcg"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                color_data_name="num_params",
                legend=legend,
                metrics=METRICS,
            )
            load_and_plot_offense_defense_plots(
                group_names=("tom_010_eval_niki_170_gcg_infix90",),
                merge_runs=("niki_170_adv_tr_gcg_spam_small"),
                title="Spam, GCG -> 90%-infix GCG",
                save_as=("offense_defense", "spam", "gcg_to_gcg_infix"),
                summary_keys=summary_keys,
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                color_data_name="num_params",
                legend=legend,
                metrics=METRICS,
            )
