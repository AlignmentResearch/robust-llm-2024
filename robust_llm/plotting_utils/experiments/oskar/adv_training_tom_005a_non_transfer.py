"""Figure 3 plots (adversarial training) on new data."""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Set the plot style
set_plot_style("paper")

# Include the name for easier debugging
summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.dataset.n_val",
    "model_size",
    "experiment_yaml.model.revision",
]
METRICS = [
    "metrics/asr@12",
    "metrics/asr@60",
    "metrics/asr@120",
    "metrics/asr@128",
]
for x_data_name in ("n_parameter_updates", "adv_training_round"):
    for legend in (True, False):
        load_and_plot_adv_training_plots(
            run_names=("tom_007_eval_niki_152_gcg ", "tom_007_eval_niki_152a_gcg "),
            title="IMDB, GCG",
            save_as=("imdb", "gcg_vs_gcg12", x_data_name),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            y_data_name="metrics/asr@12",
            metrics=METRICS,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_009_eval_niki_170_gcg ",),
            title="Spam, GCG",
            save_as=("spam", "gcg_vs_gcg12", x_data_name),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            y_data_name="metrics/asr@12",
            metrics=METRICS,
        )

        load_and_plot_adv_training_plots(
            run_names=("tom_007_eval_niki_152_gcg ", "tom_007_eval_niki_152a_gcg "),
            title="IMDB, GCG",
            save_as=("imdb", "gcg_vs_gcg60", x_data_name),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            y_data_name="metrics/asr@60",
            metrics=METRICS,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_009_eval_niki_170_gcg ",),
            title="Spam, GCG",
            save_as=("spam", "gcg_vs_gcg60", x_data_name),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            y_data_name="metrics/asr@60",
            metrics=METRICS,
        )
