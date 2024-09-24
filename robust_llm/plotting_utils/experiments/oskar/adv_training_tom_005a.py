"""Figure 4 plots (transfer) on new data."""

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
for x_data_name in ("n_parameter_updates", "adv_training_round"):
    for legend in (True, False):
        load_and_plot_adv_training_plots(
            run_names=("tom_005a_eval_niki_149_gcg ",),
            title="IMDB, RT -> GCG",
            save_as=("24-rounds", "imdb", "gcg_vs_rt", x_data_name),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            # xlim=xlim,
            legend=legend,
            # check_seeds=3,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_006_eval_niki_150_gcg ",),
            title="Spam, RT -> GCG",
            save_as=("24-rounds", "spam", "gcg_vs_rt", x_data_name),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            # xlim=xlim,
            legend=legend,
            # check_seeds=3,
        )
