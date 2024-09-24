"""
Figure 4 plots (transfer).

Based on plotting_utils/experiments/niki/adv_training/2024-05-20-transfer-adv-round.py
"""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Set the plot style
set_plot_style("paper")

# Include the name for easier debugging
summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
    "model_size",
]
xlim = (1, 10)
for x_data_name in ("n_parameter_updates", "adv_training_round"):
    for legend in (True, False):
        # load_and_plot_adv_training_plots(
        #     run_names=("niki_051_eval_enronspam_rt_gcg_transfer",),
        #     title="Spam, RT → GCG",
        #     save_as=("transfer", "spam", "rt_gcg"),
        #     summary_keys=summary_keys,
        #     xlim=xlim,
        #     x_data_name="n_parameter_updates",
        #     legend=legend,
        # )

        # load_and_plot_adv_training_plots(
        #     run_names=(
        #         "niki_065_eval_enronspam_gcg_gcg-30_transfer",
        #         "niki_065a_eval_enronspam_gcg_gcg-30_transfer",
        #     ),
        #     title="Spam, GCG → GCG 30",
        #     save_as=("transfer", "spam", "gcg_10_gcg_30"),
        #     summary_keys=summary_keys,
        #     xlim=xlim,
        #     x_data_name="n_parameter_updates",
        #     legend=legend,
        # )
        load_and_plot_adv_training_plots(
            run_names=("niki_050_eval_imdb_rt_gcg_transfer",),
            title="IMDB, RT → GCG",
            save_as=("transfer", "imdb", "rt_gcg"),
            summary_keys=summary_keys,
            xlim=xlim,
            x_data_name="n_parameter_updates",
            legend=legend,
        )
