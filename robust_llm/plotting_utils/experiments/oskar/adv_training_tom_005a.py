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
metrics = [
    "adversarial_eval/attack_success_rate",
    "train/total_flops",
]
for x_data_name in (
    "n_parameter_updates",
    "adv_training_round",
    "train_total_flops",
    "flops_fraction_pretrain",
):
    for legend in (True, False):
        load_and_plot_adv_training_plots(
            run_names=("tom_005a_eval_niki_149_gcg ",),
            merge_runs=("niki_149_adv_tr_rt_imdb_small",),
            title=r"IMDB, RT $\to$ GCG",
            save_as=("transfer", "imdb", "rt_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            metrics=metrics,
            legend=legend,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_006_eval_niki_150_gcg ",),
            merge_runs=("niki_150_adv_tr_rt_spam_small",),
            title=r"Spam, RT $\to$ GCG",
            save_as=("transfer", "spam", "rt_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            metrics=metrics,
            legend=legend,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_007_eval_niki_152_gcg ", "tom_007_eval_niki_152a_gcg "),
            merge_runs=(
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
            ),
            title=r"IMDB, GCG $\to$ GCG",
            save_as=("transfer", "imdb", "gcg_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=(
                "tom_008_eval_niki_152_gcg_infix90 ",
                "tom_008_eval_niki_152a_gcg_infix90 ",
            ),
            merge_runs=(
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
            ),
            title=r"IMDB, GCG $\to$ 90%-infix GCG",
            save_as=("transfer", "imdb", "gcg_to_gcg_infix"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_009_eval_niki_170_gcg ",),
            merge_runs="niki_170_adv_tr_gcg_spam_small",
            title=r"Spam, GCG $\to$ GCG",
            save_as=("transfer", "spam", "gcg_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_010_eval_niki_170_gcg_infix90 ",),
            merge_runs="niki_170_adv_tr_gcg_spam_small",
            title=r"Spam, GCG $\to$ 90%-infix GCG",
            save_as=("transfer", "spam", "gcg_to_gcg_infix"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_012_eval_niki_153_gcg",),
            merge_runs=("niki_153_adv_tr_rt_wl_small", "niki_159_adv_tr_rt_wl_large"),
            title=r"WL, RT $\to$ GCG",
            save_as=("transfer", "wl", "gcg_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_014_eval_niki_172_gcg ",),
            merge_runs=("niki_172_adv_tr_gcg_wl_small",),
            title=r"WL, GCG $\to$ GCG",
            save_as=("transfer", "wl", "gcg_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_013_eval_niki_171_gcg",),
            merge_runs=("niki_171_adv_tr_gcg_pm_small",),
            title=r"PM, GCG $\to$ GCG",
            save_as=("transfer", "pm", "gcg_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
        load_and_plot_adv_training_plots(
            run_names=("tom_011_eval_niki_151_gcg",),
            merge_runs=("niki_151_adv_tr_rt_pm_small", "niki_158_adv_tr_rt_pm_large"),
            title=r"PM, RT $\to$ GCG",
            save_as=("transfer", "pm", "rt_to_gcg"),
            summary_keys=summary_keys,
            x_data_name=x_data_name,
            color_data_name="num_params",
            legend=legend,
            metrics=metrics,
        )
