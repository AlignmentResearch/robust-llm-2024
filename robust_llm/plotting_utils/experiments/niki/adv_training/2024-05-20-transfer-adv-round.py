from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

transfer_summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
    "model_size",
]

save_dir = "transfer"
xlim = (1, 10)

legend = False

# Set the plot style
set_plot_style("paper")

# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_048_eval_pm_rt_gcg_transfer",
#         "niki_048a_eval_pm_rt_gcg_transfer",
#     ),
#     title="PM, RT train, GCG eval",
#     save_as="pm_rt_gcg_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_049_eval_wl_rt_gcg_transfer",),
#     title="WL, RT train, GCG eval",
#     save_as="wl_rt_gcg_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_050_eval_imdb_rt_gcg_transfer",),
#     title="IMDB, RT train, GCG eval",
#     save_as="imdb_rt_gcg_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_051_eval_enronspam_rt_gcg_transfer",),
#     title="Spam, RT train, GCG eval",
#     save_as="spam_rt_gcg_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_064_eval_imdb_gcg_gcg-30_transfer",),
#     title="IMDB, GCG train, GCG 30 its eval",
#     save_as="imdb_gcg_gcg-30_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_065_eval_enronspam_gcg_gcg-30_transfer",
#         "niki_065a_eval_enronspam_gcg_gcg-30_transfer",
#     ),
#     title="Spam, GCG train, GCG 30 its eval",
#     save_as="spam_gcg_gcg-30_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_066_eval_PasswordMatch_gcg_gcg-30_transfer",
#         "niki_066a_eval_PasswordMatch_gcg_gcg-30_transfer",
#     ),
#     title="PM, GCG train, GCG 30 its eval",
#     save_as="pm_gcg_gcg-30_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_067_eval_WordLength_gcg_gcg-30_transfer",
#         "niki_067a_eval_WordLength_gcg_gcg-30_transfer",
#         "niki_067b_eval_WordLength_gcg_gcg-30_transfer",
#         "niki_067c_eval_WordLength_gcg_gcg-30_transfer",
#     ),
#     title="WL, GCG train, GCG 30 its eval",
#     save_as="wl_gcg_gcg-30_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
#     legend=legend,
# )


# load_and_plot_adv_training_plots(
#     run_names=("niki_051_eval_enronspam_rt_gcg_transfer",),
#     title="Spam, RT → GCG",
#     save_as="spam_rt_gcg_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
# )

# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_065_eval_enronspam_gcg_gcg-30_transfer",
#         "niki_065a_eval_enronspam_gcg_gcg-30_transfer",
#     ),
#     title="Spam, GCG → GCG 30",
#     save_as="spam_gcg_gcg-30_transfer",
#     save_dir=save_dir,
#     summary_keys=transfer_summary_keys,
#     xlim=xlim,
# )

load_and_plot_adv_training_plots(
    run_names=("niki_141_test_saved_models_evaluation",),
    title="IMDB Adv. Tr.",
    save_as="eval_test",
    summary_keys=transfer_summary_keys,
    x_data_name="adv_training_round",
    color_data_name="num_params",
    xlim=xlim,
    ylim=(0, 0.3),
    legend=True,
    check_seeds=False,
)
