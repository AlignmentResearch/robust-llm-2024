from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots
from robust_llm.wandb_utils.constants import SUMMARY_KEYS

# Set the xlim here for longer or shorter number of adversarial training rounds
num_rounds = 30
xlim = (1, num_rounds)

legend = False

# Set the plot style
set_plot_style("paper")

# Include the name for easier debugging
summary_keys = SUMMARY_KEYS + ["experiment_yaml.training.force_name_to_save"]

# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_044a_long_adversarial_training_random_token_imdb_save",
#         "niki_044b_long_adversarial_training_random_token_imdb_save",
#     ),
#     title="IMDB, RandomToken attack",
#     save_as=f"imdb_rt_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_046_long_adv_training_enron-spam_rt",
#         "niki_046a_long_adv_training_enron-spam_rt",
#     ),
#     title="Spam, RandomToken attack",
#     save_as=f"spam_rt_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_045_long_adv_training_password-match_rt",),
#     title="PM, RandomToken attack",
#     save_as=f"pm_rt_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_047_long_adv_training_word-length_rt",),
#     title="WL, RandomToken attack",
#     save_as=f"wl_rt_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_052_long_adversarial_training_gcg_imdb",
#         "niki_052a_long_adversarial_training_gcg_imdb",
#     ),
#     title="IMDB, GCG attack",
#     save_as=f"imdb_gcg_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
#     title="Spam, GCG attack",
#     save_as=f"spam_gcg_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_054_long_adversarial_training_gcg_pm",
#         "niki_054a_long_adversarial_training_gcg_pm",
#         "niki_054b_long_adversarial_training_gcg_pm",
#     ),
#     title="PM, GCG attack",
#     save_as=f"pm_gcg_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_055_long_adversarial_training_gcg_wl",
#         "niki_055a_long_adversarial_training_gcg_wl",
#     ),
#     title="WL, GCG attack",
#     save_as=f"wl_gcg_adv_training_{num_rounds}_rounds_flops",
#     summary_keys=summary_keys,
#     x_data_name="n_parameter_updates",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=legend,
# )

# load_and_plot_adv_training_plots(
#     run_names=("niki_047_long_adv_training_word-length_rt",),
#     title="WL, RandomToken attack",
#     save_as="adv_tr_legend_reference",
#     summary_keys=summary_keys,
#     x_data_name="adv_training_round",
#     color_data_name="num_params",
#     xlim=xlim,
#     legend=True,
# )

load_and_plot_adv_training_plots(
    run_names=("niki_132_test_long_adv_tr",),
    title="IMDB Adv. Tr. # Param. Updates",
    save_as="test_132_updates",
    summary_keys=summary_keys,
    x_data_name="n_parameter_updates",
    color_data_name="num_params",
    xlim=xlim,
    ylim=(0, 0.3),
    legend=True,
    check_seeds=False,
)

load_and_plot_adv_training_plots(
    run_names=("niki_133_more_adv_tr",),
    title="IMDB Adv. Tr. # Param. Updates",
    save_as="test_133_updates",
    summary_keys=summary_keys,
    x_data_name="n_parameter_updates",
    color_data_name="num_params",
    xlim=xlim,
    ylim=(0, 0.3),
    legend=True,
    check_seeds=False,
)
