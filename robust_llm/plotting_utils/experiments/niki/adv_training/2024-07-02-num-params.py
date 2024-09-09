from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Set the xlim here for longer or shorter number of adversarial training rounds
num_rounds = 10
xlim = (1, num_rounds)

legend = False

# Set the plot style
set_plot_style("paper")


# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_044a_long_adversarial_training_random_token_imdb_save",
#         "niki_044b_long_adversarial_training_random_token_imdb_save",
#     ),
#     title="IMDB, RT attack",
#     save_as=f"imdb_rt_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_046_long_adv_training_enron-spam_rt",
#         "niki_046a_long_adv_training_enron-spam_rt",
#     ),
#     title="Spam, RT attack",
#     save_as=f"spam_rt_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_045_long_adv_training_password-match_rt",),
#     title="PM, RT attack",
#     save_as=f"pm_rt_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_047_long_adv_training_word-length_rt",),
#     title="WL, RT attack",
#     save_as=f"wl_rt_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_052_long_adversarial_training_gcg_imdb",
#         "niki_052a_long_adversarial_training_gcg_imdb",
#     ),
#     title="IMDB, GCG attack",
#     save_as=f"imdb_gcg_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
#     title="Spam, GCG attack",
#     save_as=f"spam_gcg_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
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
#     save_as=f"pm_gcg_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )
# load_and_plot_adv_training_plots(
#     run_names=(
#         "niki_055_long_adversarial_training_gcg_wl",
#         "niki_055a_long_adversarial_training_gcg_wl",
#     ),
#     title="WL, GCG attack",
#     save_as=f"wl_gcg_adv_training_{num_rounds}_rounds_num_params",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     xlim=xlim,
#     legend=legend,
# )


load_and_plot_adv_training_plots(
    run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
    title="Spam, GCG attack",
    save_as="legend_reference",
    x_data_name="num_params",
    color_data_name="adv_training_round",
    xlim=xlim,
    legend=True,
)
