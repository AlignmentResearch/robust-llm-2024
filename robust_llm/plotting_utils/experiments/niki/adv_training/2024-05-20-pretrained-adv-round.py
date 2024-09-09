from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

save_dir = "pretrained"

# Set the plot style
set_plot_style("paper")

load_and_plot_adv_training_plots(
    run_names=("niki_056_pretrained_adv-training_rt_imdb",),
    title="IMDB, RT, Pretrained",
    save_as="imdb_rt_pretrained_adv_training_10_rounds",
    save_dir=save_dir,
)
# load_and_plot_adv_training_plots(
#     "niki_057_pretrained_adv-training_rt_enronspam",
#     title="Spam, RT, Pretrained",
#     save_as="spam_rt_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
#     legend=True,
# )
# load_and_plot_adv_training_plots(
#     "niki_058_pretrained_adv-training_rt_pm",
#     title="PM, RT, Pretrained",
#     save_as="pm_rt_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_059_pretrained_adv-training_rt_wl",
#     title="WL, RT, Pretrained",
#     save_as="wl_rt_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
# )

# load_and_plot_adv_training_plots(
#     "niki_060_pretrained_adv-training_gcg_imdb",
#     title="IMDB, GCG, Pretrained",
#     save_as="imdb_gcg_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_061_pretrained_adv-training_gcg_enronspam",
#     title="Spam, GCG, Pretrained",
#     save_as="spam_gcg_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
#     legend=True,
# )
# load_and_plot_adv_training_plots(
#     "niki_062_pretrained_adv-training_gcg_pm",
#     title="PM, GCG, Pretrained",
#     save_as="pm_gcg_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_063_pretrained_adv-training_gcg_wl",
#     title="WL, GCG, Pretrained",
#     save_as="wl_gcg_pretrained_adv_training_10_rounds",
#     save_dir=save_dir,
# )

# load_and_plot_adv_training_plots(
#     "niki_056_pretrained_adv-training_rt_imdb",
#     title="IMDB, RT, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="imdb_rt_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_057_pretrained_adv-training_rt_enronspam",
#     title="Spam, RT, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="spam_rt_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_058_pretrained_adv-training_rt_pm",
#     title="PM, RT, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="pm_rt_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_059_pretrained_adv-training_rt_wl",
#     title="WL, RT, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="wl_rt_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )

# load_and_plot_adv_training_plots(
#     "niki_060_pretrained_adv-training_gcg_imdb",
#     title="IMDB, GCG, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="imdb_gcg_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_061_pretrained_adv-training_gcg_enronspam",
#     title="Spam, GCG, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="spam_gcg_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_062_pretrained_adv-training_gcg_pm",
#     title="PM, GCG, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="pm_gcg_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
# load_and_plot_adv_training_plots(
#     "niki_063_pretrained_adv-training_gcg_wl",
#     title="WL, GCG, Pretrained",
#     x_data_name="num_params",
#     color_data_name="adv_training_round",
#     save_as="wl_gcg_pretrained_adv_training_10_rounds_num_params",
#     save_dir=save_dir,
# )
