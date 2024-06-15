from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Pretrained
load_and_plot_adv_training_plots(
    "niki_056_pretrained_adv-training_rt_imdb", title="Random Token, IMDB, Pretrained"
)
load_and_plot_adv_training_plots(
    "niki_057_pretrained_adv-training_rt_enronspam",
    title="Random Token, EnronSpam, Pretrained",
)
load_and_plot_adv_training_plots(
    "niki_058_pretrained_adv-training_rt_pm",
    title="Random Token, PasswordMatch, Pretrained",
)
load_and_plot_adv_training_plots(
    "niki_059_pretrained_adv-training_rt_wl",
    title="Random Token, WordLength, Pretrained",
)
load_and_plot_adv_training_plots(
    "niki_060_pretrained_adv-training_gcg_imdb", title="GCG, IMDB, Pretrained"
)
load_and_plot_adv_training_plots(
    "niki_061_pretrained_adv-training_gcg_enronspam", title="GCG, EnronSpam, Pretrained"
)
load_and_plot_adv_training_plots(
    "niki_062_pretrained_adv-training_gcg_pm", title="GCG, PasswordMatch, Pretrained"
)
load_and_plot_adv_training_plots(
    "niki_063_pretrained_adv-training_gcg_wl", title="GCG, WordLength, Pretrained"
)
