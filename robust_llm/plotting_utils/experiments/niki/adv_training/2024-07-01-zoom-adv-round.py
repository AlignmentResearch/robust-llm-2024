from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Set the xlim here for longer or shorter number of adversarial training rounds
zoom_xlim = (20, 30)

legend = False

# Set the plot style
set_plot_style("paper")

load_and_plot_adv_training_plots(
    run_names=(
        "niki_044a_long_adversarial_training_random_token_imdb_save",
        "niki_044b_long_adversarial_training_random_token_imdb_save",
    ),
    title="IMDB, RandomToken attack",
    save_as="imdb_rt_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.3),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=(
        "niki_046_long_adv_training_enron-spam_rt",
        "niki_046a_long_adv_training_enron-spam_rt",
    ),
    title="Spam, RandomToken attack",
    save_as="spam_rt_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.1),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=("niki_045_long_adv_training_password-match_rt",),
    title="PM, RandomToken attack",
    save_as="pm_rt_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.01),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=("niki_047_long_adv_training_word-length_rt",),
    title="WL, RandomToken attack",
    save_as="wl_rt_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.25),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=(
        "niki_052_long_adversarial_training_gcg_imdb",
        "niki_052a_long_adversarial_training_gcg_imdb",
    ),
    title="IMDB, GCG attack",
    save_as="imdb_gcg_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.2),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
    title="Spam, GCG attack",
    save_as="spam_gcg_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.1),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=(
        "niki_054_long_adversarial_training_gcg_pm",
        "niki_054a_long_adversarial_training_gcg_pm",
        "niki_054b_long_adversarial_training_gcg_pm",
    ),
    title="PM, GCG attack",
    save_as="pm_gcg_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.01),
    legend=legend,
)
load_and_plot_adv_training_plots(
    run_names=(
        "niki_055_long_adversarial_training_gcg_wl",
        "niki_055a_long_adversarial_training_gcg_wl",
    ),
    title="WL, GCG attack",
    save_as="wl_gcg_adv_training_zoom",
    xlim=zoom_xlim,
    ylim=(0, 0.15),
    legend=legend,
)
