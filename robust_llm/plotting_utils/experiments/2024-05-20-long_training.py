from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Set the xlim here for longer or shorter number of adversarial training rounds
xlim = (1, 10)

# Make plots for the four datasets and two attacks
load_and_plot_adv_training_plots(
    "niki_044a_long_adversarial_training_random_token_imdb_save",
    title="IMDB, RT attack",
    save_as="imdb_rt_adv_training_10_rounds",
    xlim=xlim,
)
load_and_plot_adv_training_plots(
    "niki_046_long_adv_training_enron-spam_rt",
    title="Spam, RT attack",
    save_as="spam_rt_adv_training_10_rounds",
    xlim=xlim,
    legend=True,
)
load_and_plot_adv_training_plots(
    "niki_045_long_adv_training_password-match_rt",
    title="PM, RT attack",
    save_as="pm_rt_adv_training_10_rounds",
    xlim=xlim,
)
load_and_plot_adv_training_plots(
    "niki_047_long_adv_training_word-length_rt",
    title="WL, RT attack",
    save_as="wl_rt_adv_training_10_rounds",
    xlim=xlim,
)
load_and_plot_adv_training_plots(
    "niki_052_long_adversarial_training_gcg_imdb",
    title="IMDB, GCG attack",
    save_as="imdb_gcg_adv_training_10_rounds",
    xlim=xlim,
)
load_and_plot_adv_training_plots(
    "niki_053_long_adversarial_training_gcg_enron-spam",
    title="Spam, GCG attack",
    save_as="spam_gcg_adv_training_10_rounds",
    xlim=xlim,
    legend=True,
)
load_and_plot_adv_training_plots(
    "niki_054_long_adversarial_training_gcg_pm",
    title="PM, GCG attack",
    save_as="pm_gcg_adv_training_10_rounds",
    xlim=xlim,
)
load_and_plot_adv_training_plots(
    "niki_055_long_adversarial_training_gcg_wl",
    title="WL, GCG attack",
    save_as="wl_gcg_adv_training_10_rounds",
    xlim=xlim,
)
