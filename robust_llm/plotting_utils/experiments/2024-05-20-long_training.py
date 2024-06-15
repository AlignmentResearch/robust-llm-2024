from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

# Training from scratch
load_and_plot_adv_training_plots(
    "niki_044a_long_adversarial_training_random_token_imdb_save",
    title="Random Token, IMDB",
    use_size_from_name=True,
)
load_and_plot_adv_training_plots(
    "niki_046_long_adv_training_enron-spam_rt", title="Random Token, EnronSpam"
)
load_and_plot_adv_training_plots(
    "niki_045_long_adv_training_password-match_rt", title="Random Token, PasswordMatch"
)
load_and_plot_adv_training_plots(
    "niki_047_long_adv_training_word-length_rt", title="Random Token, WordLength"
)
load_and_plot_adv_training_plots(
    "niki_052_long_adversarial_training_gcg_imdb", title="GCG, IMDB"
)
load_and_plot_adv_training_plots(
    "niki_053_long_adversarial_training_gcg_enron-spam", title="GCG, EnronSpam"
)
load_and_plot_adv_training_plots(
    "niki_054_long_adversarial_training_gcg_pm", title="GCG, PasswordMatch"
)
load_and_plot_adv_training_plots(
    "niki_055_long_adversarial_training_gcg_wl", title="GCG, WordLength"
)
