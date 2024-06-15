from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

transfer_summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
]


# Pretrained
load_and_plot_adv_training_plots(
    "niki_048_eval_pm_rt_gcg_transfer",
    title="transfer/PasswordMatch, RT train, GCG eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_049_eval_wl_rt_gcg_transfer",
    title="transfer/WordLength, RT train, GCG eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_050_eval_imdb_rt_gcg_transfer",
    title="transfer/IMDB, RT train, GCG eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_051_eval_enronspam_rt_gcg_transfer",
    title="transfer/EnronSpam, RT train, GCG eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_064_eval_imdb_gcg_gcg-30_transfer",
    title="transfer/IMDB, GCG 10 its train, GCG 30 its eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_065_eval_enronspam_gcg_gcg-30_transfer",
    title="transfer/EnronSpam, GCG 10 its train, GCG 30 its eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_066_eval_PasswordMatch_gcg_gcg-30_transfer",
    title="transfer/PasswordMatch, GCG 10 its train, GCG 30 its eval",
    summary_keys=transfer_summary_keys,
)
load_and_plot_adv_training_plots(
    "niki_067_eval_WordLength_gcg_gcg-30_transfer",
    title="transfer/WordLength, GCG 10 its train, GCG 30 its eval",
    summary_keys=transfer_summary_keys,
)
