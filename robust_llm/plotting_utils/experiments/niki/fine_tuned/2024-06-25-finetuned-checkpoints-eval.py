from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import make_finetuned_plot

metrics = [
    "adversarial_eval/attack_success_rate",
    "model_size",
]

summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.evaluation.evaluation_attack.n_its",
]

save_dir = "checkpoints"

set_plot_style("paper")

# make_finetuned_plot(
#     run_names=("niki_105_gcg_imdb_eval_7checkpoints",),
#     title="IMDB GCG 70% Checkpoints",
#     save_as="imdb_gcg_70%_checkpoints",
#     save_dir=save_dir,
#     eval_summary_keys=summary_keys,
#     metrics=metrics,
#     plot_type="scatter",
# )
# make_finetuned_plot(
#     run_names=("niki_106_gcg_pm_eval_7checkpoints",),
#     title="PM GCG 70% Checkpoints",
#     save_as="pm_gcg_70%_checkpoints",
#     save_dir=save_dir,
#     eval_summary_keys=summary_keys,
#     metrics=metrics,
#     plot_type="scatter",
# )
make_finetuned_plot(
    run_names=("niki_107_gcg_imdb_eval_last_checkpoints",),
    title="IMDB GCG 97% Checkpoints",
    save_as=(save_dir, "imdb_gcg_97%_checkpoints"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    plot_type="scatter",
)
make_finetuned_plot(
    run_names=("niki_108_gcg_spam_eval_last_checkpoints",),
    title="Spam GCG 97% Checkpoints",
    save_as=(save_dir, "spam_gcg_97%_checkpoints"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    plot_type="scatter",
)
make_finetuned_plot(
    run_names=("niki_109_gcg_pm_eval_last_checkpoints",),
    title="PM GCG 97% Checkpoints",
    save_as=(save_dir, "pm_gcg_97%_checkpoints"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    plot_type="scatter",
)
