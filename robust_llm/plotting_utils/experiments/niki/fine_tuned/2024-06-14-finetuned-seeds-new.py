from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import make_finetuned_plots

metrics = [
    "adversarial_eval/attack_success_rate",
    "model_size",
]

summary_keys = [
    "experiment_yaml.model.name_or_path",
]

legend = False

set_plot_style("paper")
save_dir = "finetuned"

make_finetuned_plots(
    run_names=(
        "niki_103_eval_imdb_gcg_seeds",
        "niki_104_eval_imdb_gcg_seeds_big",
        "niki_104a_eval_imdb_gcg_seeds_big",
    ),
    title="IMDB, GCG attack",
    save_as=(save_dir, "imdb_gcg_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
make_finetuned_plots(
    run_names=(
        "niki_110_eval_spam_gcg_seeds_small",
        "niki_111_eval_spam_gcg_seeds_big",
        "niki_111a_eval_spam_gcg_seeds_big",
    ),
    title="Spam, GCG attack",
    save_as=(save_dir, "spam_gcg_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
make_finetuned_plots(
    run_names=(
        "niki_112_eval_pm_gcg_seeds_small",
        "niki_113_eval_pm_gcg_seeds_big",
    ),
    title="PasswordMatch, GCG attack",
    save_as=(save_dir, "pm_gcg_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
make_finetuned_plots(
    run_names=(
        "niki_114_eval_wl_gcg_seeds_small",
        "niki_115_eval_wl_gcg_seeds_big",
    ),
    title="WordLength, GCG attack",
    save_as=(save_dir, "wl_gcg_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)

make_finetuned_plots(
    run_names=(
        "niki_116_eval_imdb_rt_seeds_small",
        "niki_117_eval_imdb_rt_seeds_big",
    ),
    title="IMDB, RT attack",
    save_as=(save_dir, "imdb_rt_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
make_finetuned_plots(
    run_names=(
        "niki_118_eval_spam_rt_seeds_small",
        "niki_119_eval_spam_rt_seeds_big",
    ),
    title="Spam, RT attack",
    save_as=(save_dir, "spam_rt_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
make_finetuned_plots(
    run_names=(
        "niki_120_eval_pm_rt_seeds_small",
        "niki_121_eval_pm_rt_seeds_big",
    ),
    title="PasswordMatch, RT attack",
    save_as=(save_dir, "pm_rt_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
make_finetuned_plots(
    run_names=(
        "niki_122_eval_wl_rt_seeds_small",
        "niki_123_eval_wl_rt_seeds_big",
    ),
    title="WordLength, RT attack",
    save_as=(save_dir, "wl_rt_attack"),
    eval_summary_keys=summary_keys,
    metrics=metrics,
    legend=legend,
)
