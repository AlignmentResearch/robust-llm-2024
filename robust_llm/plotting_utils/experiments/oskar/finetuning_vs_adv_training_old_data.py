"""Figure 1 with old data"""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import (
    draw_min_max_median_plot_by_round,
    make_finetuned_plots,
    prepare_adv_training_data,
)
from robust_llm.wandb_utils.constants import METRICS, SUMMARY_KEYS

metrics = [
    "adversarial_eval/attack_success_rate",
    "model_size",
]

summary_keys = [
    "experiment_yaml.model.name_or_path",
]


set_plot_style("paper")

for legend in (True, False):
    make_finetuned_plots(
        run_names=[
            "niki_103_eval_imdb_gcg_seeds",
            "niki_104_eval_imdb_gcg_seeds_big",
            "niki_104a_eval_imdb_gcg_seeds_big",
        ],
        title="IMDB, GCG attack",
        save_as=("finetuned", "imdb", "gcg"),
        eval_summary_keys=summary_keys,
        metrics=metrics,
        legend=legend,
    )
    make_finetuned_plots(
        run_names=[
            "niki_110_eval_spam_gcg_seeds_small",
            "niki_111_eval_spam_gcg_seeds_big",
            "niki_111a_eval_spam_gcg_seeds_big",
        ],
        title="Spam, GCG attack",
        save_as=("finetuned", "spam", "gcg"),
        eval_summary_keys=summary_keys,
        metrics=metrics,
        legend=legend,
    )

summary_keys = SUMMARY_KEYS + [
    "experiment_yaml.training.force_name_to_save",
    "experiment_yaml.training.seed",
]
ROUNDS = [4, 9, 14, 19, 24, 29]
# IMDB
adv_data = prepare_adv_training_data(
    run_names=(
        "niki_052_long_adversarial_training_gcg_imdb",
        "niki_052a_long_adversarial_training_gcg_imdb",
    ),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
)
print("IMDB", adv_data)
for legend in (True, False):
    draw_min_max_median_plot_by_round(
        data=adv_data,
        title="IMDB, GCG attack (adversarial training)",
        save_as=("post_adv_training", "imdb", "gcg"),
        legend=legend,
        rounds=ROUNDS,
    )

# SPAM
adv_data = prepare_adv_training_data(
    run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
)
print("SPAM", adv_data)
for legend in (True, False):
    draw_min_max_median_plot_by_round(
        data=adv_data,
        title="Spam, GCG attack (adversarial training)",
        save_as=("post_adv_training", "spam", "gcg"),
        legend=legend,
        rounds=ROUNDS,
    )
