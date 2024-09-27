"""Figure 1 with old data"""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import (
    create_path_and_savefig,
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


def compute_r2_by_round(raw_data: pd.DataFrame, y_value: str = "log_asr"):
    data = raw_data.rename(
        columns={
            "adversarial_eval/attack_success_rate": "asr",
        }
    )
    # data = data.loc[data.asr.between(0, 1, inclusive="neither")]
    data = data.loc[data.adv_training_round.le(10)]
    data["log_asr"] = np.log(data["asr"])
    data["logit_asr"] = np.log(data["asr"] / (1 - data["asr"]))
    data["log_params"] = np.log(data["model_size"])
    r2_list = []
    for round, round_df in data.groupby("adv_training_round"):
        reg = smf.ols(f"{y_value} ~ log_params", data=round_df).fit()
        r2_list.append(
            {
                "adv_training_round": round,
                "r2": reg.rsquared,
                "n": np.isfinite(round_df[y_value]).sum(),
            }
        )
    return pd.DataFrame(r2_list)


def plot_r2_by_round(data: pd.DataFrame, save_as: Iterable[str]):
    r2_data = compute_r2_by_round(data)
    print("R^2 by round", r2_data)
    set_plot_style("paper")
    fig, ax = plt.subplots()
    r2_data.plot(x="adv_training_round", y="r2", ax=ax)
    ax.set_ylabel("R2")
    ax.set_xlabel("Adversarial Training Round")
    fig.suptitle(";".join(save_as) + " R2 by round")
    create_path_and_savefig(fig, *save_as)


set_plot_style("paper")
YTRANSFORM = "logit"
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
        ytransform=YTRANSFORM,
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
        ytransform=YTRANSFORM,
    )

summary_keys = SUMMARY_KEYS + [
    "experiment_yaml.training.force_name_to_save",
    "experiment_yaml.training.seed",
]
ROUNDS = [0, 1, 2, 3, 4, 9, 29]
# IMDB
adv_data = prepare_adv_training_data(
    run_names=(
        "niki_052_long_adversarial_training_gcg_imdb",
        "niki_052a_long_adversarial_training_gcg_imdb",
    ),
    summary_keys=summary_keys,
    metrics=METRICS,
)
print("IMDB", adv_data)
plot_r2_by_round(adv_data, ("post_adv_training", "imdb", "gcg", "r2"))
for legend in (True, False):
    draw_min_max_median_plot_by_round(
        data=adv_data,
        title="IMDB, GCG attack (adversarial training)",
        save_as=("post_adv_training", "imdb", "gcg"),
        legend=legend,
        rounds=ROUNDS,
        ytransform=YTRANSFORM,
    )

# SPAM
adv_data = prepare_adv_training_data(
    run_names=("niki_053_long_adversarial_training_gcg_enron-spam",),
    summary_keys=summary_keys,
    metrics=METRICS,
)
print("SPAM", adv_data)
plot_r2_by_round(adv_data, ("post_adv_training", "spam", "gcg", "r2"))
for legend in (True, False):
    draw_min_max_median_plot_by_round(
        data=adv_data,
        title="Spam, GCG attack (adversarial training)",
        save_as=("post_adv_training", "spam", "gcg"),
        legend=legend,
        rounds=ROUNDS,
        ytransform=YTRANSFORM,
    )
