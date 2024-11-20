import argparse

import pandas as pd

from robust_llm.plotting_utils.style import name_to_attack, name_to_dataset
from robust_llm.plotting_utils.tools import (
    draw_min_max_median_plot_by_dataset,
    draw_min_max_median_plot_by_round,
    read_csv_and_metadata,
)


def main(style: str = "paper"):
    ROUNDS = [0, 1, 2, 3, 4, 5, 10]
    all_data = []
    metadata = None
    for attack in ("gcg_gcg",):
        for dataset in ("imdb", "spam", "wl", "pm"):
            adv_data, metadata = read_csv_and_metadata(
                "post_adv_training", attack, dataset
            )
            for legend in (True, False):
                for ytransform in ("logit", "none"):
                    for y_data_name in ("metrics_asr_at_12", "metrics_asr_at_128"):
                        train_attack, eval_attack = attack.split("_")
                        draw_min_max_median_plot_by_round(
                            adv_data,
                            metadata,
                            title=f" {name_to_attack(eval_attack)}, "
                            f"{name_to_dataset(dataset)}",
                            save_as=("post_adv_training", attack, dataset),
                            legend=legend,
                            rounds=ROUNDS,
                            ytransform=ytransform,
                            y_data_name=y_data_name,
                            style=style,
                        )
            adv_data["attack"] = attack
            adv_data["dataset"] = dataset
            all_data.append(adv_data)
    concat_data = pd.concat(all_data)
    concat_data = concat_data.loc[concat_data.adv_training_round.eq(0)]
    for attack, attack_df in concat_data.groupby("attack"):
        assert isinstance(attack, str)
        for legend in (True, False):
            for ytransform in ("logit", "none"):
                for y_data_name in ("metrics_asr_at_12", "metrics_asr_at_128"):
                    draw_min_max_median_plot_by_dataset(
                        attack_df,
                        metadata,
                        title=f"{name_to_attack(attack)} Attack on All Tasks",
                        save_as=("post_adv_training", attack, "all", "round0"),
                        ytransform=ytransform,
                        y_data_name=y_data_name,
                        legend=legend,
                        style=style,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial training plots")
    parser.add_argument(
        "--style",
        type=str,
        default="paper",
        help="Style to be used for plotting",
    )
    args = parser.parse_args()
    main(args.style)
