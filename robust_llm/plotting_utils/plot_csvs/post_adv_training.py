import argparse

import pandas as pd

from robust_llm.plotting_utils.style import (
    name_to_attack,
    name_to_dataset,
    name_to_model,
)
from robust_llm.plotting_utils.tools import (
    draw_min_max_median_plot_by_dataset,
    draw_min_max_median_plot_by_round,
    read_csv_and_metadata,
)


def main(style: str = "paper"):
    ROUNDS = [0, 1, 2, 3, 4, 5, 10]
    all_data = []
    metadata = None
    for family, attack, dataset in [
        ("pythia", "gcg_gcg", "imdb"),
        ("pythia", "gcg_gcg", "spam"),
        ("pythia", "gcg_gcg", "wl"),
        ("pythia", "gcg_gcg", "pm"),
        ("qwen", "gcg_gcg", "harmless"),
        ("qwen", "gcg_gcg", "spam"),
    ]:
        adv_data, metadata = read_csv_and_metadata(
            "post_adv_training", family, attack, dataset
        )
        for legend in (True, False):
            for ytransform in ("logit", "none"):
                for y_data_name in ("asr_at_128",):
                    train_attack, eval_attack = attack.split("_")
                    draw_min_max_median_plot_by_round(
                        adv_data,
                        metadata,
                        title=(
                            f"{name_to_model(family)}, "
                            f"{name_to_attack(eval_attack)}, "
                            f"{name_to_dataset(dataset)}"
                        ),
                        family=family,
                        attack=attack,
                        dataset=dataset,
                        legend=legend,
                        rounds=ROUNDS,
                        ytransform=ytransform,
                        y_data_name=y_data_name,
                        style=style,
                    )
        adv_data["family"] = family
        adv_data["attack"] = attack
        adv_data["dataset"] = dataset
        all_data.append(adv_data)
    concat_data = pd.concat(all_data)
    concat_data = concat_data.loc[concat_data.adv_training_round.eq(0)]
    for (family, attack), attack_df in concat_data.groupby(["family", "attack"]):
        assert isinstance(attack, str)
        for legend in (True, False):
            for ytransform in ("logit", "none"):
                for y_data_name in ("asr_at_128",):
                    draw_min_max_median_plot_by_dataset(
                        attack_df,
                        metadata,
                        title=(
                            f"{name_to_model(family)}, {name_to_attack(attack)} "
                            "Attack on All Tasks"
                        ),
                        family=family,
                        attack=attack,
                        adversarial=True,
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
