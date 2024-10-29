"""Basic finetuned plots (robustness vs. size)"""

import argparse

import numpy as np
import pandas as pd

from robust_llm.plotting_utils.style import name_to_attack, name_to_dataset, set_style
from robust_llm.plotting_utils.tools import (
    draw_min_max_median_plot,
    draw_min_max_median_plot_by_dataset,
    read_csv_and_metadata,
)


def main(style):
    set_style(style)

    y_data_name = "adversarial_eval_attack_success_rate"
    all_data = []
    metadata = None
    for attack in ("gcg", "rt"):
        for dataset in ("imdb", "spam", "wl", "pm", "helpful", "harmless"):
            save_as = ("finetuned", attack, dataset)
            data, metadata = read_csv_and_metadata(*save_as)
            for legend in (True, False):
                for ytransform in ("logit", None):
                    draw_min_max_median_plot(
                        data,
                        metadata=metadata,
                        title=f"{name_to_dataset(dataset)}, "
                        f"{name_to_attack(attack)} Attack",
                        legend=legend,
                        y_data_name=y_data_name,
                        ytransform=ytransform,
                        save_as=save_as,
                        style=style,
                    )
            data["attack"] = attack
            data["dataset"] = dataset
            all_data.append(data)
    concat_data = pd.concat(all_data)
    concat_data["asr"] = np.select(
        [
            concat_data.dataset.isin(["pm", "imdb", "spam"])
            & concat_data.attack.eq("gcg"),
            concat_data.dataset.isin(["helpful", "harmless", "wl"])
            & concat_data.attack.eq("gcg"),
            concat_data.dataset.isin(["pm", "imdb", "spam"])
            & concat_data.attack.eq("rt"),
            concat_data.dataset.isin(["helpful", "harmless", "wl"])
            & concat_data.attack.eq("rt"),
        ],
        [
            concat_data.asr_at_10,
            concat_data.asr_at_2,
            concat_data.asr_at_1200,
            concat_data.asr_at_1200,
        ],
    )
    assert not concat_data.asr.isnull().any()
    for attack, attack_df in concat_data.groupby("attack"):
        assert isinstance(attack, str)
        for legend in (True, False):
            for ytransform in ("logit", None):
                draw_min_max_median_plot_by_dataset(
                    attack_df,
                    metadata=metadata,
                    title=f"{name_to_attack(attack)} Attack on All Tasks",
                    y_data_name="asr",
                    ytransform=ytransform,
                    save_as=("finetuned", attack, "all"),
                    legend=legend,
                    style=style,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot finetuned robustness vs. size")
    parser.add_argument(
        "--style",
        type=str,
        default="paper",
        help="Style to use for plotting",
    )
    args = parser.parse_args()
    main(args.style)
