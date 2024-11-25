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


def pick_attack_iterations(df, attack, dataset):
    for field in ("asr",):
        if not df.columns.str.contains(f"{field}_at_").any():
            print(f"Missing {field}_at_ for {attack} {dataset}")
            continue
        if dataset in ["pm", "imdb", "spam"] and attack == "gcg":
            df[field] = df[f"{field}_at_10"]
        elif dataset in ["helpful", "harmless", "wl"] and attack == "gcg":
            df[field] = df[f"{field}_at_2"]
        elif (
            dataset in ["pm", "imdb", "spam", "helpful", "harmless", "wl"]
            and attack == "rt"
        ):
            df[field] = df[f"{field}_at_1200"]
        if field == "log_mean_prob":
            df["mean_prob"] = np.exp(df["log_mean_prob"])


def main(style):
    set_style(style)

    all_data = []
    metadata = None
    for attack in ("gcg",):
        for dataset in ("spam", "harmless"):
            save_as = ("finetuned", "qwen25", attack, dataset)
            data, metadata = read_csv_and_metadata(*save_as)
            pick_attack_iterations(data, attack, dataset)
            for legend in (True, False):
                for y_data_name, ytransform in [
                    ("asr", "logit"),
                    ("asr", None),
                    ("mean_log_prob", "negative"),
                    ("log_mean_prob", "comp_exp"),
                ]:
                    if y_data_name not in data.columns:
                        continue
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
                        smoothing=0 if "prob" in y_data_name else 1,
                    )
            data["attack"] = attack
            data["dataset"] = dataset
            all_data.append(data)
    concat_data = pd.concat(all_data)
    assert not concat_data.asr.isnull().any()
    for attack, attack_df in concat_data.groupby("attack"):
        assert isinstance(attack, str)
        for legend in (True, False):
            for y_data_name, ytransform in [
                ("asr", "logit"),
                ("asr", None),
            ]:
                draw_min_max_median_plot_by_dataset(
                    attack_df,
                    metadata=metadata,
                    title=f"{name_to_attack(attack)} Attack on All Tasks",
                    y_data_name=y_data_name,
                    ytransform=ytransform,
                    save_as=("finetuned", "qwen25", attack, "all"),
                    legend=legend,
                    style=style,
                    smoothing=0 if "prob" in y_data_name else 1,
                )
                draw_min_max_median_plot_by_dataset(
                    attack_df.loc[attack_df.dataset.ne("wl")],
                    metadata=metadata,
                    title=f"{name_to_attack(attack)} Attack on All Tasks",
                    y_data_name=y_data_name,
                    ytransform=ytransform,
                    save_as=("finetuned", "qwen25", attack, "all_except_wl"),
                    legend=legend,
                    style=style,
                    smoothing=0 if "prob" in y_data_name else 1,
                )

    # Pre attack accuracy
    gcg_dataset = concat_data.loc[concat_data.attack.eq("gcg")]
    draw_min_max_median_plot_by_dataset(
        gcg_dataset,
        metadata=metadata,
        title="Pre-Attack Accuracy",
        y_data_name="adversarial_eval_pre_attack_accuracy",
        ytransform="none",
        save_as=("finetuned", "qwen25", "pre_attack_accuracy", "all"),
        legend=True,
        style=style,
        smoothing=1,
        ylim=(0, 1),
    )

    # Post attack accuracy
    for attack, attack_df in concat_data.groupby("attack"):
        draw_min_max_median_plot_by_dataset(
            attack_df,
            metadata=metadata,
            title=f"Post-Attack Accuracy ({str(attack).upper()})",
            y_data_name="post_attack_accuracy",
            ytransform="none",
            save_as=(
                "finetuned",
                "qwen25",
                f"post_{str(attack)}_attack_accuracy",
                "all",
            ),
            legend=True,
            legend_loc="upper left",
            style=style,
            smoothing=1,
            ylim=(0, 1),
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
