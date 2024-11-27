"""Basic finetuned plots (robustness vs. size)"""

import argparse

import numpy as np
import pandas as pd

from robust_llm.plotting_utils.style import (
    name_to_attack,
    name_to_dataset,
    name_to_model,
    set_style,
)
from robust_llm.plotting_utils.tools import (
    draw_min_max_median_plot,
    draw_min_max_median_plot_by_dataset,
    draw_wilson_score_interval_plot,
    read_csv_and_metadata,
)


def pick_attack_iterations(df, attack, dataset):
    # We handle strongreject separately
    if dataset in [
        "strongreject",
    ]:
        # In strongreject, we only calculated the ASR for the final attack iteration.
        df["asr"] = df["adversarial_eval_attack_success_rate"]
        return

    for field in ("asr", "mean_log_prob", "log_mean_prob"):
        if not df.columns.str.contains(f"{field}_at_").any():
            print(f"Missing {field}_at_ for {attack} {dataset}")
            continue
        elif dataset in ["pm", "imdb", "spam"] and attack == "gcg":
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


def _plot_strongreject(style: str) -> None:
    attack = "gcg"
    dataset = "strongreject"
    model = "qwen25"
    save_as = ("finetuned", model, attack, dataset)
    data, metadata = read_csv_and_metadata(*save_as)
    data["attack"] = attack
    data["dataset"] = dataset
    success_name = "adversarial_eval_n_incorrect_post_attack"
    trials_name = "adversarial_eval_n_correct_pre_attack"
    pick_attack_iterations(data, attack, dataset)
    for legend in (True, False):
        for y_data_name, ytransform in [("asr", None), ("asr", "logit")]:
            if y_data_name not in data.columns:
                print(f"Couldn't find {y_data_name} in {attack} {dataset}, skipping...")
                continue

            draw_wilson_score_interval_plot(
                data,
                metadata=metadata,
                title=(
                    f"{name_to_model(model)}, "
                    f"{name_to_dataset(dataset)}, "
                    f"{name_to_attack(attack)} Attack"
                ),
                successes_name=success_name,
                trials_name=trials_name,
                legend=legend,
                y_data_name=y_data_name,
                ytransform=ytransform,
                save_as=save_as,
                style=style,
                smoothing=0 if "prob" in y_data_name else 1,
                ylim=(0, 1) if ytransform is None else None,
            )

    # Pre attack accuracy
    success_name = "adversarial_eval_n_correct_pre_attack"
    trials_name = "adversarial_eval_n_examples"
    draw_wilson_score_interval_plot(
        data,
        metadata=metadata,
        title=f"{name_to_model(model)} Pre-Attack Accuracy",
        successes_name=success_name,
        trials_name=trials_name,
        y_data_name="adversarial_eval_pre_attack_accuracy",
        ytransform="none",
        save_as=("finetuned", model, "pre_attack_accuracy"),
        legend=True,
        style=style,
        smoothing=1,
        ylim=(0, 1),
    )

    # Post attack accuracy
    success_name = "adversarial_eval_n_correct_post_attack"
    trials_name = "adversarial_eval_n_examples"
    for attack, attack_df in data.groupby("attack"):
        title = f"{name_to_model(model)} Post-Attack Accuracy ({str(attack).upper()})"
        draw_wilson_score_interval_plot(
            attack_df,
            metadata=metadata,
            title=title,
            successes_name=success_name,
            trials_name=trials_name,
            y_data_name="post_attack_accuracy",
            ytransform="none",
            save_as=(
                "finetuned",
                model,
                f"post_{str(attack)}_attack_accuracy",
            ),
            legend=True,
            style=style,
            smoothing=1,
            ylim=(0, 1),
        )


def _plot_other_datasets(style: str) -> None:
    all_data = []
    metadata = None
    for attack in ("gcg", "rt"):
        for dataset in ("imdb", "spam", "wl", "pm", "helpful", "harmless"):
            save_as = ("finetuned", "pythia", attack, dataset)
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
                        print(
                            f"Couldn't find {y_data_name} in "
                            f"{attack} {dataset}, skipping..."
                        )
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
                        ylim=(0, 1) if ytransform is None else None,
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
                ("mean_log_prob", "negative"),
                ("log_mean_prob", "comp_exp"),
            ]:
                draw_min_max_median_plot_by_dataset(
                    attack_df,
                    metadata=metadata,
                    title=f"{name_to_attack(attack)} Attack on All Tasks",
                    y_data_name=y_data_name,
                    ytransform=ytransform,
                    save_as=("finetuned", "pythia", attack, "all"),
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
                    save_as=("finetuned", "pythia", attack, "all_except_wl"),
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
        save_as=("finetuned", "pythia", "pre_attack_accuracy", "all"),
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
                "pythia",
                f"post_{str(attack)}_attack_accuracy",
                "all",
            ),
            legend=True,
            legend_loc="upper left",
            style=style,
            smoothing=1,
            ylim=(0, 1),
        )


def main(style):
    set_style(style)
    _plot_strongreject(style)
    _plot_other_datasets(style)


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
