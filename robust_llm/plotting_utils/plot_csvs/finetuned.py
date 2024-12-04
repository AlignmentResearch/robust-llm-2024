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
    draw_min_max_median_and_wilson_plot_by_dataset,
    draw_min_max_median_plot,
    draw_min_max_median_plot_by_dataset,
    draw_wilson_score_interval_plot,
    read_csv_and_metadata,
)


def pick_attack_iterations(df, attack, dataset, family: str = "pythia") -> None:
    # We handle strongreject separately
    if dataset in ["strongreject"]:
        # In strongreject, we only calculated the ASR for the final attack iteration.
        # For Qwen, we use the last iteration.
        df["asr"] = df.asr_at_128
        return

    for field in ("asr",):  # Other options:  "mean_log_prob", "log_mean_prob"
        if not df.columns.str.contains(f"{field}_at_").any():
            print(f"Missing {field}_at_ for {attack} {dataset}")
            continue
        elif dataset in ["pm", "imdb", "spam"] and attack in ["gcg", "beast"]:
            df[field] = df[f"{field}_at_10"]
        elif dataset in ["helpful", "harmless", "wl"] and attack in ["gcg", "beast"]:
            df[field] = df[f"{field}_at_2"]
        elif (
            dataset in ["pm", "imdb", "spam", "helpful", "harmless", "wl"]
            and attack == "rt"
        ):
            df[field] = df[f"{field}_at_1200"]
        if field == "log_mean_prob":
            df["mean_prob"] = np.exp(df["log_mean_prob"])


def _plot_strongreject(style: str) -> None:
    mode = "gen"
    attack = "gcg"
    dataset = "strongreject"
    family = "qwen"
    data, metadata = read_csv_and_metadata("finetuned", family, attack, dataset)
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
                    f"{name_to_model(family)}, "
                    f"{name_to_dataset(dataset)}, "
                    f"{name_to_attack(attack)} Attack"
                ),
                mode=mode,
                family=family,
                attack=attack,
                dataset=dataset,
                successes_name=success_name,
                trials_name=trials_name,
                legend=legend,
                y_data_name=y_data_name,
                ytransform=ytransform,
                style=style,
                smoothing=0 if "prob" in y_data_name else 1,
                ylim=(-0.025, 1.025) if ytransform is None else None,
            )

    # Pre attack accuracy
    success_name = "adversarial_eval_n_correct_pre_attack"
    trials_name = "adversarial_eval_n_examples"
    draw_wilson_score_interval_plot(
        data,
        metadata=metadata,
        title=f"{name_to_model(family)} Pre-Attack Accuracy",
        mode=mode,
        family=family,
        attack=attack,
        dataset=dataset,
        successes_name=success_name,
        trials_name=trials_name,
        y_data_name="adversarial_eval_pre_attack_accuracy",
        ytransform="none",
        legend=True,
        style=style,
        smoothing=1,
        ylim=(-0.025, 1.025),
    )

    # Post attack accuracy
    success_name = "adversarial_eval_n_correct_post_attack"
    trials_name = "adversarial_eval_n_examples"
    for attack, attack_df in data.groupby("attack"):
        assert isinstance(attack, str)
        title = f"{name_to_model(family)} Post-Attack Accuracy ({str(attack).upper()})"
        draw_wilson_score_interval_plot(
            attack_df,
            metadata=metadata,
            title=title,
            successes_name=success_name,
            trials_name=trials_name,
            y_data_name="post_attack_accuracy",
            ytransform="none",
            mode=mode,
            family=family,
            attack=attack,
            dataset=dataset,
            legend=True,
            style=style,
            smoothing=1,
            ylim=(-0.025, 1.025),
        )


def _plot_pythia_clf(style: str) -> None:
    all_data = []
    metadata = None
    for family, attack, dataset in [
        ("pythia", "gcg", "imdb"),
        ("pythia", "gcg", "spam"),
        ("pythia", "gcg", "wl"),
        ("pythia", "gcg", "pm"),
        ("pythia", "gcg", "helpful"),
        ("pythia", "gcg", "harmless"),
        ("pythia", "rt", "imdb"),
        ("pythia", "rt", "spam"),
        ("pythia", "rt", "wl"),
        ("pythia", "rt", "pm"),
        ("pythia", "rt", "helpful"),
        ("pythia", "rt", "harmless"),
        ("qwen", "gcg", "spam"),
        ("qwen", "gcg", "harmless"),
        ("qwen", "beast", "harmless"),
        ("qwen", "beast", "spam"),
        ("pythia", "beast", "harmless"),
    ]:
        save_as = ("finetuned", family, attack, dataset)
        data, metadata = read_csv_and_metadata(*save_as)
        pick_attack_iterations(df=data, attack=attack, dataset=dataset, family=family)
        for legend in (True, False):
            for y_data_name, ytransform in [
                # ("asr", "logit"),
                ("asr", None),
                # ("mean_log_prob", "negative"),
                # ("log_mean_prob", "comp_exp"),
            ]:
                if y_data_name not in data.columns:
                    continue
                draw_min_max_median_plot(
                    data,
                    metadata=metadata,
                    title=(
                        f"{name_to_model(family)}, "
                        f"{name_to_dataset(dataset)}, "
                        f"{name_to_attack(attack)} Attack"
                    ),
                    legend=legend,
                    y_data_name=y_data_name,
                    ytransform=ytransform,
                    family=family,
                    attack=attack,
                    dataset=dataset,
                    style=style,
                    smoothing=0 if "prob" in y_data_name else 1,
                    ylim=(-0.025, 1.025) if ytransform is None else None,
                )
            data["family"] = family
            data["attack"] = attack
            data["dataset"] = dataset
            all_data.append(data)
    concat_data = pd.concat(all_data)
    # concat_data["asr"] = concat_data["adversarial_eval_attack_success_rate"]
    assert not concat_data.asr.isnull().any()
    for (family, attack), attack_df in concat_data.groupby(["family", "attack"]):  # type: ignore # noqa
        assert isinstance(family, str)
        assert isinstance(attack, str)
        for legend in (True, False):
            for y_data_name, ytransform in [
                ("asr", "logit"),
                ("asr", None),
                # ("mean_log_prob", "negative"),
                # ("log_mean_prob", "comp_exp"),
            ]:
                draw_min_max_median_plot_by_dataset(
                    attack_df,
                    metadata=metadata,
                    title=(f"{name_to_model(family)}, {name_to_attack(attack)}"),
                    y_data_name=y_data_name,
                    ytransform=ytransform,
                    adversarial=False,
                    family=family,
                    attack=attack,
                    legend=legend,
                    style=style,
                    smoothing=0 if "prob" in y_data_name else 1,
                    ylim=(-0.025, 1.025) if ytransform is None else None,
                )
                draw_min_max_median_plot_by_dataset(
                    attack_df.loc[attack_df.dataset.ne("wl")],
                    metadata=metadata,
                    title=f"{name_to_model(family)}, {name_to_attack(attack)}",
                    y_data_name=y_data_name,
                    ytransform=ytransform,
                    adversarial=False,
                    family=family,
                    attack=attack,
                    datasets="all_except_wl",
                    legend=legend,
                    style=style,
                    smoothing=0 if "prob" in y_data_name else 1,
                    ylim=(-0.025, 1.025) if ytransform is None else None,
                )

    for family in ["pythia"]:
        attack = "gcg"
        # Pre attack accuracy
        gcg_dataset = concat_data.loc[concat_data.attack.eq(attack)]
        draw_min_max_median_plot_by_dataset(
            gcg_dataset,
            metadata=metadata,
            title="Pre-Attack Accuracy",
            y_data_name="adversarial_eval_pre_attack_accuracy",
            ytransform="none",
            adversarial=False,
            family=family,
            attack=attack,
            datasets="all",
            legend=True,
            style=style,
            smoothing=1,
            ylim=(-0.025, 1.025),
        )

        # Post attack accuracy
        for attack, attack_df in concat_data.groupby("attack"):
            assert isinstance(attack, str)
            draw_min_max_median_plot_by_dataset(
                attack_df,
                metadata=metadata,
                title=f"Post-Attack Accuracy ({str(attack).upper()})",
                y_data_name="post_attack_accuracy",
                ytransform="none",
                adversarial=False,
                family=family,
                attack=attack,
                datasets="all",
                legend=True,
                legend_loc="upper left",
                style=style,
                smoothing=1,
                ylim=(-0.025, 1.025),
            )


def _plot_qwen_gen_and_clf(style: str) -> None:
    all_data = []
    metadata = None
    for family, attack, dataset in [
        ("qwen", "gcg", "spam"),
        ("qwen", "gcg", "harmless"),
        ("qwen", "gcg", "strongreject"),
    ]:
        save_as = ("finetuned", family, attack, dataset)
        data, metadata = read_csv_and_metadata(*save_as)
        pick_attack_iterations(data, attack, dataset, family)
        data["family"] = family
        data["attack"] = attack
        data["dataset"] = dataset
        all_data.append(data)
    concat_data = pd.concat(all_data)
    assert not concat_data.asr.isnull().any()
    for (family, attack), attack_df in concat_data.groupby(["family", "attack"]):  # type: ignore # noqa
        assert isinstance(family, str)
        assert isinstance(attack, str)
        for legend in (True, False):
            for y_data_name, ytransform in [
                ("asr", None),
            ]:
                draw_min_max_median_and_wilson_plot_by_dataset(
                    attack_df,
                    metadata=metadata,
                    title=f"{name_to_model(family)}, {name_to_attack(attack)}",
                    y_data_name=y_data_name,
                    successes_name="n_incorrect_post_attack",
                    trials_name="n_correct_pre_attack",
                    ytransform=ytransform,
                    adversarial=False,
                    family=family,
                    attack=attack,
                    legend=legend,
                    style=style,
                    smoothing=0 if "prob" in y_data_name else 1,
                    ylim=(-0.025, 1.025),
                )

        # Delete the strongreject dataset before plotting pre-attack accuracy
        # Keep the classification tasks
        attack_df = attack_df.loc[attack_df.dataset.ne("strongreject")]
        draw_min_max_median_plot_by_dataset(
            attack_df,
            metadata=metadata,
            title=f"{name_to_model(family)} Pre-Attack Accuracy",
            y_data_name="adversarial_eval_pre_attack_accuracy",
            ytransform="none",
            adversarial=False,
            family=family,
            attack=attack,
            datasets="all",
            legend=True,
            style=style,
            smoothing=1,
            ylim=(-0.025, 1.025),
        )


def main(style):
    set_style(style)
    _plot_strongreject(style)
    _plot_pythia_clf(style)
    _plot_qwen_gen_and_clf(style)


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
