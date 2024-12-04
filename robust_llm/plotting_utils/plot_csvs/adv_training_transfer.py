"""Adversarial training transfer plots."""

import argparse

from robust_llm.plotting_utils.style import (
    name_to_attack,
    name_to_dataset,
    name_to_model,
    set_style,
)
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

WHEN_TO_PLOT_CLEAN_ACCURACY = ["rt_gcg"]


def main(style, skip_clean=False):
    set_style(style)

    for x_data_name in ("defense_flops_fraction_pretrain",):
        for legend in (True, False):
            for family, attack, dataset in [
                ("pythia", "rt_gcg", "imdb"),
                ("pythia", "rt_gcg", "spam"),
                # ("pythia", "rt_gcg", "wl"),
                # ("pythia", "rt_gcg", "pm"),
                ("pythia", "gcg_gcg_infix90", "imdb"),
                ("pythia", "gcg_gcg_infix90", "spam"),
                # ("pythia", "gcg_gcg_prefix", "imdb"),
                # ("pythia", "gcg_gcg_prefix", "spam"),
                ("pythia", "gcg_no_ramp_gcg", "imdb"),
                ("qwen", "gcg_beast", "harmless"),
            ]:
                iterations = [25] if "beast" in attack else [128]
                for iteration in iterations:
                    load_and_plot_adv_training_plots(
                        family=family,
                        attack=attack,
                        dataset=dataset,
                        x_data_name=x_data_name,
                        color_data_name="num_params",
                        legend=legend,
                        y_data_name=f"asr_at_{iteration}",
                        style=style,
                    )

                    # Plot the clean performance too,
                    # as well all the post-attack accuracy
                    if not skip_clean and attack in WHEN_TO_PLOT_CLEAN_ACCURACY:
                        m_name = name_to_model(family)
                        d_name = name_to_dataset(dataset)
                        a_name = name_to_attack(attack)
                        suffix = f"({m_name}, {d_name}, {a_name})"
                        for y_transform in ["none", "logit"]:
                            load_and_plot_adv_training_plots(
                                family=family,
                                attack=attack,
                                dataset=dataset,
                                x_data_name=x_data_name,
                                color_data_name="num_params",
                                legend=legend,
                                y_data_name="adversarial_eval_pre_attack_accuracy",
                                style=style,
                                y_transform=y_transform,
                                title=f"Pre-Attack Accuracy {suffix}",
                                ylim=(0, 1) if y_transform == "none" else None,
                            )
                            load_and_plot_adv_training_plots(
                                family=family,
                                attack=attack,
                                dataset=dataset,
                                x_data_name=x_data_name,
                                color_data_name="num_params",
                                legend=legend,
                                y_data_name="post_attack_accuracy",
                                style=style,
                                y_transform=y_transform,
                                title=f"Post-Attack Accuracy {suffix}",
                                ylim=(0, 1) if y_transform == "none" else None,
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial training transfer plots.")
    parser.add_argument("--style", type=str, default="paper", help="Plot style to use")
    args = parser.parse_args()
    main(args.style)
