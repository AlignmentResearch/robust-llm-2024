"""Adversarial training transfer plots."""

import argparse

from robust_llm.plotting_utils.style import name_to_attack, name_to_dataset, set_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

WHEN_TO_PLOT_CLEAN_ACCURACY = ["rt_gcg"]


def main(style, skip_clean=False):
    set_style(style)

    for x_data_name in ("defense_flops_fraction_pretrain",):
        for legend in (True, False):
            for iteration in (12, 128):
                for attack, dataset in [
                    ("rt_gcg", "imdb"),
                    ("rt_gcg", "spam"),
                    ("rt_gcg", "wl"),
                    ("rt_gcg", "pm"),
                    ("gcg_gcg_infix90", "imdb"),
                    ("gcg_gcg_infix90", "spam"),
                    ("gcg_gcg_prefix", "imdb"),
                    ("gcg_gcg_prefix", "spam"),
                    ("gcg_no_ramp_gcg", "imdb"),
                ]:
                    load_and_plot_adv_training_plots(
                        attack=attack,
                        dataset=dataset,
                        x_data_name=x_data_name,
                        color_data_name="num_params",
                        legend=legend,
                        y_data_name=f"metrics_asr_at_{iteration}",
                        style=style,
                    )

                    # Plot the clean performance too,
                    # as well all the post-attack accuracy
                    if not skip_clean and attack in WHEN_TO_PLOT_CLEAN_ACCURACY:
                        d_name = name_to_dataset(dataset)
                        a_name = name_to_attack(attack)
                        for y_transform in ["none", "logit"]:
                            load_and_plot_adv_training_plots(
                                attack=attack,
                                dataset=dataset,
                                x_data_name=x_data_name,
                                color_data_name="num_params",
                                legend=legend,
                                y_data_name="adversarial_eval_pre_attack_accuracy",
                                style=style,
                                y_transform=y_transform,
                                title=f"Pre-Attack Accuracy ({d_name}, {a_name})",
                            )
                            load_and_plot_adv_training_plots(
                                attack=attack,
                                dataset=dataset,
                                x_data_name=x_data_name,
                                color_data_name="num_params",
                                legend=legend,
                                y_data_name="post_attack_accuracy",
                                style=style,
                                y_transform=y_transform,
                                title=f"Post-Attack Accuracy ({d_name}, {a_name})",
                            )
                            if y_transform == "none":
                                load_and_plot_adv_training_plots(
                                    attack=attack,
                                    dataset=dataset,
                                    x_data_name=x_data_name,
                                    color_data_name="num_params",
                                    legend=legend,
                                    y_data_name="adversarial_eval_pre_attack_accuracy",
                                    style=style,
                                    y_transform=y_transform,
                                    ylim=(0, 1),
                                    title=f"Pre-Attack Accuracy ({d_name}, {a_name})",
                                )
                                load_and_plot_adv_training_plots(
                                    attack=attack,
                                    dataset=dataset,
                                    x_data_name=x_data_name,
                                    color_data_name="num_params",
                                    legend=legend,
                                    y_data_name="post_attack_accuracy",
                                    style=style,
                                    y_transform=y_transform,
                                    ylim=(0, 1),
                                    title=f"Post-Attack Accuracy ({d_name}, {a_name})",
                                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial training transfer plots.")
    parser.add_argument("--style", type=str, default="paper", help="Plot style to use")
    args = parser.parse_args()
    main(args.style)
