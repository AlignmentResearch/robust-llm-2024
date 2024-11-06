"""Adversarial training plots (robustness over time)."""

import argparse

from robust_llm.plotting_utils.style import name_to_attack, name_to_dataset, set_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots


def main(style):
    set_style(style)
    for x_data_name in ("defense_flops_fraction_pretrain",):
        for legend in (True, False):
            for iteration in (12, 128):
                for attack, dataset in [
                    ("gcg_gcg", "imdb"),
                    ("gcg_gcg", "spam"),
                    ("gcg_gcg", "wl"),
                    ("gcg_gcg", "pm"),
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
                    if iteration == 128:
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
    parser = argparse.ArgumentParser(description="Adversarial training plots")
    parser.add_argument(
        "--style",
        type=str,
        default="paper",
        help="Style to be used for plotting",
    )
    args = parser.parse_args()
    main(args.style)
