"""Adversarial training plots (robustness over time)."""

import argparse

from robust_llm.plotting_utils.style import (
    name_to_attack,
    name_to_dataset,
    name_to_model,
    set_style,
)
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots


def main(style, skip_clean=False):
    set_style(style)
    for x_data_name in ("defense_flops_fraction_pretrain", "train_total_flops"):
        for legend in (True, False):
            for iteration in (128,):
                for family, attack, dataset in [
                    ("pythia", "gcg_gcg_match_seed", "imdb"),
                    ("pythia", "gcg_gcg", "imdb"),
                    ("pythia", "gcg_gcg", "spam"),
                    ("pythia", "gcg_gcg", "wl"),
                    ("pythia", "gcg_gcg", "pm"),
                    ("qwen", "gcg_gcg", "harmless"),
                    ("qwen", "gcg_gcg", "spam"),
                ]:
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
                    if not skip_clean:
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
    parser = argparse.ArgumentParser(description="Adversarial training plots")
    parser.add_argument(
        "--style",
        type=str,
        default="paper",
        help="Style to be used for plotting",
    )
    parser.add_argument(
        "--skip_clean",
        action="store_true",
        help="Whether to include clean performance in the plots",
    )
    args = parser.parse_args()
    main(args.style, args.skip_clean)
