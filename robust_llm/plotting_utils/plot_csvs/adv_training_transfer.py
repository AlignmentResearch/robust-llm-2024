"""Adversarial training transfer plots."""

import argparse

from robust_llm.plotting_utils.style import set_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots


def main(style):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial training transfer plots.")
    parser.add_argument("--style", type=str, default="paper", help="Plot style to use")
    args = parser.parse_args()
    main(args.style)
