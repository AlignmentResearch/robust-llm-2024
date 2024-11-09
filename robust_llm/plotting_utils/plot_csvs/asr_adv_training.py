"""Plot attack scaling using Tom's transfer evals."""

import argparse

from robust_llm.plotting_utils.style import set_style
from robust_llm.plotting_utils.tools import load_and_plot_asr

ROUNDS = [1e-4, 1e-3, 5e-3, -1]


def main(style):
    # Set the plot style
    set_style(style)

    for attack, dataset in [
        ("rt_gcg", "imdb"),
        ("rt_gcg", "spam"),
        ("gcg_gcg", "imdb"),
        ("gcg_gcg_infix90", "imdb"),
        ("gcg_gcg", "spam"),
        ("gcg_gcg_infix90", "spam"),
        ("gcg_gcg", "wl"),
        ("gcg_gcg_prefix", "imdb"),
        ("gcg_gcg_prefix", "spam"),
        ("gcg_no_ramp_gcg", "imdb"),
    ]:
        for x in ("attack_flops_fraction_pretrain",):
            load_and_plot_asr(
                attack=attack,
                dataset=dataset,
                rounds=ROUNDS,
                x=x,
                style=style,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot attack scaling using Tom's transfer evals."
    )
    parser.add_argument("--style", type=str, default="paper", help="Plot style to use")
    args = parser.parse_args()
    main(args.style)
