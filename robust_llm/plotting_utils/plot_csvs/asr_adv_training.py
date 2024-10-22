"""Plot attack scaling using Tom's transfer evals."""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_asr

# Set the plot style
set_plot_style("paper")

ROUNDS = [0, 1e-4, 1e-3, 5e-3, -1]


def main():
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
            )


if __name__ == "__main__":
    main()
