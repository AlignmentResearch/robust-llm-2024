"""Adversarial training transfer plots."""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

set_plot_style("paper")


def main():
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
                ]:
                    load_and_plot_adv_training_plots(
                        attack=attack,
                        dataset=dataset,
                        x_data_name=x_data_name,
                        color_data_name="num_params",
                        legend=legend,
                        y_data_name=f"metrics_asr_at_{iteration}",
                    )


if __name__ == "__main__":
    main()
