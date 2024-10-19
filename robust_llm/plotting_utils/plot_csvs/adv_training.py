"""Adversarial training plots (robustness over time)."""

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_adv_training_plots

set_plot_style("paper")


def main():
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
                    )


if __name__ == "__main__":
    main()
