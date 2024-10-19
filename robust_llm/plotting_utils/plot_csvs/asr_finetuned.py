"""Plot attack scaling using Ian's finetuned evals"""

import pandas as pd

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import (
    DEFAULT_SMOOTHING,
    PlotMetadata,
    plot_attack_scaling_base,
    postprocess_attack_compute,
    read_csv_and_metadata,
)

set_plot_style("paper")


def plot_asr_for_group(
    df: pd.DataFrame,
    metadata: PlotMetadata | None,
    attack: str,
    dataset: str,
    x: str = "iteration_x_flops",
    y: str = "logit_asr",
    smoothing: int = DEFAULT_SMOOTHING,
):
    postprocess_attack_compute(df, attack, dataset)
    plot_attack_scaling_base(
        df,
        metadata=metadata,
        attack=attack,
        dataset=dataset,
        round_info="finetuned",
        smoothing=smoothing,
        x=x,
        y=y,
    )


def main():
    for attack in ("gcg", "rt"):
        for dataset in ("imdb", "pm", "wl", "spam"):
            df, metadata = read_csv_and_metadata("asr", attack, dataset, "finetuned")
            for x in ("attack_flops_fraction_pretrain",):
                for y in ("logit_asr",):
                    plot_asr_for_group(
                        df, metadata=metadata, attack=attack, dataset=dataset, x=x, y=y
                    )


if __name__ == "_main__":
    main()
