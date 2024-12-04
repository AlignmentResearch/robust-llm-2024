"""Plot attack scaling using Ian's finetuned evals"""

import argparse

import numpy as np
import pandas as pd

from robust_llm.plotting_utils.style import set_style
from robust_llm.plotting_utils.tools import (
    DEFAULT_SMOOTHING,
    PlotMetadata,
    plot_attack_scaling_base,
    postprocess_attack_compute,
    read_csv_and_metadata,
)


def plot_asr_for_group(
    df: pd.DataFrame,
    metadata: PlotMetadata | None,
    family: str,
    attack: str,
    dataset: str,
    x: str = "iteration_x_flops",
    y_data_name: str = "asr",
    y_transform: str = "logit",
    smoothing: int = DEFAULT_SMOOTHING,
    style: str = "paper",
    datapoints: int = 130,
):
    postprocess_attack_compute(df, family, attack, dataset)
    if df.iteration.nunique() > datapoints:
        # If datapoints=10, filter out all but empirical deciles
        df = df.loc[
            df.iteration.isin(df.iteration.quantile(np.linspace(0, 1, datapoints)))
        ]
    plot_attack_scaling_base(
        df,
        metadata=metadata,
        family=family,
        attack=attack,
        dataset=dataset,
        round_info="finetuned",
        smoothing=smoothing,
        x_data_name=x,
        y_data_name=y_data_name,
        y_transform=y_transform,
        style=style,
    )


def main(style: str):
    set_style(style)
    family = "pythia"
    for family, attack, dataset in [
        ("pythia", "gcg", "imdb"),
        ("pythia", "gcg", "pm"),
        ("pythia", "gcg", "wl"),
        ("pythia", "gcg", "spam"),
        ("pythia", "gcg", "helpful"),
        ("pythia", "gcg", "harmless"),
        ("pythia", "rt", "imdb"),
        ("pythia", "rt", "pm"),
        ("pythia", "rt", "wl"),
        ("pythia", "rt", "spam"),
        ("pythia", "rt", "helpful"),
        ("pythia", "rt", "harmless"),
        ("qwen", "gcg", "spam"),
        ("qwen", "gcg", "harmless"),
        ("qwen", "beast", "spam"),
        ("qwen", "beast", "harmless"),
        ("pythia", "beast", "harmless"),
    ]:
        df, metadata = read_csv_and_metadata(
            "asr", family, attack, dataset, "finetuned"
        )
        for x in ("attack_flops_fraction_pretrain",):
            for y_transf, y_data in [
                ("logit", "asr"),
                # ("comp_exp", "log_mean_prob"),
                # ("negative", "mean_log_prob"),
            ]:
                plot_asr_for_group(
                    df,
                    metadata=metadata,
                    family=family,
                    attack=attack,
                    dataset=dataset,
                    x=x,
                    y_data_name=y_data,
                    y_transform=y_transf,
                    style=style,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot attack scaling using Ian's finetuned evals"
    )
    parser.add_argument("--style", type=str, default="paper", help="Plot style to use")
    args = parser.parse_args()
    main(args.style)
