"""Plot slopes of slopes for attack scaling"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.ticker import ScalarFormatter

from robust_llm.plotting_utils.constants import AXIS_LABELS
from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import (
    create_path_and_savefig,
    postprocess_attack_compute,
    read_csv_and_metadata,
    set_up_paper_plot,
)

set_plot_style("paper")


def regress_attack_scaling(attack: str, dataset: str, round: str):
    df, metadata = read_csv_and_metadata("asr", attack, dataset, round)
    postprocess_attack_compute(df, attack, dataset)
    df = df.loc[np.isfinite(df.logit_asr)]
    gradients = dict()
    reg = None
    for num_params, params_df in df.groupby("num_params"):
        reg = smf.ols(
            "logit_asr ~ np.log10(attack_flops_fraction_pretrain)", data=params_df
        ).fit()
        gradients[num_params] = reg.params.iloc[-1]
    grad_df = pd.DataFrame(
        gradients.items(), columns=["num_params", "gradient"]  # type: ignore
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    set_up_paper_plot(fig, ax)

    # Perform linear regression
    model = smf.ols("gradient ~ np.log10(num_params)", data=grad_df).fit()

    # Extract statistics
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r_squared = model.rsquared

    description = (
        r"$\mathrm{logit}_{10}$(Attack Success Rate) vs."
        "\n"
        r"$\log_{10}$(Attack Compute)"
    )
    fig.suptitle(
        f"{attack}/{dataset}".upper() + f" Regression slopes of {description}, "
        "split by model size ",
        fontsize=8,
    )

    # Plot the data and regression line
    sns.scatterplot(
        x="num_params",
        y="gradient",
        data=grad_df,
        ax=ax,
        color="black",
    )
    # draw a red dashed regression line
    x = np.linspace(grad_df.num_params.min(), grad_df.num_params.max(), 100)
    y = slope * np.log10(x) + intercept
    ax.plot(x, y, color="red", linestyle="--", zorder=-1, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel(AXIS_LABELS["num_params"])
    ax.set_ylabel(f"Slope of {description}")
    # Set y-axis to scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # Add equation and R² inside the plot
    ax.text(
        0.05,
        0.95,
        f"$R^2 = {r_squared:.2f}$",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    # plt.tight_layout()

    create_path_and_savefig(
        fig,
        "asr",
        attack,
        dataset,
        round,
        "attack_flops_fraction_pretrain",
        "logit_asr",
        "smoothing-1",
        "regplot",
        data=df,
        metadata=metadata,
    )


def main():
    for attack in ["gcg", "gcg_gcg", "rt"]:
        for dataset in ["imdb", "spam", "pm", "wl", "helpful", "harmless"]:
            if dataset in ["helpful", "harmless"] and attack == "gcg_gcg":
                continue
            rounds = (
                [
                    "round_0",
                    "pretrain_fraction_10_bps",
                    "pretrain_fraction_50_bps",
                    "final_round",
                ]
                if "_" in attack
                else [
                    "finetuned",
                ]
            )
            for round in rounds:
                regress_attack_scaling(attack, dataset, round)


if __name__ == "__main__":
    main()
