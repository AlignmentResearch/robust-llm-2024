# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from robust_llm.plotting_utils.constants import FUDGE_FOR_12B
from robust_llm.plotting_utils.experiments.pretrain_compute_per_model import (
    ESTIMATED_PRETRAIN_COMPUTE,
)
from robust_llm.plotting_utils.tools import (
    create_legend,
    create_path_and_savefig,
    get_cached_asr_data,
    get_color_palette,
    get_legend_handles,
    prepare_adv_training_data,
    set_up_paper_plot,
)
from robust_llm.plotting_utils.utils import add_model_idx_inplace

GROUPS = [
    "ian_106_gcg_pythia_imdb",
    "ian_107_gcg_pythia_pm",
    "ian_108_gcg_pythia_wl",
    "ian_109_gcg_pythia_spam",
    "ian_110_rt_pythia_imdb",
    "ian_111_rt_pythia_pm",
    "ian_112_rt_pythia_wl",
    "ian_113_rt_pythia_spam",
]


# %%


# %%
def plot_asr_for_group(
    df: pd.DataFrame,
    group_name: str,
    x: str = "iteration_x_flops",
    y: str = "logit_asr",
    log_x: bool = True,
):

    attack = group_name.split("_")[2]
    dataset = group_name.split("_")[-1]

    if y == "sigmoid_asr":
        df["sigmoid_asr"] = 1 / (1 + np.exp(-df.asr))
    elif y == "logit_asr":
        df["logit_asr"] = np.log(df.asr / (1 - df.asr))

    if "flop" in x:
        flop_data = prepare_adv_training_data(
            (group,),
            summary_keys=[
                "experiment_yaml.model.name_or_path",
                "experiment_yaml.dataset.n_val",
                "model_size",
                "experiment_yaml.model.revision",
            ],
            metrics=[
                "flops_per_iteration",
            ],
        )
        flop_data["seed_idx"] = (
            flop_data.model_name_or_path.str.split("_s-").str[-1].astype(int)
        )
        flop_data = add_model_idx_inplace(flop_data, reference_col="model_size")
        # Fudge the compute for the 12b model due to issue recording multi-GPU
        # flops.
        flop_data.loc[flop_data.model_idx == 9, "flops_per_iteration"] *= FUDGE_FOR_12B
        flop_data["pretrain_compute"] = flop_data.model_idx.map(
            ESTIMATED_PRETRAIN_COMPUTE
        )

        df = df.merge(
            flop_data,
            on=["model_idx", "seed_idx"],
            how="left",
            validate="m:1",
            suffixes=("", "_flop"),
        )
        assert df.flops_per_iteration.notnull().all()
        df.flops_per_iteration = df.groupby(
            ["model_idx", "iteration"]
        ).flops_per_iteration.transform("mean")
        df["iteration_x_flops"] = df.iteration * df.flops_per_iteration
        df["flops_percent_pretrain"] = 100 * df.iteration_x_flops / df.pretrain_compute

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)
    color_data_name = "num_params"
    palette = get_color_palette(df, color_data_name)
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=color_data_name,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_xlabel(
        {
            "iteration_x_params": "Attack compute (Iterations $\times$ Parameters)",
            "iteration_x_flops": "Attack compute (FLOPs)",
            "iteration": "Attack iteration",
            "flops_percent_pretrain": "Attack compute (% of pretrain)",
        }[x]
    )
    ax.set_ylabel(
        {
            "logit_asr": "logit(attack success)",
            "asr": "attack success rate",
        }[y]
    )
    if log_x:
        ax.set_xscale("log")
    fig.suptitle(f"{attack}/{dataset}".upper())
    create_path_and_savefig(fig, "asr", attack, dataset, x, y, "no_legend")
    legend_handles = get_legend_handles(df, color_data_name, palette)
    create_legend(color_data_name, ax, legend_handles, outside=False)
    save_path = create_path_and_savefig(fig, "asr", attack, dataset, x, y, "legend")
    df.to_csv(str(save_path).replace("legend.pdf", "data.csv"), index=False)


# %%
for group in tqdm(GROUPS):
    df = get_cached_asr_data(group)
    if df.empty:
        print(f"Group {group} does not exist")
        continue
    for x in (
        "iteration",
        "iteration_x_params",
        "iteration_x_flops",
        "flops_percent_pretrain",
    ):
        for y in ("asr", "logit_asr"):
            plot_asr_for_group(df, group, x, y)

# %%
