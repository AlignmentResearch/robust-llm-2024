# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.tools import (
    create_legend,
    create_path_and_savefig,
    get_color_palette,
    get_legend_handles,
    set_up_paper_plot,
)
from robust_llm.wandb_utils.constants import MODEL_SIZES

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
def plot_asr_for_group(group_name: str):
    root = compute_repo_path()
    path = os.path.join(root, "outputs", f"asr_{group_name}.csv")
    if not os.path.exists(path):
        print(f"Group {group_name} does not exist")
        return
    attack = group_name.split("_")[2]
    dataset = group_name.split("_")[-1]
    df = pd.read_csv(path)
    assert df.columns.tolist() == ["model_idx", "seed_idx", "asr", "iteration"]
    n_models = 10
    n_seeds = 5
    n_iterations = 11 if "gcg" in group_name else 1281
    assert df.model_idx.between(0, n_models - 1).all()
    assert df.seed_idx.between(0, n_seeds - 1).all()
    assert df.iteration.between(0, n_iterations - 1).all()
    assert len(df) == n_models * n_seeds * n_iterations
    if n_iterations > 1000:
        df = df.loc[df.iteration.mod(100) == 0]
    df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])
    df["iteration_x_params"] = df.iteration * df.num_params
    df["sigmoid_asr"] = 1 / (1 + np.exp(-df.asr))
    df["logit_asr"] = np.log(df.asr / (1 - df.asr))
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)
    color_data_name = "num_params"
    palette = get_color_palette(df, color_data_name)
    sns.lineplot(
        data=df,
        x="iteration_x_params",
        y="logit_asr",
        hue=color_data_name,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_xlabel("Attack compute (Iterations $\times$ Parameters)")
    ax.set_ylabel("Attack success rate (logit)")
    ax.set_xscale("log")
    fig.suptitle(f"{attack}/{dataset}".upper())
    create_path_and_savefig(fig, "asr", attack, dataset, "no_legend")
    legend_handles = get_legend_handles(df, color_data_name, palette)
    create_legend(color_data_name, ax, legend_handles, outside=False)
    save_path = create_path_and_savefig(fig, "asr", attack, dataset, "legend")
    df.to_csv(str(save_path).replace("legend.pdf", "data.csv"), index=False)


# %%
for group in tqdm(GROUPS):
    plot_asr_for_group(group)

# %%
