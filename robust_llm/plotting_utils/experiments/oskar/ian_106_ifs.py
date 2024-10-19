# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.constants import AXIS_LABELS
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
def plot_ifs_for_group(group_name: str):
    root = compute_repo_path()
    path = os.path.join(root, "outputs", f"ifs_{group_name}.csv")
    if not os.path.exists(path):
        return
    attack = group_name.split("_")[2]
    dataset = group_name.split("_")[-1]
    df = pd.read_csv(path)
    assert df.columns.tolist() == ["model_idx", "seed_idx", "ifs", "decile"]
    assert df.model_idx.between(0, 9).all()
    assert df.seed_idx.between(0, 4).all()
    assert df.decile.between(0, 10).all()
    assert len(df) == 11 * 5 * 10  # 11 deciles, 5 seeds, 10 models
    df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])
    df["asr"] = df.decile.mul(10)
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)
    color_data_name = "num_params"
    palette = get_color_palette(df, color_data_name)
    sns.lineplot(
        data=df,
        x="asr",
        y="ifs",
        hue=color_data_name,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_ylabel("Iterations required to reach ASR")
    ax.set_xlabel(AXIS_LABELS["asr"])
    fig.suptitle(f"{attack}/{dataset}".upper())
    create_path_and_savefig(fig, "ifs", attack, dataset, "no_legend")
    legend_handles = get_legend_handles(df, color_data_name, palette)
    create_legend(color_data_name, ax, legend_handles, outside=False)
    save_path = create_path_and_savefig(fig, "ifs", attack, dataset, "legend")
    df.to_csv(str(save_path).replace("legend.pdf", "data.csv"), index=False)


# %%
for group in GROUPS:
    plot_ifs_for_group(group)

# %%
