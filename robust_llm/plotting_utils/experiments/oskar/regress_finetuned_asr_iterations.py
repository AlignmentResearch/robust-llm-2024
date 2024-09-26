# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.tools import (
    create_legend,
    get_color_palette,
    get_legend_handles,
    set_up_paper_plot,
)
from robust_llm.wandb_utils.constants import MODEL_NAMES, MODEL_SIZES

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
def get_data_for_group(group_name: str):
    root = compute_repo_path()
    path = os.path.join(root, "outputs", f"asr_{group_name}.csv")
    if not os.path.exists(path):
        print(f"Group {group_name} does not exist")
        return
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
    df.sort_values("model_idx", inplace=True)
    return df


# %%
data = get_data_for_group(GROUPS[0])
assert data is not None
# %%
df = data.loc[data.asr.between(0, 1, inclusive="neither")].copy()
# df = data.loc[data.asr.between(0.01, 0.5)].copy()
df["model"] = df.model_idx.apply(lambda x: MODEL_NAMES[x])
df["log_asr"] = np.log(df.asr)
df["logit_asr"] = np.log(df.asr / (1 - df.asr))
df["log_iterations_x_params"] = np.log(df.iteration_x_params)
df["log_params"] = np.log(df.num_params)
df["gompit_asr"] = -np.log(-np.log(df.asr))
df["sqrt_neg_log_complement_asr"] = np.sqrt(-np.log(1 - df["asr"]))
# %%
Y_VAR = "logit_asr"
# print(df.loc[~np.isfinite(df[Y_VAR])].head())
assert not df[Y_VAR].isnull().any()
assert np.isfinite(df[Y_VAR]).all()
# %%
fig, ax = plt.subplots(dpi=200)
set_up_paper_plot(fig, ax)
color_data_name = "num_params"
palette = get_color_palette(df, color_data_name)
sns.lineplot(
    data=df,
    x="iteration_x_params",
    y=Y_VAR,
    hue=color_data_name,
    ax=ax,
    palette=palette,
    legend=False,
)
ax.set_xlabel("Attack compute (Iterations x Parameters)")
ax.set_ylabel(Y_VAR)
ax.set_xscale("log")
fig.suptitle("GCG/IMDB".upper())
legend_handles = get_legend_handles(df, color_data_name, palette)
create_legend(color_data_name, ax, legend_handles, outside=False)
fig.show()
# %%
model = smf.ols(f"{Y_VAR} ~ log_iterations_x_params + log_params", data=df)
results = model.fit()
print(results.summary())
# %%
# %%
