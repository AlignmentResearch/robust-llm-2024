# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.interpolate import griddata

from robust_llm.plotting_utils.constants import AXIS_LABELS, MODEL_PLOTTING_NAMES
from robust_llm.plotting_utils.experiments.pretrain_compute_per_model import (
    ESTIMATED_PRETRAIN_COMPUTE,
)
from robust_llm.plotting_utils.tools import (
    TRANSFORMS,
    load_flops_data,
    prepare_adv_training_data,
)

# set max columns to 100
pd.set_option("display.max_columns", 100)
# %%
summary_keys = [
    "experiment_yaml.dataset.n_val",
    "model_size",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
]
METRICS = [
    "metrics/asr@12",
    "metrics/asr@60",
    "metrics/asr@120",
    "metrics/asr@128",
]
iteration = 12
run_names = ("tom_007_eval_niki_152_gcg", "tom_007_eval_niki_152a_gcg")
merge_runs = (
    "niki_152a_adv_tr_gcg_imdb_small",
    "niki_152_adv_tr_gcg_imdb_small",
)
title = "IMDB, GCG"
save_as = ("imdb", f"gcg_vs_gcg{iteration}")
summary_keys = summary_keys
color_data_name = "num_params"
y_data_name = f"metrics_asr_at_{iteration}"
metrics = METRICS
data = prepare_adv_training_data(
    group_names=run_names,
    summary_keys=summary_keys,
    metrics=metrics,
)
if merge_runs is not None:
    data["model_key"] = data.model_name_or_path.str.replace(
        "AlignmentResearch/", ""
    ).str.replace("robust_llm_", "")
    train_data = load_flops_data(merge_runs)
    data = data.merge(
        train_data,
        on=["model_key", "adv_training_round"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_train"),
    )
    assert (
        data.train_total_flops.notnull().all()
    ), "Some adversarial training rounds are missing FLOPs data. "

# %%
df = data.copy()
df["model_idx"] = df.model_size.rank(method="dense").astype(int) - 1
df["pretrain_compute"] = df.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)
df["logit_asr_at_12"] = TRANSFORMS["logit"](df.metrics_asr_at_12)
df["log_pretrain_compute"] = df.pretrain_compute.apply(TRANSFORMS["log"])
df["log_train_total_flops"] = TRANSFORMS["log"](df.train_total_flops)
df["pretrain_compute_percent"] = (df.train_total_flops / df.pretrain_compute).astype(
    float
)
df["model_name"] = df.model_idx.apply(lambda x: MODEL_PLOTTING_NAMES[x])
df = df.loc[
    np.isfinite(df.logit_asr_at_12)
    & np.isfinite(df.log_pretrain_compute)
    & np.isfinite(df.log_train_total_flops)
]
df.head()
# %%
adv_reg = smf.ols("logit_asr_at_12 ~ log_train_total_flops", data=df).fit()
adv_reg.summary()
# %%
# plot residuals by pretrain compute
fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle("Residuals vs. Pretraining FLOPs")
sns.boxplot(x="log_pretrain_compute", y=adv_reg.resid, data=df, ax=ax)
ax.set_xlabel(AXIS_LABELS["log_pretrain_compute"])
ax.set_ylabel("Residuals")
# rotate x ticks
_ = plt.xticks(rotation=45)
# %%
# %%
reg = smf.ols("logit_asr_at_12 ~ pretrain_compute_percent", data=df).fit()
reg.summary()
# %%
reg = smf.ols(
    "logit_asr_at_12 ~ pretrain_compute_percent + C(num_params)", data=df
).fit()
reg.summary()
# %%
model_gradients = (
    reg.params.filter(like="C(num_params)", axis=0)
    + reg.params["pretrain_compute_percent"]
)
model_names = model_gradients.index.str.extract(r"\[T.(.*)\]").squeeze()
# %%
fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle("Gradient of ASR vs. % Pretrain Compute by Model Size")
ax.plot(model_names.astype(int).values, model_gradients)
ax.set_xlabel(AXIS_LABELS["model_size"])
ax.set_xscale("log")
ax.set_ylabel("Gradient of ASR vs. % Pretrain Compute")
# %%
for pretrain_compute, group in df.groupby("log_pretrain_compute"):
    reg = smf.ols("logit_asr_at_12 ~ log_train_total_flops", data=group).fit()
    print(f"Pretrain Compute: {pretrain_compute:.2f}")
    print(
        reg.params["Intercept"]
        + reg.params["log_train_total_flops"] * df["log_train_total_flops"].mean()
    )
# %%
mod = smf.ols("logit_asr_at_12 ~ log_train_total_flops", data=df).fit()
for pretrain_compute, group in df.groupby("log_pretrain_compute"):
    resid = group["logit_asr_at_12"] - mod.predict(group)
    print(f"Pretrain Compute: {pretrain_compute:.2f}, Mean resid: {resid.mean():.2f}")
# %%
fig, ax = plt.subplots(figsize=(12, 10))
ax.set(xscale="log", yscale="log")
sns.scatterplot(
    x="pretrain_compute",
    y="train_total_flops",
    hue="logit_asr_at_12",
    data=df,
    ax=ax,
)
ax.set_ylabel(AXIS_LABELS["train_total_flops"])
ax.set_xlabel(AXIS_LABELS["pretrain_compute"])
# %%
# Create a grid for interpolation
x = df.log_pretrain_compute
y = df.log_train_total_flops
z = df["logit_asr_at_12"]

# %%
# Determine the common range for both axes
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())

# Determine the range for each axis
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Add a small padding to ensure all points are included
padding_x = 0.05 * (x_max - x_min)
padding_y = 0.05 * (y_max - y_min)
x_min -= padding_x
x_max += padding_x
y_min -= padding_y
y_max += padding_y

# Create the grid
xi, yi = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Interpolate the scattered data onto the grid using 'linear' method
zi = griddata((x, y), z, (xi, yi), method="linear")

# Check the range of interpolated values
print(f"Original z range: [{z.min():.2f}, {z.max():.2f}]")
print(f"Interpolated z range: [{np.nanmin(zi):.2f}, {np.nanmax(zi):.2f}]")

# Limit the interpolated values to the original data range
zi = np.clip(zi, z.min(), z.max())

# Create the plot
fig, ax = plt.subplots(figsize=(12, 12))  # Square figure

# Create filled contour plot
contourf = ax.contourf(
    xi, yi, zi, levels=20, cmap="YlOrRd", extend="both", vmin=z.min(), vmax=z.max()
)
plt.colorbar(contourf, label="Logit ASR at 12")

# Add contour lines
contour = ax.contour(xi, yi, zi, levels=10, colors="k", alpha=0.3)
ax.clabel(contour, inline=True, fontsize=8)

# Plot the original scattered data points
scatter = ax.scatter(
    x,
    y,
    c=z,
    cmap="YlOrRd",
    s=50,
    alpha=0.7,
    edgecolors="k",
    vmin=z.min(),
    vmax=z.max(),
)

ax.set_xlabel(AXIS_LABELS["pretrain_compute"])
ax.set_ylabel("Adversarial Training FLOPs (log10)")
ax.set_title("Logit ASR at 12 vs Pretraining and Adversarial Training FLOPs")

# Set equal aspect ratio
# ax.set_aspect('equal')

# # Set the same limits for both axes
# ax.set_xlim(min_val, max_val)
# ax.set_ylim(min_val, max_val)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


# Convert log scale ticks to original values
def log_tick_formatter(val, pos=None):
    return f"1e{int(val)}"


ax.xaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))  # type: ignore
ax.yaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))  # type: ignore

# # Add diagonal line
# ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal FLOPs')
ax.legend()

plt.tight_layout()
plt.show()
# %%
# Determine the overall min and max for both types of FLOPs
step = (
    max(
        df.log_train_total_flops.max() - df.log_train_total_flops.min(),
        df.log_pretrain_compute.max() - df.log_pretrain_compute.min(),
    )
    / 10
)

# Create equally spaced bins across the entire range
pretrain_bins = np.arange(
    df.log_pretrain_compute.min(), df.log_pretrain_compute.max() + step, step
)
adv_train_bins = np.arange(
    df.log_train_total_flops.min(), df.log_train_total_flops.max() + step, step
)

df["pretrain_bin"] = pd.cut(df.log_pretrain_compute, bins=pretrain_bins)
df["adv_train_bin"] = pd.cut(df.log_train_total_flops, bins=adv_train_bins)

# Create a pivot table
pivot = df.pivot_table(
    values="logit_asr_at_12",
    index="adv_train_bin",
    columns="pretrain_bin",
    aggfunc="mean",
)
pivot = pivot.sort_index(ascending=False)
pivot.style.background_gradient(cmap="YlOrRd", axis=None).format(na_rep="")
# %%
# fit linear regression to bin mids
df["pretrain_mid"] = df.pretrain_bin.apply(lambda x: x.mid).astype(float)
df["adv_train_mid"] = df.adv_train_bin.apply(lambda x: x.mid).astype(float)
smf.ols("logit_asr_at_12 ~ pretrain_mid + adv_train_mid", data=df).fit().summary()
# %%

# Calculate diagonal minima
diagonal_mins = set()
skipped_diagonals = []
for k in range(-(len(pivot.columns) - 1), len(pivot.index)):
    diagonal = np.diagonal(pivot.values, offset=k)
    if np.all(np.isnan(diagonal)):
        skipped_diagonals.append(k)
        continue
    min_index = np.nanargmin(diagonal)
    if k < 0:
        i, j = min_index, min_index - k
    else:
        i, j = min_index + k, min_index
    diagonal_mins.add((j, i))


# Create a mask for highlighting
mask = pivot.copy()
mask[:] = False
for i, j in diagonal_mins:
    mask.iloc[i, j] = True

# Create the plot
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd_r",
    ax=ax,
    cbar_kws={"label": AXIS_LABELS["logit_asr_at_12"]},
)

# Highlight the diagonal minima
for i, j in diagonal_mins:
    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="blue", lw=3))  # type: ignore # noqa

ax.set_title(
    "Attack Success Rate vs Pretraining and Adversarial Training FLOPs", fontsize=16
)
ax.set_xlabel(AXIS_LABELS["log_pretrain_compute"], fontsize=12)
ax.set_ylabel(AXIS_LABELS["train_total_flops"], fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.show()
# %%
mod = smf.ols(
    "logit_asr_at_12 ~ log_pretrain_compute + log_train_total_flops", data=df
).fit()
mod.summary()
# %%
mod = smf.ols(
    "logit_asr_at_12 ~ log_pretrain_compute * log_train_total_flops", data=df
).fit()
mod.summary()
# %%
df[["log_pretrain_compute", "log_train_total_flops", "logit_asr_at_12"]].describe()
# %%
df.loc[np.isclose(df.log_train_total_flops, df.log_train_total_flops.iloc[50])]
# %%
# Create a grid of points
x = np.linspace(19, 22, 100)
y = np.linspace(15, 19, 100)
X, Y = np.meshgrid(x, y)

# Prepare the data for prediction
X_pred = pd.DataFrame(
    {"log_pretrain_compute": X.ravel(), "log_train_total_flops": Y.ravel()}
)


# Make predictions
Z = mod.predict(X_pred).values.reshape(X.shape)

# Create the contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
plt.colorbar(contour, label=AXIS_LABELS["logit_asr_at_12"])
plt.xlabel(AXIS_LABELS["log_pretrain_compute"])
plt.ylabel(AXIS_LABELS["log_train_total_flops"])
plt.title("Contour Plot of Regression Model")

# Add contour lines
contour_lines = plt.contour(X, Y, Z, levels=10, colors="white", linestyles="dashed")
plt.clabel(contour_lines, inline=True, fontsize=8)

plt.tight_layout()
plt.show()
# %%
