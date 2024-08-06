# %%
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robust_llm.plotting_utils.tools import (
    extract_size_from_model_name,
    get_metrics_adv_training,
)

# %%
# increase pandas max string length
pd.options.display.max_colwidth = 100

# %%
SUMMARY_KEYS = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.training.seed",
    "model_size",
    "experiment_yaml.dataset.n_val",
    "experiment_yaml.evaluation.evaluation_attack.adversary.name_or_path",
    "experiment_yaml.evaluation.evaluation_attack.n_its",
    "experiment_yaml.run_name",
    "experiment_yaml.experiment_name",
]

GROUPS = [
    path.replace(".py", "")
    for path in os.listdir("/home/oskar/projects/robust-llm/experiments/oskar")
    if path.startswith("oskar_009e")
    or path.startswith("oskar_009f")
    or path.startswith("oskar_009g")
    or path.startswith("oskar_009m")
]
METRICS = ["adversarial_eval/attack_success_rate"]
# %%
print(GROUPS)
# %%
# ##############################################################################
# Load data
# ##############################################################################
# %%
data = []
for group in GROUPS:
    group_df = get_metrics_adv_training(group, METRICS, SUMMARY_KEYS)
    group_df["group"] = group
    data.append(group_df)
# %%
concat_df = pd.concat(data)
dupes = concat_df.experiment_yaml_run_name.duplicated()
null_columns = concat_df.isnull().all(axis=1)
assert isinstance(dupes, pd.Series)
assert isinstance(null_columns, pd.Series)
assert not dupes.any()
assert not null_columns.any()
# %%
concat_df.loc[concat_df.experiment_yaml_run_name.duplicated()]
# %%
print(concat_df.columns.tolist())
# %%
print(concat_df.experiment_yaml_experiment_name.value_counts())


# %%
def extract_attack_from_experiment_name(experiment_name):
    lm_attacks = ["pair", "few_shot", "zs"]
    for name in lm_attacks:
        if name in experiment_name:
            return name
    gcg_match = re.search(r"gcg_(\d+)it", experiment_name)
    if gcg_match:
        return f"gcg-{gcg_match.group(1)}-its"

    return "unknown"


# %%
df = concat_df.reset_index(drop=True)
df.columns = df.columns.str.replace("experiment_yaml_", "")
df.columns = df.columns.str.replace("adversarial_eval/", "")
df["attack_name"] = df.experiment_name.apply(extract_attack_from_experiment_name)
df.model_name_or_path = (
    df.model_name_or_path.str.replace("-Chat", "")
    .str.replace("-Instruct", "")
    .str.replace("-chat", "")
)
df["model_family"] = df.model_name_or_path.str.split("/").str[1].str.split("-").str[0]
df["adversary_size"] = df.evaluation_evaluation_attack_adversary_name_or_path.apply(
    extract_size_from_model_name
)
df["adversary_family"] = (
    df.evaluation_evaluation_attack_adversary_name_or_path.str.split("/")
    .str[1]
    .str.split("-")
    .str[0]
)
df.model_size = df.model_size.astype(int)
print(df.attack_name.value_counts())
# %%
df[["model_size", "attack_success_rate", "attack_name"]].style.background_gradient(
    cmap="RdBu", axis=0
).format({"attack_success_rate": "{:.2f}", "model_size": "{:.1E}"})
# %%
# Use attack_success_rate and dataset_n_val to calculate standard error
df["attack_success_rate_se"] = df.eval(
    "(attack_success_rate * (1 - attack_success_rate) / dataset_n_val) ** 0.5"
)
# %%
# Create the ASR error bar figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the lines with seaborn
lineplot = sns.lineplot(
    data=df,
    x="model_size",
    y="attack_success_rate",
    hue="attack_name",
    ax=ax,
    marker="o",  # Add markers to the line plot
)

# Add error bars with colors matching the lines
for line in lineplot.lines:
    color = line.get_color()
    for name, group in df.groupby("attack_name"):
        assert isinstance(group, pd.DataFrame)
        if line.get_label() == name:
            ax.errorbar(
                group["model_size"],
                group["attack_success_rate"],
                yerr=group["attack_success_rate_se"],
                fmt="none",  # Do not plot data points again
                capsize=5,  # Add caps to the error bars
                color=color,  # Match the error bar color with the line color
                label="_nolegend_",  # Do not add extra legend entries
            )

# Set y-axis minimum value to 0
ax.set_ylim(bottom=0)

# Add labels and title if needed
ax.set_xlabel("Model Size")
ax.set_ylabel("Attack Success Rate")
ax.set_title("Attack Success Rate vs Model Size with Error Bars")

# Show the plot
plt.show()
# %%
