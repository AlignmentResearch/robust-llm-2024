# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from robust_llm.plotting_utils.tools import get_metrics_adv_training

# %%
# increase pandas max string length
pd.options.display.max_colwidth = 100

# %%
SUMMARY_KEYS = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.training.seed",
    "model_size",
    "experiment_yaml.evaluation.evaluation_attack.adversary.name_or_path",
]

GROUPS = [
    path.replace(".py", "")
    for path in os.listdir("/home/oskar/projects/robust-llm/experiments/oskar")
    if path.startswith("oskar_001o")
]
METRICS = ["adversarial_eval/attack_success_rate"]
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


# %%
def convert_to_int(value):
    assert isinstance(value, str)
    value = value.lower()
    if "m" in value:
        return int(float(value.replace("m", "")) * 1_000_000)
    elif "b" in value:
        return int(float(value.replace("b", "")) * 1_000_000_000)
    elif "k" in value:
        return int(float(value.replace("k", "")) * 1_000)
    return value


# %%
df = concat_df.copy()
df = df.loc[df.run_id.ne("iyssvifc")].copy()
df.columns = df.columns.str.replace("experiment_yaml_", "")
df.columns = df.columns.str.replace("adversarial_eval/", "")
df.model_name_or_path = (
    df.model_name_or_path.str.replace("-Chat", "")
    .str.replace("-Instruct", "")
    .str.replace("-chat", "")
)
df["model_family"] = df.model_name_or_path.str.split("/").str[1].str.split("-").str[0]
df["adversary_size"] = (
    df.evaluation_evaluation_attack_adversary_name_or_path.str.replace("-hf", "")
    .str.split("-")
    .str[-1]
    .apply(convert_to_int)
)
df["adversary_family"] = (
    df.evaluation_evaluation_attack_adversary_name_or_path.str.split("/")
    .str[1]
    .str.split("-")
    .str[0]
)
df.model_size = df.model_size.astype(int)
df.head()
# %%
df.model_family.value_counts()
# %%
len(df)
# %%
# %%
# ##############################################################################
# Diagonal
# ##############################################################################
# %%
diagonal = df.loc[
    df.model_name_or_path == df.evaluation_evaluation_attack_adversary_name_or_path,
    [
        "model_name_or_path",
        "attack_success_rate",
        "model_size",
        "model_family",
    ],
]
# %%
fig, ax = plt.subplots()
sns.scatterplot(
    data=diagonal,
    x="model_size",
    y="attack_success_rate",
    hue="model_family",
    alpha=0.5,
    ax=ax,
)
ax.set(xscale="log")
fig.suptitle("Diagonal attack success rate vs model size (chat-tuned victim)")
# %%
# ##############################################################################
# Two-way Table
# ##############################################################################
# %%
qwen1_5 = df.loc[df.model_family.eq("Qwen1.5") & df.adversary_family.eq("Qwen1.5")]
pt = qwen1_5.pivot(
    index="adversary_size",
    columns="model_size",
    values="attack_success_rate",
)
styler = (
    pt.style.background_gradient(cmap="RdBu", axis=None)
    .format("{:.0%}")
    .highlight_null(color="white")
)
styler.set_caption("Qwen1.5 two-way table of Attack Success Rate")
# format index and columns in scientific notation
styler.format_index("{:.0e}", axis=0)
styler.format_index("{:.0e}", axis=1)

styler
# %%
# ##############################################################################
# Regressions
# ##############################################################################
# %%
lm = smf.ols(
    "attack_success_rate ~ model_size",
    data=df.loc[df.model_family.eq("Qwen1.5") & df.adversary_family.eq("Qwen1.5")],
).fit()
lm.summary()
# %%
lm = smf.ols(
    "attack_success_rate ~ model_size + adversary_size",
    data=df.loc[df.model_family.eq("Qwen1.5") & df.adversary_family.eq("Qwen1.5")],
).fit()
lm.summary()
# %%
# ##############################################################################
# Marginals
# ##############################################################################
# %%
fig, ax = plt.subplots()
sns.lineplot(
    x="adversary_size",
    y="attack_success_rate",
    hue="model_size",
    data=qwen1_5,
    ax=ax,
)
# %%
fig, ax = plt.subplots()
sns.lineplot(
    x="model_size",
    y="attack_success_rate",
    hue="adversary_size",
    data=qwen1_5,
    ax=ax,
)
# %%
