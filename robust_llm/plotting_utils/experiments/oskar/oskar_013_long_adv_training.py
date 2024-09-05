# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robust_llm.plotting_utils.tools import get_metrics_adv_training

# %%
# increase pandas max string length
pd.options.display.max_colwidth = 100
# increase pandas max columns
pd.options.display.max_columns = 100

# %%
SUMMARY_KEYS = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.training.seed",
    "model_size",
    "experiment_yaml.run_name",
    "experiment_yaml.experiment_name",
    "experiment_yaml.environment.deterministic",
    "experiment_yaml.training.adversarial.only_add_successful_adversarial_examples",
    "experiment_yaml.training.adversarial.num_examples_to_generate_each_round",
    "experiment_yaml.training.adversarial.max_adv_data_proportion",
    "experiment_yaml.training.adversarial.target_adversarial_success_rate",
]

GROUPS = [
    path.replace(".py", "")
    for path in os.listdir("/home/oskar/projects/robust-llm/experiments/oskar")
    if path.startswith("oskar_013")
]
METRICS = [
    "adversarial_eval/attack_success_rate",
    "adversarial_eval/pre_attack_accuracy",
]
# %%
print(GROUPS)
# %%
# ##############################################################################
# Load data
# ##############################################################################
# %%
data = []
for group in GROUPS:
    group_df = get_metrics_adv_training(group, METRICS, SUMMARY_KEYS, verbose=False)
    group_df["group"] = group
    data.append(group_df)
# %%
concat_df = pd.concat(data)
concat_df = concat_df.loc[concat_df.run_state.eq("finished")]
dupes = concat_df[
    ["experiment_yaml_run_name", "adv_training_round", "_step"]
].duplicated()
null_columns = concat_df.isnull().all(axis=1)
assert isinstance(dupes, pd.Series)
assert isinstance(null_columns, pd.Series)
print("Dupes:", dupes.sum())
assert not null_columns.any()
# %%
print(concat_df.columns.tolist())
# %%
print(concat_df.experiment_yaml_experiment_name.value_counts())
# %%
OVERRIDE_LABEL_DICT = {
    "013a": [
        "baseline",
        "max_adv_prop_90",
        "target_asr_75",
        "target_asr_90",
        "adv_sampling_decay_1_pct",
        "adv_sampling_decay_1_bp",
        "adv_sampling_decay_1_bp_rerun",
    ],
    "013b": [
        "uniform_sampling",
        "only_add_successful",
        "loss_rank_weight_50pct",
        "loss_rank_weight_100pct",
    ],
}


# %%
df = (
    concat_df.groupby(["experiment_yaml_run_name", "adv_training_round"])
    .last()
    .reset_index()
).sort_values("adv_training_round")
df.columns = df.columns.str.replace("experiment_yaml_", "")
df.columns = df.columns.str.replace("adversarial_eval/", "")
df.columns = df.columns.str.replace("training_adversarial_", "")
df["experiment_num"] = df.run_name.str.split("-").str[-1].astype(int)
df["group_short"] = df.group.str.split("_").str[1]
df["n_override_labels"] = df.group_short.map(lambda x: len(OVERRIDE_LABEL_DICT[x]))
df["override_num"] = df.experiment_num.mod(df.n_override_labels)
df["override_name"] = df.apply(
    lambda x: OVERRIDE_LABEL_DICT[x.group_short][x.override_num], axis=1
)
assert ((df.experiment_num // df.n_override_labels) == df.training_seed).all()
df.model_name_or_path = (
    df.model_name_or_path.str.replace("-Chat", "")
    .str.replace("-Instruct", "")
    .str.replace("-chat", "")
)
assert set(df.override_name.tolist()) == set(
    [value for values in OVERRIDE_LABEL_DICT.values() for value in values]
)
df = df.loc[~df.override_name.str.endswith("rerun")]
print(df.override_name.value_counts())
# %%
# ##############################################################################
# Clean accuracy
# ##############################################################################
# %%
fig, ax = plt.subplots(figsize=(16, 12))
fig.suptitle("Pre-attack accuracy during adversarial training (1 SE bars)")
sns.lineplot(
    data=df,
    x="adv_training_round",
    y="pre_attack_accuracy",
    hue="override_name",
    errorbar=("se", 1),
)
# %%
# ##############################################################################
# Attack success rate curve
# ##############################################################################
# %%
fig, ax = plt.subplots(figsize=(16, 12))
fig.suptitle("Attack success rate during adversarial training (1 SE bars)")
sns.lineplot(
    data=df,
    x="adv_training_round",
    y="attack_success_rate",
    hue="override_name",
    errorbar=("se", 1),
    alpha=0.5,
)
# %%
# ##############################################################################
# Attack success rate difference from baseline
# ##############################################################################
# %%
baseline_df = df.loc[df.override_name.eq("baseline")]
merge_df = df.merge(
    baseline_df,
    on=["adv_training_round", "training_seed"],
    suffixes=("", "_baseline"),
    validate="many_to_one",
)
merge_df["attack_success_rate_diff"] = (
    merge_df.attack_success_rate - merge_df.attack_success_rate_baseline
)
merge_df["attack_success_rate_pct_diff"] = (
    100 * merge_df.attack_success_rate_diff / merge_df.attack_success_rate_baseline
)
merge_df.head()
# %%
fig, ax = plt.subplots(figsize=(16, 12))
fig.suptitle("Attack success rate difference from baseline")
sns.boxplot(data=merge_df, x="override_name", y="attack_success_rate_diff")
_ = plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
# %%
# plot mean and std error instead of boxplot
fig, ax = plt.subplots(figsize=(16, 12))
fig.suptitle("Attack success rate difference from baseline (1 SE bars)")
sns.barplot(
    data=merge_df,
    x="override_name",
    y="attack_success_rate_diff",
    errorbar=("se", 1),
)
_ = plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
# zero x-axis
plt.axhline(0, color="black", linewidth=0.5)
# %%
