# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
OUTPUTS_DIR = "../../../../outputs"
# %%
MODEL_NAMES = [
    "pythia-14m",
    "pythia-31m",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]


# %%
def plot_ifs_for_group(group_name: str):
    path = f"{OUTPUTS_DIR}/ifs_{group_name}.csv"
    if not os.path.exists(path):
        return
    attack = group_name.split("_")[2]
    dataset = group_name.split("_")[-1]
    Path(f"{OUTPUTS_DIR}/ifs/{attack}").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    assert df.columns.tolist() == ["model_idx", "seed_idx", "ifs", "decile"]
    assert df.model_idx.between(0, 9).all()
    assert df.seed_idx.between(0, 4).all()
    assert df.decile.between(0, 10).all()
    assert len(df) == 11 * 5 * 10  # 11 deciles, 5 seeds, 10 models

    df.to_csv(f"{OUTPUTS_DIR}/ifs/{attack}/{dataset}.csv", index=False)
    df["model"] = df.model_idx.apply(lambda x: MODEL_NAMES[x])
    df["asr"] = df.decile.mul(10)
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    fig.suptitle(f"Iterations For Success metric for {group_name}")
    sns.lineplot(data=df, x="asr", y="ifs", hue="model", ax=ax, palette="viridis")
    ax.set_ylabel("Iterations required to reach attack success rate")
    ax.set_xlabel("Attack success rate (%)")
    fig.savefig(f"{OUTPUTS_DIR}/ifs/{attack}/{dataset}.pdf")


# %%
for group in GROUPS:
    plot_ifs_for_group(group)

# %%
