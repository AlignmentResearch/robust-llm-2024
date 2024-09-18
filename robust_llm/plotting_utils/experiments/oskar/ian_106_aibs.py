# %%
import os

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
def plot_aib_for_group(group_name: str):
    path = f"{OUTPUTS_DIR}/aibs_{group_name}.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    assert df.columns.tolist() == ["model_idx", "seed_idx", "aib", "decile"]
    assert df.model_idx.between(0, 4).all()
    assert df.seed_idx.between(0, 9).all()
    assert df.decile.between(0, 10).all()
    assert len(df) == 11 * 5 * 10  # 11 deciles, 5 seeds, 10 models

    df["model"] = df.model_idx.apply(lambda x: MODEL_NAMES[x])
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    fig.suptitle(f"Average Initial Breach metric for {group_name}")
    sns.lineplot(data=df, x="decile", y="aib", hue="model", ax=ax, palette="viridis")
    ax.set_ylabel("Average iteration of Initial Breach for bottom (x * 10%)")


# %%
for group in GROUPS:
    plot_aib_for_group(group)

# %%
