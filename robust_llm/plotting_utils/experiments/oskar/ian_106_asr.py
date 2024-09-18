# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

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
def plot_asr_for_group(group_name: str):
    path = f"{OUTPUTS_DIR}/asr_{group_name}.csv"
    if not os.path.exists(path):
        return
    attack = group_name.split("_")[2]
    dataset = group_name.split("_")[-1]
    Path(f"{OUTPUTS_DIR}/asr/{attack}").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    assert df.columns.tolist() == ["model_idx", "seed_idx", "asr", "iteration"]
    assert df.model_idx.between(0, 9).all()
    assert df.seed_idx.eq(0, 4).all()
    assert df.iteration.between(0, 10).all()
    assert len(df) == 11 * 5 * 10  # 11 iterations, 5 seeds, 10 models

    df.to_csv(f"{OUTPUTS_DIR}/asr/{attack}/{dataset}.csv", index=False)
    if df.iteration.max() > 1000:
        df = df.loc[df.iteration.mod(100) == 0]
    df["model"] = df.model_idx.apply(lambda x: MODEL_NAMES[x])
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    fig.suptitle(f"Attack Scaling for {group_name}")
    sns.lineplot(data=df, x="iteration", y="asr", hue="model", ax=ax, palette="viridis")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Attack success rate (%)")
    fig.savefig(f"{OUTPUTS_DIR}/asr/{attack}/{dataset}.pdf")


# %%
for group in tqdm(GROUPS):
    plot_asr_for_group(group)

# %%
