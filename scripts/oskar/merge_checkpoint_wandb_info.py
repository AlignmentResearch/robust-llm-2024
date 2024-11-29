# %%
from pathlib import Path

import pandas as pd

from robust_llm.file_utils import compute_repo_path
from robust_llm.wandb_utils.wandb_api_tools import get_runs_for_group


# %%
def color_rows(row):
    """Return a Series with background color for rows modified within last hour"""
    is_recent = (row["reference_ts"] - row["modified_ts"]) < 3600  # 1 hour in seconds
    is_finished = row["current_epochs"] == row["total_epochs"]
    if is_finished:
        color = "background-color: green"
    elif is_recent:
        color = "background-color: yellow"
    else:
        color = "background-color: red"
    return [color] * len(row)


# %%
def style_df(df):
    return (
        df.style.apply(color_rows, axis=1)
        .format({"progress": "{:.1%}"})
        .hide(subset=["modified_ts", "reference_ts"], axis=1)
    )


# %%
root = compute_repo_path()
path = Path(root) / "outputs" / "checkpoint_analysis" / "1732648120.540357.csv"
df = pd.read_csv(path)
df["run_name"] = df["group"] + "_" + df["run"].apply(lambda x: f"{x:04d}")
max_df = df.groupby(["group", "run"]).max().reset_index()
# %%
max_df
# %%
for group, _ in df.groupby("group"):
    assert isinstance(group, str)
    runs = get_runs_for_group(group, use_cache=True)
    print(runs)
    break
# %%
