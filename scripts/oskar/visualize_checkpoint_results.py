# %%
from pathlib import Path

import pandas as pd

from robust_llm.file_utils import compute_repo_path


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
path = Path(root) / "outputs" / "checkpoint_analysis.csv"
df = pd.read_csv(path)
df.progress = df.current_epochs / df.total_epochs
max_df = df.groupby(["group", "run"]).max().reset_index()
max_df.progress = max_df.current_epochs / max_df.total_epochs
max_df.to_csv(path.with_name("checkpoint_analysis_max.csv"), index=False)
# %%
style_df(df)
# %%
style_df(max_df)
# %%
