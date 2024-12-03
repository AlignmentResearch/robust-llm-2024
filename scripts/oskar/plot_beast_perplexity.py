# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from robust_llm.file_utils import compute_repo_path

# Examples from:
# https://wandb.ai/farai/robust-llm/runs/nborq6x9/
# https://wandb.ai/farai/robust-llm/runs/6ghur78c
# https://wandb.ai/farai/robust-llm/runs/vwgm1iqx
# https://wandb.ai/farai/robust-llm/runs/dysrirku
# https://wandb.ai/farai/robust-llm/runs/y6of822u
# %%
WINDOW_SIZE = 10
# %%
torch.set_grad_enabled(False)
root = Path(compute_repo_path()) / "scripts" / "oskar" / "beast_examples"
assert root.exists()
# %%
data_list = []
for path in root.iterdir():
    df = pd.read_csv(path)
    df.attacked_success = df.attacked_success.astype(bool)
    df = df.loc[~df.attacked_success]
    data_list.append(df)
data = pd.concat(data_list).reset_index(drop=True)
# %%
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")


# %%
def get_perplexity(text: str) -> torch.Tensor:
    encoding = tokenizer(text, return_tensors="pt")
    tokens = encoding.input_ids  # [batch, pos]
    logits = model(**encoding).logits  # [batch, pos, d_vocab]
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather the logprobs corresponding to the next token
    gathered = torch.gather(logprobs, -1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    # https://huggingface.co/docs/transformers/en/perplexity
    # Perplexity is defined as the exponentiated average negative log-likelihood
    perplexity = torch.exp(-gathered)
    return perplexity.squeeze()


# %%
def get_rolling_mean(tensor: torch.Tensor, window_size: int = WINDOW_SIZE) -> pd.Series:
    """Compute rolling means over window_size and plot the results."""
    perplexity = tensor.squeeze().numpy()
    rolling_mean = pd.Series(perplexity).rolling(window=window_size).mean()
    assert isinstance(rolling_mean, pd.Series)
    assert perplexity.shape == rolling_mean.shape
    head = rolling_mean[: window_size - 1]
    assert isinstance(head, pd.Series)
    assert bool(head.isnull().all())
    return rolling_mean


# %%
def plot_original_attacked_perplexity(
    original: torch.Tensor,
    attacked: torch.Tensor,
    example: int,
):
    attack_position = len(original)
    assert torch.allclose(
        attacked[: len(original) - WINDOW_SIZE],
        original[: len(original) - WINDOW_SIZE],
        rtol=0.1,
        atol=10,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    attacked_rolling = get_rolling_mean(attacked)
    ax.plot(attacked_rolling)
    ax.vlines(
        attack_position,
        ymin=attacked_rolling.min(),
        ymax=attacked_rolling.max(),
        color="red",
        linestyle="--",
    )
    ax.set_title(f"Perplexity in window (size={WINDOW_SIZE}, example={example})")
    # Add text to the plot for original.mean() and attacked.mean()
    ax.text(
        attack_position,
        0.9 * attacked_rolling.min(),
        "Attack position",
        color="red",
    )
    ax.text(
        attack_position,
        attacked_rolling.max().item(),
        f"Original mean: {original.mean().item():.2E}\n"
        f"Attacked mean: {attacked.mean().item():.2E}",
        color="black",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    return fig, ax


# %%
for idx, row in data.iterrows():
    assert isinstance(idx, int)
    original_perplexity = get_perplexity(row.original_text)
    attacked_perplexity = get_perplexity(row.attacked_text)
    rolling_perplexity = get_rolling_mean(attacked_perplexity)
    attack_position = len(original_perplexity)
    data.loc[idx, "original_perplexity_mean"] = original_perplexity.mean().item()  # type: ignore # noqa
    data.loc[idx, "attacked_perplexity_mean"] = attacked_perplexity.mean().item()  # type: ignore # noqa
    data.loc[idx, "original_rolling_max"] = rolling_perplexity[:attack_position].max()  # type: ignore # noqa
    data.loc[idx, "attacked_rolling_max"] = rolling_perplexity.max()  # type: ignore # noqa
    plot_original_attacked_perplexity(
        original_perplexity, attacked_perplexity, example=idx
    )


# %%
def plot_comparison_histogram(
    original: str = "original_perplexity_mean",
    attacked: str = "attacked_perplexity_mean",
    title: str = "Comparison of perplexity means",
):
    # Pre-compute bins using both distributions
    all_values = pd.concat([data[original], data[attacked]])
    bins = np.linspace(all_values.min(), all_values.max(), 30)

    # Plot two histograms using the same bins
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)
    sns.histplot(
        data=data,
        x=original,
        color="blue",
        alpha=0.5,
        label="Original",
        bins=bins,
        ax=ax,
        multiple="layer",
    )
    sns.histplot(
        data=data,
        x=attacked,
        color="red",
        alpha=0.5,
        label="Attacked",
        bins=bins,
        ax=ax,
        multiple="layer",
    )

    ax.legend()
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


# %%
plot_comparison_histogram(
    "original_perplexity_mean",
    "attacked_perplexity_mean",
    "Comparison of perplexity means",
)
# %%
plot_comparison_histogram(
    "original_rolling_max",
    "attacked_rolling_max",
    "Comparison of rolling maxima",
)
# %%
data[
    [
        "original_perplexity_mean",
        "attacked_perplexity_mean",
        "original_rolling_max",
        "attacked_rolling_max",
    ]
].style.format(
    "{:.2E}"
).background_gradient(  # type: ignore
    cmap="Reds"
)
# %%
data.eval("attacked_rolling_max > original_rolling_max").value_counts()

# %%
