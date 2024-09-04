import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robust_llm.file_utils import compute_repo_path
from robust_llm.wandb_utils.constants import (
    FINAL_PYTHIA_CHECKPOINT,
    METRICS,
    MODEL_NAMES,
    MODEL_SIZES,
    SUMMARY_KEYS,
)
from robust_llm.wandb_utils.wandb_api_tools import (
    get_metrics_adv_training,
    get_metrics_single_step,
)


def extract_size_from_model_name(name: str | None) -> int | None:
    if name is None:
        return None
    name = name.lower()
    pattern = r"-(\d+(\.\d+)?)([kmb])(-|$)"
    match = re.search(pattern, name)
    if match:
        number = float(match.group(1))
        suffix = match.group(3)
        if suffix == "k":
            return int(number * 1e3)
        elif suffix == "m":
            return int(number * 1e6)
        elif suffix == "b":
            return int(number * 1e9)
        else:
            raise ValueError(f"Unknown suffix {suffix}")
    else:
        raise ValueError(f"Could not find model size in {name}")


def make_finetuned_plot(
    run_names: tuple[str, ...],
    eval_summary_keys: tuple[list[str], ...],
    metrics: tuple[list[str], ...],
    title: str,
    save_as: str,
    save_dir: str,
    custom_ys: list[float] | None = None,
    custom_xs_and_ys: list[tuple[float, float]] | None = None,
    scatterplot: bool = False,
):
    """
    Plot model size on the x-axis and attack success rate on the y-axis.

    Args:
        run_names: The names of the runs to pull data from.
        eval_summary_keys: The keys to summarize the data by.
        metrics: The metrics to plot.
        title: The title to give the plot.
        save_as: The name to save the plot as.
        save_dir: The directory to save the plot in.
        custom_ys: Custom y-values to add to the plot. The x-values are taken
            from MODEL_SIZES, assuming that the custom y-values are for
            the largest models. Mutually exclusive with custom_xs_and_ys.
        custom_xs_and_ys: Custom x-values and y-values to add to the plot.
            Mutually exclusive with custom_ys.
        scatterplot: Whether to use a scatterplot instead of a line plot.
    """
    if title is None:
        title = run_names[0]

    if metrics is None:
        metrics = (METRICS,) * len(run_names)

    if eval_summary_keys is None:
        eval_summary_keys = (SUMMARY_KEYS,) * len(run_names)

    print("Plotting with data from ", run_names)

    runs = []
    for run_name, metric_list, summary_key_list in zip(
        run_names, metrics, eval_summary_keys
    ):
        run = get_metrics_single_step(
            group=run_name,
            metrics=metric_list,
            summary_keys=summary_key_list,
        )
        runs.append(run)

    print("Now the runs are", runs)

    for run in runs:
        postprocess_data(run)

    # Concatenate the runs together
    run = pd.concat(runs, ignore_index=True)

    draw_plot(
        run,
        title=title,
        save_as=save_as,
        save_dir=save_dir,
        custom_ys=custom_ys,
        custom_xs_and_ys=custom_xs_and_ys,
        scatterplot=scatterplot,
    )


def load_and_plot_adv_training_plots(
    name: str,
    title: str,
    save_as: str,
    save_dir: str = "",
    summary_keys: list[str] | None = None,
    metrics: list[str] | None = None,
    xlim: tuple[float, float] = (1, 10),
    legend: bool = False,
):
    """
    Make adversarial training plots for a given run, pulling data from W&B.

    Args:
        name: The name of the run to pull data from.
        title: The title to give the plot (also used for saving).
        save_as: The name to save the plot as.
        save_dir: The directory to save the plot in.
        summary_keys: The keys to summarize the data by.
        metrics: The metrics to plot.
        xlim: The x-axis limits.
        legend: Whether to include the legend in the plot.
    """
    if summary_keys is None:
        summary_keys = SUMMARY_KEYS

    if metrics is None:
        metrics = METRICS

    data = prepare_adv_training_data(
        name,
        summary_keys=summary_keys,
        metrics=metrics,
    )
    draw_plot_adv_training(
        data=data,
        x="adv_training_round",
        split_curves_by="num_params",
        title=title,
        save_as=save_as,
        save_dir=save_dir,
        xlim=xlim,
        legend=legend,
    )


def prepare_adv_training_data(
    name: str,
    summary_keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    assert isinstance(name, str)
    out = get_metrics_adv_training(
        group=name,
        metrics=metrics,
        summary_keys=summary_keys,
    )
    postprocess_data(df=out)

    return out


def extract_seed_from_name(name):
    # name_or_path=AlignmentResearch/robust_llm_pythia-31m_niki-045_pm_random-token-1280_seed-0
    seed_str = name.split(".")[-1]
    seed = int(seed_str)
    assert 0 <= seed <= 10, f"seed was {seed}"
    return seed


def _set_up_paper_plot(fig, ax) -> None:
    fig.set_size_inches(3.5, 2.5)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", length=6, width=1)
    ax.tick_params(axis="both", which="minor", length=4, width=0.5)
    ax.set_ylim((0, 1))


def _draw_plot_adv_training(
    data: pd.DataFrame,
    x: str,
    split_curves_by: str,
    title: str,
    save_as: str,
    save_dir: str,
    use_scatterplot: bool = False,
    metric: str = "adversarial_eval/attack_success_rate",
    xlim: tuple[float, float] = (0, 10),
    legend: bool = False,
):
    data = data.copy()

    plot_fn = sns.scatterplot if use_scatterplot else sns.lineplot

    kwargs = {}
    if not use_scatterplot:
        kwargs["errorbar"] = ("ci", 95)

    # Make it one-indexed since evaluation happens after the training
    data[x] += 1

    ax = plot_fn(
        data=data,
        x=x,
        y=metric,
        hue=split_curves_by,
        palette=get_discrete_palette_for_values(data[split_curves_by]),
        marker=".",
        legend=legend,
        **kwargs,
    )
    if legend:
        ax.get_legend().set_title("Model Size")
    ax.set_xlabel("Adversarial Training Round")
    ax.set_ylabel("Attack Success Rate")

    _set_up_paper_plot(plt.gcf(), ax)

    plt.title(title)
    plt.xlim(xlim)

    "scatter" if use_scatterplot else "line"
    create_path_and_savefig(filename=save_as, subdirectory="adv_training/" + save_dir)
    plt.close()


def draw_plot_adv_training(
    data: pd.DataFrame,
    x: str,
    split_curves_by: str,
    title: str,
    save_as: str,
    save_dir: str = "",
    use_scatterplot: bool = False,
    xlim: tuple[float, float] = (0, 10),
    legend: bool = False,
):
    n_adv_rounds = xlim[1]
    if data["adv_training_round"].max() + 1 < n_adv_rounds:
        raise ValueError(
            f"Found {data['adv_training_round'].max()} adv rounds, "
            f"but requested {n_adv_rounds}"
        )
    data_copy = data.copy()[data["adv_training_round"] <= n_adv_rounds]
    assert isinstance(data_copy, pd.DataFrame)
    _draw_plot_adv_training(
        data_copy,
        x=x,
        split_curves_by=split_curves_by,
        title=title,
        save_as=save_as,
        save_dir=save_dir,
        use_scatterplot=use_scatterplot,
        xlim=xlim,
        legend=legend,
    )


def create_path_and_savefig(filename: str, subdirectory: str):
    repo_path = compute_repo_path()
    directory = Path(f"{repo_path}/plots/{subdirectory}/")
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def draw_plot(
    data: pd.DataFrame,
    title: str,
    save_as: str,
    save_dir: str,
    custom_ys=None,
    custom_xs_and_ys=None,
    scatterplot=False,
):

    # Refine data here as necessary
    # data = data[data.n_its == 10]
    # data = data[data.search_type == "gcg"]
    # data = data[~data['model_name_or_path'].str.contains("-s-", case=False, na=False)]

    print("Found", len(data), "runs to use for the plot")

    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Model size (# parameters)")
    plt.ylabel("Attack Success Rate")
    plt.ylim((0, 1))

    relevant_data = data[["num_params", "adversarial_eval/attack_success_rate"]]

    # Add the custom xs and ys
    # xs are "num_params"
    # ys are values
    if custom_xs_and_ys is not None or custom_ys is not None:
        assert custom_xs_and_ys is None or custom_ys is None
        print("Adding custom data to the plot")
        if custom_ys is not None:
            xs = MODEL_SIZES[-len(custom_ys) :]
        elif custom_xs_and_ys is not None:
            xs, custom_ys = zip(*custom_xs_and_ys)  # type: ignore
        else:
            raise ValueError("should not happen")

        new_data = pd.DataFrame(
            {"num_params": xs, "adversarial_eval/attack_success_rate": custom_ys}
        )
        new_data = new_data[relevant_data.columns]
        relevant_data = pd.concat([relevant_data, new_data], ignore_index=True)

    plot_fn = sns.scatterplot if scatterplot else sns.lineplot

    kwargs = {}
    if not scatterplot:
        kwargs["errorbar"] = ("ci", 95)
    plot_fn(
        data=pd.melt(relevant_data, ["num_params"]),  # type: ignore
        x="num_params",
        y="value",
        hue="variable",
        marker=".",
        legend=False,
        **kwargs,
    )

    _set_up_paper_plot(fig, ax)

    create_path_and_savefig(filename=save_as, subdirectory=save_dir)


def draw_scatter_with_color_from_metric(
    data,
    metric,
    color_metric,
    title="",
    ylim01=True,
    use_colors=True,
    figsize=(3.5, 2.5),
    scatter_size=20,
):
    data = data[["num_params"] + [metric, color_metric]]

    plt.figure(figsize=figsize)

    kwargs = {}
    if use_colors:
        kwargs["c"] = data[color_metric]
        kwargs["cmap"] = "brg"
    plt.scatter(
        data["num_params"],
        data[metric],
        s=scatter_size,
        **kwargs,
    )

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("num_params")
    plt.ylabel(metric)
    if ylim01:
        plt.ylim(0.0, 1.0)

    if use_colors:
        plt.colorbar(label=color_metric)

    plt.show()


def get_discrete_palette_for_values(values):
    values = sorted(set(values))
    return {
        v: sns.color_palette("viridis", n_colors=len(values))[i]
        for i, v in enumerate(values)
    }


def _get_num_params_from_name(name: str) -> int:
    sizes_in_name = [i for i, size in enumerate(MODEL_NAMES) if size in name]
    assert len(sizes_in_name) == 1, f"Found {sizes_in_name} in {name}"
    return MODEL_SIZES[sizes_in_name[0]]


def _get_pretraining_fraction(name: str) -> float:
    # get checkpoint number
    index = name.rfind("-ch-")
    if index == -1:
        checkpoint = FINAL_PYTHIA_CHECKPOINT
    else:
        index += len("-ch-")
        checkpoint = int(name[index:])
    fraction = checkpoint / FINAL_PYTHIA_CHECKPOINT
    return fraction


def postprocess_data(df):
    if "model_size" not in df:
        if "name_or_path" not in df:
            df["name_or_path"] = df["model_name_or_path"]
        df["num_params"] = df["name_or_path"].map(_get_num_params_from_name)
    else:
        df["num_params"] = df["model_size"]

    df["pretraining_fraction"] = df["name_or_path"].map(_get_pretraining_fraction)
