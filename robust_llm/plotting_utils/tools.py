import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.constants import MODEL_PLOTTING_NAMES
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

iter_str = tuple[str, ...] | list[str]
TRANSFORMS: dict[str, Callable] = {
    "log": np.log,
    "logit": lambda x: np.log(x / (1 - x)),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}


def make_finetuned_plots(
    run_names: iter_str,
    title: str,
    save_as: iter_str | str,
    eval_summary_keys: list[str],
    metrics: list[str],
    legend: bool = False,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval/attack_success_rate",
):
    """Make scatter and min/max/median plot for many runs on the same plot."""
    make_finetuned_plot(
        run_names=run_names,
        title=title,
        save_as=list(save_as)
        + [
            "_min_max_median",
        ],
        eval_summary_keys=eval_summary_keys,
        metrics=metrics,
        plot_type="min_max_median",
        legend=legend,
        ylim=ylim,
        ytransform=ytransform,
        y_data_name=y_data_name,
    )
    make_finetuned_plot(
        run_names=run_names,
        title=title,
        save_as=list(save_as)
        + [
            "_scatter",
        ],
        eval_summary_keys=eval_summary_keys,
        metrics=metrics,
        plot_type="scatter",
        ylim=ylim,
        ytransform=ytransform,
        y_data_name=y_data_name,
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
    run_names: iter_str,
    eval_summary_keys: list[str] | tuple[list[str], ...],
    metrics: list[str] | tuple[list[str], ...],
    title: str,
    save_as: iter_str | str,
    custom_ys: list[float] | None = None,
    custom_xs_and_ys: list[tuple[float, float]] | None = None,
    plot_type: str = "scatter",
    legend: bool = False,
    check_seeds: bool = True,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval/attack_success_rate",
):
    """
    Plot model size on the x-axis and attack success rate on the y-axis.

    Args:
        run_names: The names of the runs to pull data from.
        eval_summary_keys: The keys to summarize the data by.
        metrics: The metrics to plot.
        title: The title to give the plot.
        save_as: The name to save the plot as.
        custom_ys: Custom y-values to add to the plot. The x-values are taken
            from MODEL_SIZES, assuming that the custom y-values are for
            the largest models. Mutually exclusive with custom_xs_and_ys.
        custom_xs_and_ys: Custom x-values and y-values to add to the plot.
            Mutually exclusive with custom_ys.
        plot_type:
            The type of plot to make. We currently support
            "scatter" and "min_max_median".
        legend: Whether to include the legend in the plot.
        check_seeds:
            Whether to check that the correct number of seeds are present,
            and that those are the correct actual seed numbers.
        ylim: The y-axis limits.
        ytransform: The transformation to apply to the y-values.
        y_data_name: The name of the data to use for the y-axis.
    """
    if plot_type not in ["scatter", "min_max_median"]:
        raise ValueError(f"Unknown plot type {plot_type}")

    # Use same metrics for all runs if only one set is provided
    elif isinstance(metrics, list):
        metrics = (metrics,) * len(run_names)
    assert len(metrics) == len(run_names)

    # Use same eval_summary_keys for all runs if only one set is provided
    if isinstance(eval_summary_keys, list):
        eval_summary_keys = (eval_summary_keys,) * len(run_names)
    assert len(eval_summary_keys) == len(run_names)

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

    for run in runs:
        postprocess_data(run)

    # Concatenate the runs together
    run = pd.concat(runs, ignore_index=True)

    match plot_type:
        case "min_max_median":
            draw_min_max_median_plot(
                run,
                title=title,
                save_as=save_as,
                legend=legend,
                check_seeds=check_seeds,
                ylim=ylim,
                ytransform=ytransform,
                y_data_name=y_data_name,
            )
        case "scatter":
            draw_scatterplot(
                run,
                title=title,
                save_as=save_as,
                custom_ys=custom_ys,
                custom_xs_and_ys=custom_xs_and_ys,
                check_seeds=check_seeds,
                ylim=ylim,
                ytransform=ytransform,
                y_data_name=y_data_name,
            )
        case _:
            raise ValueError(f"Unknown plot type {plot_type}")


def load_and_plot_adv_training_plots(
    run_names: iter_str | str,
    title: str,
    save_as: iter_str | str,
    summary_keys: list[str] | None = None,
    metrics: list[str] | None = None,
    x_data_name: str = "adv_training_round",
    color_data_name: str = "num_params",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    check_seeds: int | None = None,
    use_cache: bool = True,
    y_data_name: str = "adversarial_eval/attack_success_rate",
):
    """
    Make adversarial training plots for given runs, pulling data from W&B.

    Args:
        run_names: The names of the runs to pull data from.
        title: The title to give the plot (also used for saving).
        save_as: The nested directory structure to save the plot in.
        summary_keys: The keys to summarize the data by.
        metrics: The metrics to plot.
        x_data_name: The name of the data to use for the x-axis.
        color_data_name: The name of the data to use for the different colors.
        xlim: The x-axis limits.
        ylim: The y-axis limits.
        legend: Whether to include the legend in the plot.
        check_seeds:
            Whether to check that the correct number of seeds are present,
            and that those are the correct actual seed numbers.
        use_cache: Whether to use the cache for the data.
        y_data_name: The name of the data to use for the y-axis.
    """
    if isinstance(run_names, str):
        run_names = (run_names,)
    if summary_keys is None:
        summary_keys = SUMMARY_KEYS

    if metrics is None:
        metrics = METRICS

    data = prepare_adv_training_data(
        run_names=run_names,
        summary_keys=summary_keys,
        metrics=metrics,
        use_cache=use_cache,
    )
    draw_plot_adv_training(
        data=data,
        x_data_name=x_data_name,
        color_data_name=color_data_name,
        title=title,
        save_as=save_as,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        check_seeds=check_seeds,
        y_data_name=y_data_name,
    )


def prepare_adv_training_data(
    run_names: iter_str,
    summary_keys: list[str],
    metrics: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    assert all(isinstance(name, str) for name in run_names)
    run_info_list = []

    for run_name in run_names:
        run_info = get_metrics_adv_training(
            group=run_name,
            metrics=metrics,
            summary_keys=summary_keys,
            use_cache=use_cache,
        )
        assert (
            isinstance(run_info, pd.DataFrame) and not run_info.empty
        ), f"Found no data for {run_name}"
        postprocess_data(df=run_info)
        run_info_list.append(run_info)

    run_info_df = pd.concat(run_info_list, ignore_index=True)

    return run_info_df


def extract_seed_from_name(name):
    # name_or_path=AlignmentResearch/robust_llm_pythia-31m_niki-045_pm_random-token-1280_seed-0
    seed_str = name.split(".")[-1]
    seed = int(seed_str)
    assert 0 <= seed <= 10, f"seed was {seed}"
    return seed


def set_up_paper_plot(fig, ax) -> None:
    fig.set_size_inches(3.5, 2.5)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", length=6, width=1)
    ax.tick_params(axis="both", which="minor", length=4, width=0.5)
    ax.set_axisbelow(True)


def _get_n_parameter_updates(data: pd.DataFrame) -> None:
    """
    Calculate the number of parameter updates for each row in the data.

    Estimated as num params * adv train dataset size * training round.
    """

    # TODO(niki): extend this to include the FLOPs from searching
    # for adversarial examples as well---that counts too!
    if "max_augmented_data_size" not in data:
        ma_data_size = 2000
    else:
        ma_data_size = data["max_augmented_data_size"]

    data["n_parameter_updates"] = (
        data["num_params"] * ma_data_size * data["adv_training_round"]
    )


def get_color_palette(data: pd.DataFrame, color_data_name: str) -> dict:
    if color_data_name == "num_params":
        palette_color = "viridis"
    else:
        palette_color = "magma"
    palette = sns.color_palette(
        palette_color, data[color_data_name].nunique()  # type: ignore
    )
    palette_dict = dict(zip(sorted(data[color_data_name].unique()), palette))
    return palette_dict


def get_legend_handles(
    data: pd.DataFrame, color_data_name: str, palette_dict: dict
) -> dict:
    legend_handles = {}
    for name, _ in reversed(sorted(data.groupby(color_data_name))):
        if name not in legend_handles:
            if name in MODEL_SIZES:
                ([model_index],) = np.where(np.array(MODEL_SIZES) == name)
                label = MODEL_PLOTTING_NAMES[model_index]
            else:
                label = name

            legend_handles[name] = plt.Line2D(  # type: ignore
                xdata=[0],
                ydata=[0],
                color=palette_dict[name],
                marker=".",
                linestyle="-",
                label=label,
            )
    return legend_handles


def create_legend(
    color_data_name: str,
    ax: Axes,
    legend_handles: dict,
    loc: str = "best",
    outside: bool = False,
    bbox_to_anchor: tuple[float, float] = (1.05, 1),
) -> None:
    kwargs = {
        "loc": loc,
        "handles": legend_handles.values(),
        "title_fontsize": "xx-small",
        "fontsize": "xx-small",  # Reduce font size
        "labelspacing": 0.2,  # Reduce vertical space between legend entries
        "handletextpad": 0.5,  # Reduce space between handle and text
        "borderpad": 0.3,  # Reduce padding between legend edge and content
        "framealpha": 0.8,  # Add some transparency to the legend box
    }
    if outside:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
    if color_data_name == "adv_training_round":
        ax.legend(
            title="Adversarial Training Round",
            ncols=10,
            **kwargs,
        )
    elif color_data_name == "num_params":
        ax.legend(
            title="# params",
            **kwargs,
        )
    else:
        raise ValueError(f"We don't yet support {color_data_name} in the legend")


def _draw_plot_adv_training(
    data: pd.DataFrame,
    x_data_name: str,
    color_data_name: str,
    title: str,
    save_as: iter_str | str,
    y_data_name: str = "adversarial_eval/attack_success_rate",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    y_transform: str = "logit",
):
    if isinstance(save_as, str):
        save_as = (save_as,)
    data = data.copy()

    # Make it one-indexed since evaluation happens after the training
    data["adv_training_round"] += 1

    fig, ax = plt.subplots()
    if x_data_name == "num_params":
        ax.set_xlabel("Model Size (# parameters)")
        plt.xscale("log")
    elif x_data_name == "adv_training_round":
        ax.set_xlabel("Adversarial Training Round")
        if xlim is not None:
            plt.xlim(xlim)
    elif x_data_name == "n_parameter_updates":
        _get_n_parameter_updates(data)
        ax.set_xlabel("# Parameter Updates")
        plt.xscale("log")
    else:
        raise ValueError(f"We don't yet support {x_data_name} on the x-axis")

    palette_dict = get_color_palette(data, color_data_name)

    plt.title(title)
    y_name_clean = y_data_name.replace("adversarial_eval/", "").replace("metrics/", "")
    y_transf_data_name = f"{y_transform}_{y_name_clean}"
    data[y_transf_data_name] = TRANSFORMS[y_transform](data[y_data_name])
    ax.set_ylabel(y_transf_data_name.replace("_", " ").title())
    if ylim is not None:
        plt.ylim(ylim)
    set_up_paper_plot(fig, ax)

    grouped = (
        data.groupby([color_data_name, x_data_name])[y_transf_data_name]
        .agg(["min", "max", "median"])
        .reset_index()
    )

    for name, group in reversed(sorted(grouped.groupby(color_data_name))):
        plt.fill_between(
            group[x_data_name],
            group["min"],
            group["max"],
            color=palette_dict[name],
            alpha=0.2,
        )
        plt.plot(
            group[x_data_name],
            group["median"],
            marker=".",
            label=name,
            color=palette_dict[name],
            alpha=0.8,
        )

    if legend:
        legend_handles = get_legend_handles(data, color_data_name, palette_dict)
        create_legend(color_data_name, ax, legend_handles)

    if isinstance(save_as, str):
        save_as = (save_as,)
    create_path_and_savefig(
        fig, "adv_training", *save_as, "legend" if legend else "no_legend"
    )


def draw_plot_adv_training(
    data: pd.DataFrame,
    x_data_name: str,
    color_data_name: str,
    title: str,
    save_as: iter_str | str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    check_seeds: int | None = None,
    y_data_name: str = "adversarial_eval/attack_success_rate",
):
    if xlim is not None and data["adv_training_round"].max() + 1 < xlim[1]:
        raise ValueError(
            f"Found {data['adv_training_round'].max()} adv rounds, "
            f"but requested {xlim[1]}"
        )
    data_copy = data.copy()
    if xlim is not None:
        data_copy = data_copy[data_copy["adv_training_round"] < xlim[1]]
    assert isinstance(data_copy, pd.DataFrame)

    if check_seeds is not None:
        _check_correct_num_seeds(data_copy, num_seeds=check_seeds, adversarial=True)

    _draw_plot_adv_training(
        data_copy,
        x_data_name,
        color_data_name=color_data_name,
        title=title,
        save_as=save_as,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        y_data_name=y_data_name,
    )


def create_path_and_savefig(fig, *nested):
    assert isinstance(fig, Figure)
    repo_path = compute_repo_path()
    directory = Path(repo_path) / "plots" / "/".join(nested[:-1])
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{nested[-1]}.pdf"
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    return save_path


def draw_scatterplot(
    data: pd.DataFrame,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    check_seeds=True,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval/attack_success_rate",
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
    ylabel = y_data_name.replace("adversarial_eval/", "").replace("_", " ").title()
    if ytransform is not None:
        ylabel += f" ({ytransform})"
    plt.ylabel(ylabel)
    if ytransform is None:
        data["y_value"] = data[y_data_name]
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
    if ylim is not None:
        plt.ylim(ylim)

    relevant_data = data[["num_params", "y_value"]]

    if check_seeds:
        _check_correct_num_seeds(relevant_data, num_seeds=3, adversarial=False)

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

        new_data = pd.DataFrame({"num_params": xs, "y_value": custom_ys})
        new_data = new_data[relevant_data.columns]
        relevant_data = pd.concat([relevant_data, new_data], ignore_index=True)

    if check_seeds:
        _check_correct_num_seeds(relevant_data, num_seeds=3, adversarial=False)

    sns.scatterplot(
        data=pd.melt(relevant_data, ["num_params"]),  # type: ignore
        x="num_params",
        y="value",
        hue="variable",
        marker="o",
        legend=False,
        alpha=0.5,
    )

    set_up_paper_plot(fig, ax)

    if isinstance(save_as, str):
        save_as = (save_as,)
    create_path_and_savefig(fig, *save_as)


def _maybe_get_custom_xs_and_maybe_ys(relevant_data, custom_ys, custom_xs_and_ys):
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

    return relevant_data


def _get_seed_from_name(name: str) -> int:
    # name_or_path=AlignmentResearch/robust_llm_pythia-31m_niki-045_pm_random-token-1280_seed-0
    # or 'AlignmentResearch/robust_llm_clf_imdb_pythia-14m_s-0_adv_tr_rt_t-0'
    if "seed" in name:
        seed_str = name.split("_")[-1]
        assert "seed" in seed_str
        seed_num = int(seed_str.split("-")[-1])
    else:
        seed_num = int(name.split("_t-")[-1])
    return seed_num


def _get_seeds(data: pd.DataFrame) -> list[int]:
    # If the seed is already in the dataframe, just use that
    if "seed" in data.columns:
        return data["seed"].tolist()
    if "training_seed" in data.columns:
        return data["training_seed"].tolist()
    # Otherwise, get it from the name
    if "training_force_name_to_save" in data.columns:
        return data["training_force_name_to_save"].apply(_get_seed_from_name).tolist()
    return data["model_name_or_path"].apply(_get_seed_from_name).tolist()


def _check_correct_seed_numbers(data: pd.DataFrame) -> None:
    # Check that the seeds are correct
    seeds = sorted(_get_seeds(data))
    seeds_should_be = [i for _, i in enumerate(seeds)]
    if not seeds == seeds_should_be:
        print("Found seeds", seeds)
        print("Expected seeds", seeds_should_be)
        raise ValueError("Incorrect seeds")


def _check_correct_num_seeds(relevant_data, num_seeds: int, adversarial: bool) -> None:
    """
    Check that the seeds are correct.

    Make sure that for a given model size and adversarial training
    round, there are exactly `num_seeds` examples, and that they
    are the correct seed numbers.
    """

    if adversarial:
        for model_size in relevant_data["num_params"].unique():
            for adv_round in relevant_data["adv_training_round"].unique():
                matching_datapoints = relevant_data[
                    (relevant_data["num_params"] == model_size)
                    & (relevant_data["adv_training_round"] == adv_round)
                ]
                num_that_size = len(matching_datapoints)

                # Check correct number of seeds
                if num_that_size != num_seeds:
                    print(
                        f"Found {num_that_size} data points for model size {model_size}"
                        f" and adv round {adv_round}"
                    )
                    print(f"Expected {num_seeds} data points")
                    print("Datapoints found: ")
                    for _, row in matching_datapoints.iterrows():
                        if "force_name_to_save" in row:
                            print(f"model {row['training_force_name_to_save']}")
                        else:
                            print(f"model {row['model_name_or_path']}")
                        if "run_name" in row:
                            print(f"run name {row['run_name']}")
                    raise ValueError("Incorrect number of data points")

                # Check correct seeds
                _check_correct_seed_numbers(matching_datapoints)

    else:
        for model_size in relevant_data["num_params"].unique():
            matching_datapoints = relevant_data[
                relevant_data["num_params"] == model_size
            ]
            num_that_size = len(matching_datapoints)
            if num_that_size != num_seeds:
                print(f"Found {num_that_size} data points for model size {model_size}")
                print(f"Expected {num_seeds} data points")
                print("Datapoints found: ")
                for _, row in matching_datapoints.iterrows():
                    print(row["model_name_or_path"])
                raise ValueError("Incorrect number of data points")


def draw_min_max_median_plot(
    data: pd.DataFrame,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    legend: bool = False,
    check_seeds: bool = True,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval/attack_success_rate",
):
    print("Found", len(data), "runs to use for the plot")

    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Model size (# parameters)")
    ylabel = y_data_name.replace("adversarial_eval/", "").replace("_", " ").title()
    if ytransform is not None:
        ylabel += f" ({ytransform})"
    plt.ylabel(ylabel)
    if ytransform is None:
        data["y_value"] = data[y_data_name]
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
    if ylim is not None:
        plt.ylim(ylim)

    relevant_data = data[["num_params", "y_value", "model_name_or_path"]]

    relevant_data = _maybe_get_custom_xs_and_maybe_ys(
        relevant_data, custom_ys, custom_xs_and_ys
    )

    if check_seeds:
        _check_correct_num_seeds(relevant_data, num_seeds=3, adversarial=False)

    # Group by num_params and calculate min, max, and median
    grouped = (
        relevant_data.groupby("num_params")
        .agg({"y_value": ["min", "max", "median"]})
        .reset_index()
    )
    grouped.columns = ["num_params", "y_min", "y_max", "y_median"]

    sns_blue = sns.color_palette()[0]

    # Plot the band between the min and max values
    plt.fill_between(
        grouped["num_params"],
        grouped["y_min"],
        grouped["y_max"],
        alpha=0.2,
        label="Min-Max Range",
        color=sns_blue,
    )
    # Plot the median values
    sns.lineplot(
        x=grouped["num_params"],
        y=grouped["y_median"],
        label="Median",
        marker="o",
        color=sns_blue,
        alpha=0.5,
        zorder=5,
    )

    set_up_paper_plot(fig, ax)

    # Turn off the legend if we don't want it
    if not legend:
        ax.get_legend().remove()

    if isinstance(save_as, str):
        save_as = (save_as,)

    create_path_and_savefig(fig, *save_as, "legend" if legend else "no_legend")


def draw_min_max_median_plot_by_round(
    data: pd.DataFrame,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    legend: bool = True,
    check_seeds: bool = True,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval/attack_success_rate",
    rounds: list[int] | None = None,
):
    print("Found", len(data), "runs to use for the plot")

    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Model size (# parameters)")
    ylabel = y_data_name.replace("adversarial_eval/", "").replace("_", " ").title()
    if ytransform is not None:
        ylabel += f" ({ytransform})"
    plt.ylabel(ylabel)
    if ytransform is None:
        data["y_value"] = data[y_data_name]
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
    if ylim is not None:
        plt.ylim(ylim)

    relevant_data = data[
        [
            "num_params",
            "y_value",
            "training_force_name_to_save",
            "adv_training_round",
        ]
    ]
    if check_seeds:
        _check_correct_num_seeds(relevant_data, num_seeds=3, adversarial=True)

    relevant_data = _maybe_get_custom_xs_and_maybe_ys(
        relevant_data, custom_ys, custom_xs_and_ys
    )

    # Group by num_params and adv_training_round then calculate min, max, and median
    grouped = (
        relevant_data.groupby(["num_params", "adv_training_round"])
        .agg({"y_value": ["min", "max", "median"]})
        .reset_index()
    )
    grouped.columns = ["num_params", "adv_training_round", "y_min", "y_max", "y_median"]

    if rounds is None:
        rounds = sorted(grouped["adv_training_round"].unique())

    # Color palette for different rounds
    colors = sns.color_palette("husl", n_colors=len(rounds))
    color_map = dict(zip(rounds, colors))

    for i, round_val in enumerate(rounds):
        round_data = grouped[grouped["adv_training_round"] == round_val]

        # Plot the band between the min and max values
        plt.fill_between(
            round_data["num_params"],
            round_data["y_min"],
            round_data["y_max"],
            alpha=0.2,
            label=f"Round {round_val} Min-Max",
            color=colors[i],
        )

        # Plot the median values
        sns.lineplot(
            x=round_data["num_params"],
            y=round_data["y_median"],
            label=f"Round {round_val} Median",
            marker="o",
            color=colors[i],
            alpha=0.7,
            zorder=5,
        )

    set_up_paper_plot(fig, ax)

    # Adjust legend
    if legend:
        # Create legend elements
        round_legend = [
            Line2D([0], [0], color=color_map[r], lw=2, label=f"Round {r}")
            for r in rounds
        ]
        style_legend = [
            Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="",
                markersize=8,
                label="Median",
            ),
            plt.Rectangle(  # type: ignore
                (0, 0), 1, 1, fc="gray", alpha=0.2, label="Min-Max Range"
            ),
        ]

        # Combine all legend elements
        all_legend_elements = round_legend + style_legend

        # Create a single, compact legend
        plt.legend(
            handles=all_legend_elements,
            loc="upper right",
            ncol=1,
            fontsize="xx-small",
        )
    else:
        ax.get_legend().remove()

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    if isinstance(save_as, str):
        save_as = (save_as,)

    create_path_and_savefig(fig, *save_as, "legend" if legend else "no_legend")


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


def _update_model_sizes_as_necessary(df: pd.DataFrame) -> None:
    # Sometimes, the model size is not a size that we recognize.
    # In those cases, just take the model size directly from
    # the model name.
    for i, row in df.iterrows():
        if row["num_params"] not in MODEL_SIZES:
            size_from_name = _get_num_params_from_name(
                row["model_name_or_path"]  # type: ignore
            )
            print(
                f"Couldn't find model size {row['num_params']}, "
                f"so taking from name ({size_from_name})"
            )
            df.at[i, "num_params"] = size_from_name


def postprocess_data(df):
    df.rename(columns={"name_or_path": "model_name_or_path"}, inplace=True)
    if "model_size" not in df:
        df["num_params"] = df["model_name_or_path"].map(_get_num_params_from_name)
    else:
        df["num_params"] = df["model_size"]

    _update_model_sizes_as_necessary(df)

    assert "model_name_or_path" in df

    df["pretraining_fraction"] = df["model_name_or_path"].map(_get_pretraining_fraction)


def prepare_asr_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    asr_columns = [col for col in data.columns if col.startswith("metrics/asr@")]
    other_columns = [
        col for col in data.columns if "@" not in col and "adversarial_eval/" not in col
    ]
    melted_df = pd.melt(
        data,
        id_vars=other_columns,
        value_vars=asr_columns,
        var_name="iteration",
        value_name="asr",
    )
    melted_df["iteration"] = melted_df["iteration"].str.extract(r"(\d+)").astype(int)
    return melted_df


def prepare_ifs_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    asr_columns = [col for col in data.columns if col.startswith("metrics/ifs@")]
    other_columns = [
        col for col in data.columns if "@" not in col and "adversarial_eval/" not in col
    ]
    melted_df = pd.melt(
        data,
        id_vars=other_columns,
        value_vars=asr_columns,
        var_name="asr",
        value_name="ifs",
    )
    # e.g. metrics/ifs@0.1
    melted_df["asr"] = melted_df["asr"].str.extract(r"metrics/ifs@(.*)").astype(float)
    melted_df["decile"] = (melted_df["asr"] * 10).astype(int)
    return melted_df


def plot_attack_scaling(
    orig_df: pd.DataFrame,
    attack: str,
    dataset: str,
    n_models: int,
    n_seeds: int,
    n_iterations: int,
    datapoints: int | None = None,
    check_seeds: bool = True,
    x: str = "log_iteration_x_params",
    y: str = "logit_asr",
) -> None:
    df = orig_df.copy()
    if "model_size" in df and "model_idx" not in df:
        df["model_idx"] = df.model_size.rank(method="dense").astype(int) - 1
    if "seed_idx" not in df and "model_name_or_path" in df:
        df["seed_idx"] = df.model_name_or_path.apply(_get_seed_from_name)
    if "num_params" not in df:
        df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])
    assert df.model_idx.between(0, n_models - 1).all()
    assert df.seed_idx.between(0, n_seeds - 1).all()
    assert df.iteration.between(0, n_iterations).all()
    if check_seeds:
        assert len(df) == n_models * n_seeds * (n_iterations + 1)
    if datapoints is not None and df.iteration.nunique() > datapoints:
        # If datapoints=10, filter out all but empirical deciles
        df = df.loc[
            df.iteration.isin(df.iteration.quantile(np.linspace(0, 1, datapoints)))
        ]
    df["iteration_x_params"] = df.iteration * df.num_params
    df["log_iteration_x_params"] = np.log(df.iteration_x_params)
    assert df.asr.between(0, 1).all()
    df["sigmoid_asr"] = 1 / (1 + np.exp(-df.asr))
    df["logit_asr"] = np.log(df.asr / (1 - df.asr))
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)
    color_data_name = "num_params"
    palette = get_color_palette(df, color_data_name)
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=color_data_name,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    fig.suptitle(f"{attack}/{dataset}".upper())
    create_path_and_savefig(fig, "asr", attack, dataset, "no_legend")
    legend_handles = get_legend_handles(df, color_data_name, palette)
    create_legend(color_data_name, ax, legend_handles, outside=False)
    save_path = create_path_and_savefig(fig, "asr", attack, dataset, "legend")
    df.to_csv(str(save_path).replace("legend.pdf", "data.csv"), index=False)


def plot_ifs(
    orig_df: pd.DataFrame,
    attack: str,
    dataset: str,
    n_models: int,
    n_seeds: int,
    check_seeds: bool = True,
    x: str = "log_asr",
    y: str = "log_ifs",
) -> None:
    df = orig_df.copy()
    if "model_size" in df and "model_idx" not in df:
        df["model_idx"] = df.model_size.rank(method="dense").astype(int) - 1
    if "seed_idx" not in df and "model_name_or_path" in df:
        df["seed_idx"] = df.model_name_or_path.apply(_get_seed_from_name)
    if "num_params" not in df:
        df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])
    assert df.model_idx.between(0, n_models - 1).all()
    assert df.seed_idx.between(0, n_seeds - 1).all()
    assert df.asr.between(0, 1).all()
    assert df.decile.between(0, 10).all()
    if check_seeds:
        assert len(df) == n_models * n_seeds * 11
    if "asr" not in df:
        df["asr"] = df.decile.mul(0.1)
    df["log_ifs"] = np.log(df.ifs)
    df["log_asr"] = np.log(df.asr)
    df["logit_asr"] = np.log(df.asr / (1 - df.asr))
    df.sort_values("model_idx", inplace=True)

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)
    color_data_name = "num_params"
    palette = get_color_palette(df, color_data_name)
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=color_data_name,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_ylabel(
        y.replace("_", " ").replace("ifs", "Iterations required to reach ASR").title()
    )
    ax.set_xlabel(x.replace("_", " ").replace("asr", "ASR").title())
    fig.suptitle(f"{attack}/{dataset}".upper())
    create_path_and_savefig(fig, "ifs", attack, dataset, "no_legend")
    legend_handles = get_legend_handles(df, color_data_name, palette)
    create_legend(color_data_name, ax, legend_handles, outside=False)
    save_path = create_path_and_savefig(fig, "ifs", attack, dataset, "legend")
    df.to_csv(str(save_path).replace("legend.pdf", "data.csv"), index=False)


def load_and_plot_asr_and_ifs(
    run_names: tuple[str, ...],
    summary_keys: list[str],
    metrics: list[str],
    attack: str,
    dataset: str,
    n_models: int,
    n_seeds: int,
    n_iterations: int,
    check_seeds: bool = True,
    asr_x: str = "log_iteration_x_params",
    asr_y: str = "logit_asr",
    ifs_x: str = "log_asr",
    ifs_y: str = "log_ifs",
    datapoints: int | None = None,
    use_cache: bool = True,
):
    adv_data = prepare_adv_training_data(
        run_names=run_names,
        summary_keys=summary_keys,
        metrics=metrics,
        use_cache=use_cache,
    )
    asr_data = prepare_asr_data(adv_data)
    ifs_data = prepare_ifs_data(adv_data)
    plot_attack_scaling(
        asr_data,
        attack,
        dataset,
        n_models,
        n_seeds,
        n_iterations,
        datapoints,
        check_seeds,
        asr_x,
        asr_y,
    )
    plot_ifs(ifs_data, attack, dataset, n_models, n_seeds, check_seeds, ifs_x, ifs_y)
