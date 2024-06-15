from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.constants import (
    FINAL_PYTHIA_CHECKPOINT,
    METRICS,
    MODEL_SIZE_DICT,
    PROJECT_NAME,
    SUMMARY_KEYS,
)

wandb.login()
API = wandb.Api(timeout=60)


def load_and_plot_adv_training_plots(
    name: str,
    title: str,
    summary_keys: list[str] = SUMMARY_KEYS,
    metrics: list[str] = METRICS,
    use_size_from_name: bool = False,
) -> None:
    """
    Make adversarial training plots for a given run, pulling data from W&B.

    Args:
        name: The name of the run to pull data from.
        title: The title to give the plot (also used for saving).
        summary_keys: The keys to summarize the data by.
        metrics: The metrics to plot.
        use_size_from_name: Whether to use the model size from
            the run name (legacy) or to take it from W&B.
    """
    data = prepare_adv_training_data(
        name,
        summary_keys=summary_keys,
        metrics=metrics,
        use_size_from_name=use_size_from_name,
    )
    plot_line_and_scatter_adv_training(
        data, title=title, use_size_from_name=use_size_from_name
    )


def prepare_adv_training_data(
    name: str,
    summary_keys: list[str],
    metrics: list[str],
    use_size_from_name: bool = False,
) -> pd.DataFrame:
    assert isinstance(name, str)
    out = get_metrics_adv_training(
        group=name,
        metrics=metrics,
        summary_keys=summary_keys,
    )
    postprocess_data(df=out, use_size_from_name=use_size_from_name)

    return out


def plot_line_and_scatter_adv_training(
    run: pd.DataFrame, title: str = "", use_size_from_name=False
):
    draw_plot_adv_training(
        run,
        x="adv_training_round",
        split_curves_by="num_params",
        logscale=False,
        title=title,
        use_size_from_name=use_size_from_name,
    )
    draw_plot_adv_training(
        run,
        x="adv_training_round",
        split_curves_by="num_params",
        logscale=False,
        use_scatterplot=True,
        title=title,
        use_size_from_name=use_size_from_name,
    )


def extract_seed_from_name(name):
    # name_or_path=AlignmentResearch/robust_llm_pythia-31m_niki-045_pm_random-token-1280_seed-0
    seed_str = name.split(".")[-1]
    seed = int(seed_str)
    assert 0 <= seed <= 10, f"seed was {seed}"
    return seed


def _draw_plot_adv_training(
    data,
    x,
    split_curves_by: str,
    title="",
    use_scatterplot=False,
    errorbar=("ci", 95),
    metric="adversarial_eval/attack_success_rate",
    ylim=(0, 1),
    logscale=False,
    y_logscale=False,
    xlim=None,
    ytransf=None,
    use_size_from_name=False,
    figsize: tuple[float, float] = (7.5, 5),
):
    plt.figure(figsize=figsize)

    data = data.copy()

    plot_fn = sns.scatterplot if use_scatterplot else sns.lineplot

    kwargs = {}
    if not use_scatterplot:
        kwargs["errorbar"] = errorbar

    if ytransf is not None:
        data[metric] = data[metric].apply(ytransf)

    plot_fn(
        data=data,
        x=x,
        y=metric,
        hue=split_curves_by,
        palette=get_discrete_palette_for_values(data[split_curves_by]),
        marker="o",
        **kwargs,
    )

    if logscale:
        plt.xscale("log")

    if y_logscale:
        plt.yscale("log")

    if ylim is not None:
        plt.ylim(ylim)

    if xlim is not None:
        plt.xlim(xlim)

    add_to_path = ""
    if title:
        split_title = title.split("/")
        add_to_path = "/".join(split_title[:-1]) + "/"
        if "/" in title:
            title = split_title[-1]
        plt.title(title + " (model sizes from name)" if use_size_from_name else title)

    repo_path = compute_repo_path()
    d = repo_path + "/plots/" + add_to_path
    t = "scatter" if use_scatterplot else "line"
    plt.savefig(d + f"{title} {t}.png" if title else "plot {t}.png")
    plt.close()


def draw_plot_adv_training(
    data: pd.DataFrame,
    x: str,
    split_curves_by: str,
    logscale: bool,
    y_logscale: bool = False,
    title: str = "",
    use_scatterplot: bool = False,
    ylim: Optional[tuple[float, float]] = None,
    xlim=None,
    ytransf=None,
    use_size_from_name=False,
):
    for n_adv_rounds in [10, 20, 30]:
        if data["adv_training_round"].max() + 1 >= n_adv_rounds:
            data_copy = data.copy()[data["adv_training_round"] <= n_adv_rounds]
            _draw_plot_adv_training(
                data_copy,
                x=x,
                split_curves_by=split_curves_by,
                logscale=logscale,
                y_logscale=y_logscale,
                title=title + f", {n_adv_rounds} adv rounds train",
                use_scatterplot=use_scatterplot,
                ylim=ylim,
                xlim=xlim,
                ytransf=ytransf,
                use_size_from_name=use_size_from_name,
            )


def draw_plot(
    data,
    title="",
    use_scatterplot=False,
    errorbar=("ci", 95),
    metrics=None,
    ylim01=True,
    figsize=(7.5, 5),
):
    if metrics is None:
        metrics = METRICS

    plt.figure(figsize=figsize)

    data = data[["num_params"] + metrics]

    plot_fn = sns.scatterplot if use_scatterplot else sns.lineplot

    kwargs = {}
    if not use_scatterplot:
        kwargs["errorbar"] = errorbar
    plot = plot_fn(
        data=pd.melt(data, ["num_params"]),
        x="num_params",
        y="value",
        hue="variable",
        marker="o",
        **kwargs,
    )

    plot.set(xscale="log")
    plot.set_title(title)
    if ylim01:
        plt.ylim(0.0, 1.0)

    plt.show()


def draw_scatter_with_color_from_metric(
    data,
    metric,
    color_metric,
    title="",
    ylim01=True,
    use_colors=True,
    figsize=(10, 5),
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


def _extract_adv_round(revision):
    # adv-training-round-0
    return int(revision.split("-")[-1])


def get_metrics_adv_training(
    group,
    metrics,
    summary_keys,
    filters=None,
    check_num_runs=None,
    check_num_data_per_run=None,
):
    if filters is None:
        filters = {}
    filters = deepcopy(filters)
    filters["group"] = group
    # filters["state"] = "finished"

    runs = API.runs(path=PROJECT_NAME, filters=filters)
    if check_num_runs is not None:
        if type(check_num_runs) is int:
            assert len(runs) == check_num_runs
        elif type(check_num_runs) is list:
            assert check_num_runs[0] <= len(runs) <= check_num_runs[1]
        else:
            assert False, "bad check_num_runs!"

    res = []
    for run in runs:
        history = run.history(keys=metrics)

        if check_num_data_per_run is not None:
            assert len(history) == check_num_data_per_run

        if len(history) == 0:
            continue

        if history is None:
            continue

        for key in summary_keys:
            history[key.split(".")[-1]] = _get_value_iterative(run.summary, key)
        history["run_id"] = run.id

        # Hack: create 'round' column based on increasing steps.
        history = history.sort_values(by="_step")

        if "revision" in history:
            history["adv_training_round"] = [
                _extract_adv_round(name) for name in history["revision"]
            ]
        else:
            history["adv_training_round"] = np.arange(len(history))

        res.append(history)

    res = pd.concat(res, ignore_index=True)

    return res


def draw_for_multi_seed_exp(
    data, dataset, search_type, n_its, use_colors=True, only_scatter=True
):
    data = data.copy()
    data = data[
        (data.dataset_type == dataset)
        & (data.search_type == search_type)
        & (data.n_its == n_its)
    ]
    title = f"{dataset}, {search_type}, {n_its=}"

    if not only_scatter:
        print("Shaded areas are 95% CIs")
        draw_plot(data, title=title, metrics=METRICS)

    draw_scatter_with_color_from_metric(
        data,
        metric="adversarial_eval/attack_success_rate",
        color_metric="pretraining_fraction",
        title=title,
        use_colors=use_colors,
    )


def get_discrete_palette_for_values(values):
    values = sorted(set(values))
    return {
        v: sns.color_palette("viridis", n_colors=len(values))[i]
        for i, v in enumerate(values)
    }


def _get_value_iterative(d, key):
    for k in key.split("."):
        d = d[k]
    return d


def _get_num_params_from_name(name: str) -> int:
    sizes_in_name = [size for size in MODEL_SIZE_DICT.keys() if size in name]
    assert len(sizes_in_name) == 1, f"Found {sizes_in_name} in {name}"
    return MODEL_SIZE_DICT[sizes_in_name[0]]


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


def postprocess_data(df, use_size_from_name=False):
    if use_size_from_name:
        df["num_params"] = df["name_or_path"].map(_get_num_params_from_name)
    else:
        df["num_params"] = df["model_size"]

    df["pretraining_fraction"] = df["name_or_path"].map(_get_pretraining_fraction)

    df["pre_post_accuracy_gap"] = (
        df["adversarial_eval/pre_attack_accuracy"]
        - df["adversarial_eval/post_attack_accuracy_including_original_mistakes"]
    )
