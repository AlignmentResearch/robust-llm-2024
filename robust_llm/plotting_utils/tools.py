import json
import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from robust_llm.file_utils import compute_repo_path
from robust_llm.metrics.asr_per_iteration import interpolated_iteration_for_asr
from robust_llm.plotting_utils.constants import (
    AXIS_LABELS,
    DEFAULT_SMOOTHING,
    LOG_SCALE_VARIABLES,
    MODEL_PLOTTING_NAMES,
    get_fudge_factor,
    get_offense_defense_ylabel_title,
)
from robust_llm.plotting_utils.experiments.pretrain_compute_per_model import (
    ESTIMATED_PRETRAIN_COMPUTE,
)
from robust_llm.plotting_utils.style import (
    name_to_attack,
    name_to_dataset,
    set_plot_style,
)
from robust_llm.plotting_utils.utils import (
    add_model_idx_inplace,
    drop_duplicates,
    merge_adv_and_train_data,
)
from robust_llm.wandb_utils.constants import (
    FINAL_PYTHIA_CHECKPOINT,
    METRICS,
    MODEL_NAMES,
    MODEL_SIZES,
    SUMMARY_KEYS,
)
from robust_llm.wandb_utils.wandb_api_tools import (
    get_cache_root,
    get_group_enriched_history,
)

iter_str = tuple[str, ...] | list[str]
TRANSFORMS: dict[str, Callable] = {
    "log": np.log10,
    "logit": lambda x: np.log10(x / (1 - x)),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "none": lambda x: x,
}


def get_csv_root():
    return get_cache_root() / "plot_csvs"


def _get_csv_path(*save_as) -> Path:
    if isinstance(save_as, str):
        save_as = (save_as,)
    path = get_csv_root()
    for part in save_as:
        path = path / part
    if not path.exists():
        path.mkdir(parents=True)
    path = path / "data.csv"
    return path


def _get_metadata_path(*save_as) -> Path:
    csv_path = _get_csv_path(*save_as)
    return csv_path.with_suffix(".metadata.json")


def _read_csv_from_path_fragments(*path_fragments):
    path = _get_csv_path(*path_fragments)
    return pd.read_csv(path)


def _read_metadata_from_path_fragments(*path_fragments):
    path = _get_metadata_path(*path_fragments)
    if not path.exists():
        print(f"Metadata file {path} does not exist.")
        return None
    return PlotMetadata.from_json(path)


def read_csv_and_metadata(*path_fragments):
    return _read_csv_from_path_fragments(
        *path_fragments
    ), _read_metadata_from_path_fragments(*path_fragments)


def _export_csv(data: pd.DataFrame, *save_as: str) -> None:
    path = _get_csv_path(*save_as)
    data.to_csv(path, index=False)


class PlotMetadata:
    def __init__(
        self, commit: str, timestamp: str, function_name: str, arguments: dict
    ):
        self.commit = commit
        self.timestamp = timestamp
        self.function_name = function_name
        self.arguments = arguments

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "git_commit": self.commit,
                    "timestamp": self.timestamp,
                    "function_name": self.function_name,
                    "arguments": self.arguments,
                },
                f,
            )

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            commit=data["git_commit"],
            timestamp=data["timestamp"],
            function_name=data["function_name"],
            arguments=data["arguments"],
        )

    @classmethod
    def from_function_and_args(cls, function_name: str, arguments: dict):
        return cls(
            commit=os.popen("git rev-parse HEAD").read().strip(),
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            function_name=function_name,
            arguments=arguments,
        )

    @classmethod
    def export(cls, function_name: str, arguments: dict, *path_fragments: str):
        md = cls.from_function_and_args(function_name, arguments)
        path = _get_metadata_path(*path_fragments)
        md.to_json(path)

    def to_latex(
        self,
        fig_path: Path,
        width: str | None = None,
        caption: str | None = None,
        trim: tuple[float, float, float, float] | None = None,
        clip: bool | None = None,
    ) -> None:
        latex_path = fig_path.with_name("fig.tex")
        # Remove fragments from the fig path before `plots`
        relative_path = Path(
            *[
                part
                for i, part in enumerate(fig_path.parts)
                if i > fig_path.parts.index("plots")
            ]
        )
        fig_name = "-".join(relative_path.parts)
        if caption is None:
            caption = "/".join(relative_path.parts[:-1])
            caption = caption.replace("_", " ").replace("-", " ").title()
        graphics_options = []
        if width is not None:
            graphics_options.append(f"width={width}")
        if trim is not None:
            graphics_options.append(f"trim={' '.join(f'{t:.1f}mm' for t in trim)}")
        if clip:
            graphics_options.append("clip")
        graphics_brackets = (
            "[" + ", ".join(graphics_options) + "]" if graphics_options else ""
        )
        with open(latex_path, "w") as f:
            f.write(
                f"% To regenerate the following figure, run\n"
                f"% git checkout {self.commit}\n"
                f"% {self.function_name}(**{self.arguments})\n"
                f"\\includegraphics{graphics_brackets}{{figs/{relative_path}}}\n"
                f"\\caption{{{caption}}}\n"
                f"\\label{{fig:{fig_name}}}"
            )


def export_csv_and_metadata(
    data: pd.DataFrame,
    function_name: str,
    arguments: dict,
    *save_as: str,
) -> None:
    _export_csv(data, *save_as)
    PlotMetadata.export(function_name, arguments, *save_as)


def apply_laplace_smoothing(
    data: pd.DataFrame, y_data_name: str, smoothing: int, size_name: str | None = None
) -> None:
    """Apply Laplace smoothing to the y-values in the dataframe in-place.

    See https://en.wikipedia.org/wiki/Additive_smoothing for more details.

    Args:
        data: The dataframe to apply smoothing to.
        y_data_name: The name of the column in the dataframe that contains the observed
            rates, e.g. `adversarial_eval_attack_success_rate`.
        smoothing: The amount of Laplace smoothing to apply. A value of 1 here
            corresponds to Laplace's rule of succession.
        size_name: The name of the column in the dataframe that contains the size of
            the dataset. Standard values are `n_val` or `dataset_n_val`.
    """
    if size_name is None:
        size_name = "n_val" if "n_val" in data.columns else "dataset_n_val"
    assert size_name in data.columns, (
        f"Expected {size_name} to be in the columns of the data, "
        f"but found {data.columns}"
    )
    assert data[y_data_name].between(0, 1).all(), (
        f"Expected {y_data_name} to be between 0 and 1, but found "
        f"{data[y_data_name].min()} and {data[y_data_name].max()}"
    )
    # Assume binary classification setting
    num_classes = 2
    data[y_data_name] = (data[y_data_name] * data[size_name] + smoothing) / (
        data[size_name] + num_classes * smoothing
    )


def set_yticks_for_logit(ax: Axes) -> None:
    # Get the current y-axis limits based on the data
    y_min, y_max = ax.get_ylim()

    p_min = 1 / (1 + np.power(10, -y_min))
    p_max = 1 / (1 + np.power(10, -y_max))

    # Define the percentage values for major ticks
    major_percentages = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    major_percentages = [p for p in major_percentages if p_min <= p <= p_max]

    # Define percentage values for minor ticks
    minor_percentages = []
    for p1, p2 in zip(major_percentages[:-1], major_percentages[1:]):
        gap = p2 - p1
        minor_percentages.extend([p1 + 0.25 * gap, p1 + 0.5 * gap, p1 + 0.75 * gap])

    # Convert percentages to logit values
    major_logit_values = [TRANSFORMS["logit"](p) for p in major_percentages]
    minor_logit_values = [TRANSFORMS["logit"](p) for p in minor_percentages]

    # Set the major tick locations and labels
    ax.set_yticks(major_logit_values)
    ax.set_yticklabels([f"{p:.2f}" for p in major_percentages])

    # Set the minor tick locations
    ax.set_yticks(minor_logit_values, minor=True)

    ax.set_ylim(y_min, y_max)


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


def make_finetuned_data(
    group_names: iter_str,
    eval_summary_keys: list[str] | tuple[list[str], ...],
    metrics: list[str] | tuple[list[str], ...],
    save_as: iter_str | str,
    use_group_cache: bool = True,
) -> pd.DataFrame:
    """
    Create CSV file for use in plotting data for finetuned models.

    Args:
        group_names: The names of the wandb groups to pull data from.
        eval_summary_keys: The keys to summarize the data by.
        metrics: The metrics to plot.
        save_as: The name to save the plot as.
        use_group_cache: Whether to use the group cache when pulling data from W&B.
    """

    # Use same metrics for all runs if only one set is provided
    if isinstance(metrics, list):
        metrics = (metrics,) * len(group_names)
    assert len(metrics) == len(group_names)

    # Use same eval_summary_keys for all runs if only one set is provided
    if isinstance(eval_summary_keys, list):
        eval_summary_keys = (eval_summary_keys,) * len(group_names)
    assert len(eval_summary_keys) == len(group_names)

    print("Plotting with data from ", group_names)

    runs = []
    for group, metric_list, summary_key_list in zip(
        group_names, metrics, eval_summary_keys
    ):
        run = get_group_enriched_history(
            group_name=group,
            metrics=metric_list,
            summary_keys=summary_key_list,
            use_group_cache=use_group_cache,
        )
        postprocess_data(run)
        run_asr = get_unstacked_cached_asr_data(group)
        run = run.merge(
            run_asr,
            on=["model_idx", "seed_idx", "adv_training_round"],
            how="left",
            validate="1:1",
            suffixes=("", "_asr"),
        )
        runs.append(run)

    # Concatenate the runs together
    run = pd.concat(runs, ignore_index=True)
    run.columns = run.columns.str.replace("/", "_").str.replace("@", "_at_")

    export_csv_and_metadata(
        run,
        "make_finetuned_data",
        {
            "group_names": group_names,
            "eval_summary_keys": eval_summary_keys,
            "metrics": metrics,
            "save_as": save_as,
            "use_group_cache": use_group_cache,
        },
        *save_as,
    )

    return run


def load_flops_data(
    group_names: iter_str | str,
    use_group_cache: bool = True,
):
    if isinstance(group_names, str):
        group_names = (group_names,)
    data = _prepare_adv_training_data(
        group_names=group_names,
        summary_keys=[
            "experiment_yaml.run_name",
            "experiment_yaml.model.name_or_path",
            "experiment_yaml.training.force_name_to_save",
            "experiment_yaml.training.save_name",
            "experiment_yaml.training.adversarial.skip_first_training_round",
            "experiment_yaml.training.seed",
        ],
        metrics=[
            "train/total_flops",
            "adv_training_round",
        ],
        use_group_cache=use_group_cache,
    )
    data["model_key"] = data.training_force_name_to_save.where(
        data.training_force_name_to_save.notnull(), data.training_save_name
    )
    data["mean_train_total_flops"] = data.groupby(["model_idx", "adv_training_round"])[
        "train_total_flops"
    ].transform("mean")
    data["pretrain_compute"] = data.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)
    data["defense_flops_fraction_pretrain"] = (
        data.mean_train_total_flops / data.pretrain_compute
    )
    return data


def assert_flops_data_not_missing(data: pd.DataFrame, train_data: pd.DataFrame) -> None:
    missing_flops = data.loc[data.train_total_flops.isnull()]
    if not missing_flops.empty:
        print("Missing FLOPs data, e.g.")
        print(missing_flops[["model_key", "seed_idx", "adv_training_round"]].head())
        print("Missing models:")
        print(missing_flops.model_key.unique())
        print("Missing training rounds:")
        print(missing_flops.adv_training_round.unique())
        print("Models we have FLOPs for: ")
        print(train_data.model_key.unique())
        print("Training rounds we have FLOPs for: ")
        print(train_data.adv_training_round.unique())
        raise ValueError("Some adversarial training rounds are missing FLOPs data.")


def save_adv_training_data(
    group_names: iter_str | str,
    attack: str,
    dataset: str,
    merge_runs: iter_str | str | None = None,
    summary_keys: list[str] | None = None,
    metrics: list[str] | None = None,
    use_group_cache: bool = True,
):
    if isinstance(group_names, str):
        group_names = (group_names,)
    if summary_keys is None:
        summary_keys = SUMMARY_KEYS

    if metrics is None:
        metrics = METRICS

    data = prepare_adv_training_data(
        group_names=group_names,
        merge_runs=merge_runs,
        summary_keys=summary_keys,
        metrics=metrics,
        use_group_cache=use_group_cache,
    )
    export_csv_and_metadata(
        data,
        save_adv_training_data.__name__,
        {
            "group_names": group_names,
            "attack": attack,
            "dataset": dataset,
            "merge_runs": merge_runs,
            "summary_keys": summary_keys,
            "metrics": metrics,
            "use_group_cache": use_group_cache,
        },
        "adv_training",
        attack,
        dataset,
    )


def load_and_plot_adv_training_plots(
    attack: str,
    dataset: str,
    x_data_name: str = "adv_training_round",
    color_data_name: str = "num_params",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    legend: bool = False,
    check_seeds: int | None = None,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    smoothing: int = DEFAULT_SMOOTHING,
):
    """
    Make adversarial training plots for given runs, pulling data from W&B.

    Args:
        attack: The training/eval attacks to plot (e.g. "gcg_gcg").
        dataset: The dataset to plot (e.g. "imdb").
        x_data_name: The name of the data to use for the x-axis.
        color_data_name: The name of the data to use for the different colors.
        xlim: The x-axis limits.
        ylim: The y-axis limits.
        xscale: The scale of the x-axis.
        yscale: The scale of the y-axis.
        legend: Whether to include the legend in the plot.
        check_seeds:
            Whether to check that the correct number of seeds are present,
            and that those are the correct actual seed numbers.
        y_data_name: The name of the data to use for the y-axis.
        smoothing: The amount of Laplace smoothing to apply to y-values.
    """
    data, metadata = read_csv_and_metadata("adv_training", attack, dataset)
    draw_plot_adv_training(
        data=data,
        metadata=metadata,
        x_data_name=x_data_name,
        color_data_name=color_data_name,
        attack=attack,
        dataset=dataset,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        check_seeds=check_seeds,
        y_data_name=y_data_name,
        smoothing=smoothing,
        xscale=xscale,
        yscale=yscale,
    )


def _prepare_adv_training_data(
    group_names: iter_str,
    summary_keys: list[str],
    metrics: list[str] | None = None,
    save_as: iter_str | str | None = None,
    adjust_flops_for_n_val: bool = False,
    use_group_cache: bool = True,
) -> pd.DataFrame:
    assert all(isinstance(name, str) for name in group_names)
    if "experiment_yaml.run_name" not in summary_keys:
        summary_keys.append("experiment_yaml.run_name")
    run_info_list = []

    for group_name in group_names:
        run_info = get_group_enriched_history(
            group_name=group_name,
            metrics=metrics,
            summary_keys=summary_keys,
            use_group_cache=use_group_cache,
        )
        assert (
            isinstance(run_info, pd.DataFrame) and not run_info.empty
        ), f"Found no data for {group_name}"
        postprocess_data(df=run_info, adjust_flops_for_n_val=adjust_flops_for_n_val)
        run_info_list.append(run_info)

    run_info_df = pd.concat(run_info_list, ignore_index=True)
    # Only add model idx after concat so that we have all the sizes.
    run_info_df = add_model_idx_inplace(
        run_info_df, reference_col="num_params", exists_ok=True
    )
    run_info_df.columns = run_info_df.columns.str.replace("/", "_").str.replace(
        "@", "_at_"
    )
    run_info_df.sort_values("run_created_at", inplace=True, ascending=True)

    run_info_df = drop_duplicates(
        run_info_df,
        keys=["run_name", "adv_training_round"],
        name="adv_data",
        keep="last",
    )
    if save_as is not None:
        export_csv_and_metadata(
            run_info_df,
            _prepare_adv_training_data.__name__,
            {
                "group_names": group_names,
                "summary_keys": summary_keys,
                "metrics": metrics,
                "save_as": save_as,
                "adjust_flops_for_n_val": adjust_flops_for_n_val,
                "use_group_cache": use_group_cache,
            },
            *save_as,
        )
    return run_info_df  # type: ignore


def prepare_adv_training_data(
    group_names: iter_str | str,
    summary_keys: list[str],
    metrics: list[str] | None = None,
    save_as: iter_str | str | None = None,
    adjust_flops_for_n_val: bool = False,
    merge_runs: iter_str | str | None = None,
    use_group_cache: bool = True,
):
    if isinstance(group_names, str):
        group_names = (group_names,)
    adv_data = _prepare_adv_training_data(
        group_names,
        summary_keys,
        metrics,
        save_as=save_as,
        use_group_cache=use_group_cache,
        adjust_flops_for_n_val=adjust_flops_for_n_val,
    )
    if merge_runs is not None:
        adv_data["model_key"] = adv_data.model_name_or_path.str.replace(
            "AlignmentResearch/", ""
        ).str.replace("robust_llm_", "")
        train_data = load_flops_data(merge_runs, use_group_cache=use_group_cache)
        adv_data = merge_adv_and_train_data(adv_data, train_data)
        assert_flops_data_not_missing(adv_data, train_data)
    return adv_data


def extract_seed_from_name(name):
    # name_or_path=AlignmentResearch/robust_llm_pythia-31m_niki-045_pm_random-token-1280_seed-0
    seed_str = name.split(".")[-1]
    seed = int(seed_str)
    assert 0 <= seed <= 10, f"seed was {seed}"
    return seed


def set_up_paper_plot(fig, ax, diagonal_gridlines: bool = False) -> None:
    fig.set_size_inches(3.5, 2.5)
    if diagonal_gridlines:
        for pos in np.linspace(-2, 1, 30):
            ax.axline(
                (pos, 0),
                slope=1,
                linestyle="--",
                color="gray",
                linewidth=0.5,
                alpha=0.5,
                zorder=0,
                transform=plt.gca().transAxes,
            )
    else:
        ax.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
            color="gray",
        )

    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", length=6, width=1)
    ax.tick_params(axis="both", which="minor", length=4, width=0.5)
    ax.set_axisbelow(True)
    set_plot_style("paper")


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
    metadata: PlotMetadata | None,
    x_data_name: str,
    color_data_name: str,
    attack: str,
    dataset: str,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    y_transform: str = "logit",
    smoothing: int = DEFAULT_SMOOTHING,
    xscale: str | None = None,
    yscale: str | None = None,
    add_parity_line: bool = False,
    diagonal_gridlines: bool = False,
):
    title = f"{name_to_attack(attack)}, {name_to_dataset(dataset)}"
    data = data.copy()
    orig_len = len(data)
    data = data.loc[data[y_data_name].notnull()]
    if len(data) < orig_len:
        print(f"Removed {orig_len - len(data)} rows with NaN values in {y_data_name}")

    # Make it one-indexed since evaluation happens after the training
    data["adv_training_round"] += 1

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax, diagonal_gridlines=diagonal_gridlines)
    ax.set_xlabel(AXIS_LABELS[x_data_name])
    if x_data_name == "num_params":
        plt.xscale("log")
    elif x_data_name == "adv_training_round":
        if xlim is not None:
            plt.xlim(xlim)
    elif x_data_name == "n_parameter_updates":
        _get_n_parameter_updates(data)
    elif x_data_name == "train_total_flops":
        data = data.loc[data.train_total_flops.gt(0)]
        # Handle slight deviations in FLOPs
        data["train_total_flops"] = (
            data.groupby(["num_params", "adv_training_round"])["train_total_flops"]
            .transform("mean")
            .astype(int)
        )
    elif x_data_name == "defense_flops_fraction_pretrain":
        data = data.loc[data.train_total_flops.gt(0)]
        # Handle slight deviations in FLOPs
        data["train_total_flops"] = (
            data.groupby(["num_params", "adv_training_round"])["train_total_flops"]
            .transform("mean")
            .astype(int)
        )
        data["pretrain_compute"] = data.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)
        data["defense_flops_fraction_pretrain"] = (
            data.train_total_flops / data.pretrain_compute
        )
        plt.xscale("log")
    else:
        raise ValueError(f"We don't yet support {x_data_name} on the x-axis")

    if smoothing != 0:
        apply_laplace_smoothing(data, y_data_name, smoothing)
    # TODO(ian): Clean up axis label
    try:
        y_label, title = get_offense_defense_ylabel_title(y_data_name, title)
        ax.set_ylabel(y_label)
        y_label_already_set = True
    except ValueError:
        y_label_already_set = False

    if xscale is not None:
        plt.xscale(xscale)
    elif x_data_name in LOG_SCALE_VARIABLES:
        plt.xscale("log")
    if xlim is not None:
        plt.xlim(xlim)
    if yscale is not None:
        plt.yscale(yscale)
    palette_dict = get_color_palette(data, color_data_name)

    plt.title(title)
    y_name_clean = y_data_name.replace("adversarial_eval_", "").replace("metrics_", "")
    y_transf_data_name = (
        f"{y_transform}_{y_name_clean}" if y_transform != "none" else y_name_clean
    )
    data[y_transf_data_name] = TRANSFORMS[y_transform](data[y_data_name])
    if data[y_transf_data_name].dtype == "O":
        data[y_transf_data_name] = data[y_transf_data_name].astype(float)
    data = data.loc[np.isfinite(data[y_transf_data_name])]
    if not y_label_already_set:
        try:
            ax.set_ylabel(AXIS_LABELS[y_transf_data_name])
        except KeyError:
            ax.set_ylabel(y_transf_data_name.replace("_", " ").title())
    if ylim is not None:
        plt.ylim(ylim)
    elif y_transform == "none" and ("asr" in y_data_name or "success" in y_data_name):
        plt.ylim(0, 1)

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
    if add_parity_line:
        # Plot the x = y line
        axis_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
        axis_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        plt.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="red",
            alpha=0.5,
            zorder=0,
            label="x = y",
        )

    if legend:
        legend_handles = get_legend_handles(data, color_data_name, palette_dict)
        create_legend(color_data_name, ax, legend_handles)
    if y_transform == "logit":
        set_yticks_for_logit(ax)
    create_path_and_savefig(
        fig,
        "adv_training",
        attack,
        dataset,
        x_data_name,
        y_transf_data_name,
        f"smoothing-{smoothing}",
        "legend" if legend else "no_legend",
        data=data if legend else None,
        metadata=metadata if legend else None,
    )


def draw_plot_adv_training(
    data: pd.DataFrame,
    metadata: PlotMetadata | None,
    x_data_name: str,
    color_data_name: str,
    attack: str,
    dataset: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    check_seeds: int | None = None,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    y_transform: str = "logit",
    xscale: str | None = None,
    yscale: str | None = None,
    smoothing: int = DEFAULT_SMOOTHING,
    add_parity_line: bool = False,
    diagonal_gridlines: bool = False,
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
        metadata,
        x_data_name,
        color_data_name=color_data_name,
        attack=attack,
        dataset=dataset,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        y_data_name=y_data_name,
        y_transform=y_transform,
        xscale=xscale,
        yscale=yscale,
        smoothing=smoothing,
        add_parity_line=add_parity_line,
        diagonal_gridlines=diagonal_gridlines,
    )


def create_path_and_savefig(
    fig: Figure,
    *nested,
    data: pd.DataFrame | None = None,
    metadata: PlotMetadata | None = None,
    close: bool = True,
    dpi: int = 600,
    bbox_inches: str = "tight",
    width: str | None = None,
    caption: str | None = None,
    trim: tuple[float, float, float, float] | None = None,
    clip: bool = False,
) -> Path:
    assert isinstance(fig, Figure)
    repo_path = compute_repo_path()
    directory = Path(repo_path) / "plots" / "/".join(nested[:-1])
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{nested[-1]}.pdf"
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    if close:
        plt.close(fig)
    print(f"Saved plot to {save_path}")
    if data is not None:
        data.to_csv(save_path.with_name("data.csv"), index=False)
    if metadata is not None:
        metadata.to_latex(save_path, caption=caption, width=width, trim=trim, clip=clip)
    return save_path


def draw_scatterplot(
    orig_data: pd.DataFrame,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    check_seeds: int | None = None,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    smoothing: int = DEFAULT_SMOOTHING,
):
    data = orig_data.copy()
    if smoothing != 0:
        apply_laplace_smoothing(data, y_data_name, smoothing)
    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    plt.xlabel(AXIS_LABELS["num_params"])
    plt.ylabel(AXIS_LABELS[y_data_name])
    if ytransform is None:
        data["y_value"] = data[y_data_name]
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
    if ylim is not None:
        plt.ylim(ylim)

    relevant_data = data[["num_params", "y_value"]]

    if check_seeds is not None:
        _check_correct_num_seeds(
            relevant_data, num_seeds=check_seeds, adversarial=False
        )

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

    if check_seeds is not None:
        _check_correct_num_seeds(
            relevant_data, num_seeds=check_seeds, adversarial=False
        )

    sns.scatterplot(
        data=pd.melt(relevant_data, ["num_params"]),  # type: ignore
        x="num_params",
        y="value",
        hue="variable",
        marker="o",
        legend=False,
        alpha=0.5,
    )
    if ytransform == "logit":
        set_yticks_for_logit(ax)

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
            {"num_params": xs, "adversarial_eval_attack_success_rate": custom_ys}
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
    elif "_t-" in name:
        seed_num = int(name.split("_t-")[-1].split("_")[0])
    else:
        seed_num = int(name.split("_s-")[-1])
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
    orig_data: pd.DataFrame,
    metadata: PlotMetadata | None,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    legend: bool = False,
    check_seeds: int | None = None,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    smoothing: int = DEFAULT_SMOOTHING,
):
    data = orig_data.copy()
    print("Found", len(data), "runs to use for the min/max/median plot")

    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    if smoothing != 0:
        apply_laplace_smoothing(data, y_data_name, smoothing)
    if ytransform is None:
        data["y_value"] = data[y_data_name]
        y_transf_data_name = y_data_name
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
        y_transf_data_name = (
            f"{ytransform}_{y_data_name}" if ytransform != "none" else y_data_name
        )
    if ylim is not None:
        plt.ylim(ylim)
    elif ytransform == "none" and ("asr" in y_data_name or "success" in y_data_name):
        plt.ylim(0, 1)
    plt.xlabel(AXIS_LABELS["num_params"])
    plt.ylabel(AXIS_LABELS[y_transf_data_name])

    relevant_data = data[["num_params", "y_value", "model_name_or_path"]]

    relevant_data = _maybe_get_custom_xs_and_maybe_ys(
        relevant_data, custom_ys, custom_xs_and_ys
    )

    if check_seeds is not None:
        _check_correct_num_seeds(
            relevant_data, num_seeds=check_seeds, adversarial=False
        )

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
    if ytransform == "logit":
        set_yticks_for_logit(ax)

    # Turn off the legend if we don't want it
    if not legend:
        ax.get_legend().remove()

    if isinstance(save_as, str):
        save_as = (save_as,)

    create_path_and_savefig(
        fig,
        *save_as,
        "num_params",
        y_transf_data_name,
        f"smoothing-{smoothing}",
        "legend" if legend else "no_legend",
        data=grouped if legend else None,
        metadata=metadata if legend else None,
    )


def draw_min_max_median_plot_by_round(
    orig_data: pd.DataFrame,
    metadata: PlotMetadata | None,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    legend: bool = True,
    check_seeds: int | None = None,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    rounds: list[int] | None = None,
    smoothing: int = DEFAULT_SMOOTHING,
):
    data = orig_data.copy()
    data = data.loc[data[y_data_name].notnull()]
    print("Found", len(data), "runs to use for the plot from", len(orig_data), "total")

    if smoothing != 0:
        apply_laplace_smoothing(data, y_data_name, smoothing)

    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    if ytransform is None:
        data["y_value"] = data[y_data_name]
        y_transf_data_name = y_data_name
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
        y_transf_data_name = (
            f"{ytransform}_{y_data_name}" if ytransform != "none" else y_data_name
        )
    plt.xlabel(AXIS_LABELS["num_params"])
    plt.ylabel(AXIS_LABELS[y_transf_data_name])
    if ylim is not None:
        plt.ylim(ylim)

    relevant_data = data[
        [
            "num_params",
            "y_value",
            "adv_training_round",
        ]
    ]
    if check_seeds is not None:
        _check_correct_num_seeds(relevant_data, num_seeds=check_seeds, adversarial=True)

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
    colors = sns.color_palette("plasma", n_colors=len(rounds))
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
                markersize=5,
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

    if ytransform == "logit":
        set_yticks_for_logit(ax)

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    if isinstance(save_as, str):
        save_as = (save_as,)

    create_path_and_savefig(
        fig,
        *save_as,
        "num_params",
        y_transf_data_name,
        f"smoothing-{smoothing}",
        "legend" if legend else "no_legend",
        data=grouped if legend else None,
        metadata=metadata if legend else None,
    )


def draw_min_max_median_plot_by_dataset(
    orig_data: pd.DataFrame,
    metadata: PlotMetadata | None,
    title: str,
    save_as: iter_str | str,
    custom_ys=None,
    custom_xs_and_ys=None,
    legend: bool = True,
    check_seeds: int | None = None,
    ylim: tuple[float, float] | None = None,
    ytransform: str | None = None,
    y_data_name: str = "adversarial_eval_attack_success_rate",
    smoothing: int = DEFAULT_SMOOTHING,
    legend_loc: str = "lower left",
):
    data = orig_data.copy()
    data = data.loc[data[y_data_name].notnull()]
    print("Found", len(data), "runs to use for the plot from", len(orig_data), "total")

    if smoothing != 0:
        apply_laplace_smoothing(data, y_data_name, smoothing)

    fig, ax = plt.subplots()

    plt.xscale("log")
    plt.title(title)
    if ytransform is None:
        data["y_value"] = data[y_data_name]
        y_transf_data_name = y_data_name
    else:
        data["y_value"] = TRANSFORMS[ytransform](data[y_data_name])
        y_transf_data_name = (
            f"{ytransform}_{y_data_name}" if ytransform != "none" else y_data_name
        )
    plt.xlabel(AXIS_LABELS["num_params"])
    plt.ylabel(AXIS_LABELS[y_data_name])
    if ylim is not None:
        plt.ylim(ylim)

    relevant_data = data[["num_params", "y_value", "dataset"]]
    if check_seeds is not None:
        _check_correct_num_seeds(relevant_data, num_seeds=check_seeds, adversarial=True)

    relevant_data = _maybe_get_custom_xs_and_maybe_ys(
        relevant_data, custom_ys, custom_xs_and_ys
    )

    # Group by num_params and adv_training_round then calculate min, max, and median
    grouped = (
        relevant_data.groupby(["num_params", "dataset"])
        .agg({"y_value": ["min", "max", "median"]})
        .reset_index()
    )
    grouped.columns = ["num_params", "dataset", "y_min", "y_max", "y_median"]

    # Color palette for different datasets
    datasets = [
        ds
        for ds in ["spam", "imdb", "pm", "wl", "helpful", "harmless"]
        if ds in data["dataset"].unique()
    ]
    pretty_datasets = [name_to_dataset(ds) for ds in datasets]
    colors = sns.color_palette(n_colors=len(datasets))
    color_map = dict(zip(datasets, colors))

    for i, dataset in enumerate(datasets):
        round_data = grouped[grouped["dataset"] == dataset]

        # Plot the band between the min and max values
        plt.fill_between(
            round_data["num_params"],
            round_data["y_min"],
            round_data["y_max"],
            alpha=0.2,
            label="Round {} Min-Max".format(pretty_datasets[i]),
            color=colors[i],
        )

        # Plot the median values
        sns.lineplot(
            x=round_data["num_params"],
            y=round_data["y_median"],
            label="{} Median".format(pretty_datasets[i]),
            marker="o",
            color=colors[i],
            alpha=0.7,
            zorder=5,
            legend=False,
        )

    set_up_paper_plot(fig, ax)

    # Adjust legend
    if legend:
        # Create legend elements
        dataset_legend = [
            Line2D(
                [0], [0], color=color_map[d], lw=2, label=name_to_dataset(d), alpha=0.7
            )
            for d in datasets
        ]
        style_legend = [
            Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="",
                markersize=5,
                label="Median",
            ),
            plt.Rectangle(  # type: ignore
                (0, 0), 1, 1, fc="gray", alpha=0.2, label="Min-Max Range"
            ),
        ]

        # Combine all legend elements
        all_legend_elements = dataset_legend + style_legend

        # Create a single, compact legend
        plt.legend(
            handles=all_legend_elements,
            loc=legend_loc,
            ncol=1,
            fontsize="xx-small",
        )

    if ytransform == "logit":
        set_yticks_for_logit(ax)

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    if isinstance(save_as, str):
        save_as = (save_as,)

    create_path_and_savefig(
        fig,
        *save_as,
        "num_params",
        y_transf_data_name,
        f"smoothing-{smoothing}",
        "legend" if legend else "no_legend",
        data=grouped if legend else None,
        metadata=metadata if legend else None,
    )


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
    plt.xlabel(AXIS_LABELS["num_params"])
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


def postprocess_data(df, adjust_flops_for_n_val: bool = False):
    df.rename(columns={"name_or_path": "model_name_or_path"}, inplace=True)
    if "model_size" in df:
        df["num_params"] = df["model_size"]
    else:
        df["num_params"] = df["model_name_or_path"].map(_get_num_params_from_name)

    if "model_idx" not in df:
        add_model_idx_inplace(df, reference_col="num_params")

    if adjust_flops_for_n_val:
        # Adjust FLOPs for the number of validation examples, since
        # we are interested in the compute per example.
        assert "flops_per_iteration", "dataset_n_val" in df
        df["flops_per_iteration"] = df["flops_per_iteration"] / df["dataset_n_val"]

    _update_model_sizes_as_necessary(df)

    assert "model_name_or_path" in df, (
        "model_name_or_path is necessary for plotting, "
        "but it was not found in the dataframe"
    )
    if "seed_idx" not in df and "training_seed" in df:
        df["seed_idx"] = df["training_seed"]
    if "seed_idx" not in df and "model_name_or_path" in df:
        df["seed_idx"] = df.model_name_or_path.apply(_get_seed_from_name)
    if "seed_idx" in df:
        df.seed_idx = df.seed_idx.astype(int)
    df["pretraining_fraction"] = df["model_name_or_path"].map(_get_pretraining_fraction)


def prepare_asr_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    asr_columns = [col for col in data.columns if col.startswith("metrics_asr_at")]
    other_columns = [
        col
        for col in data.columns
        if "_at_" not in col and "adversarial_eval" not in col
    ]
    melted_df = pd.melt(
        data,
        id_vars=other_columns,
        value_vars=asr_columns,
        var_name="iteration",
        value_name="asr",
    )
    melted_df["iteration"] = melted_df["iteration"].str.split("_").str[-1].astype(int)
    return melted_df


def restrict_asr_to_round(
    orig_df: pd.DataFrame,
    n_models: int,
    n_seeds: int,
    n_iterations: int,
    round: int | float | None = None,
    check_seeds: bool = True,
) -> pd.DataFrame:
    df = orig_df.copy()
    add_columns_for_attack_scaling(df)
    if "model_idx" not in df:
        df = add_model_idx_inplace(df, reference_col="model_size")
    if "seed_idx" not in df and "model_name_or_path" in df:
        df["seed_idx"] = df.model_name_or_path.apply(_get_seed_from_name)
    if "num_params" not in df:
        df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])
    df = df.drop_duplicates(
        subset=["model_idx", "seed_idx", "adv_training_round", "iteration"]
    )
    assert df.model_idx.between(0, n_models - 1).all()
    assert df.seed_idx.between(0, n_seeds - 1).all()
    assert df.iteration.between(0, n_iterations).all()
    if check_seeds:
        assert len(df) == n_models * n_seeds * (n_iterations + 1)

    if round == -1:
        final_round = df.groupby(
            ["model_idx", "seed_idx", "iteration"]
        ).adv_training_round.transform("max")
        df = df.loc[df.adv_training_round == final_round]
    elif isinstance(round, int):
        assert round >= 0
        df = df.loc[df.adv_training_round == round]
    elif isinstance(round, float):
        # Interpret round as % pretrain compute.
        assert 0 < round < 1
        round_data = []
        df.sort_values("model_idx", inplace=True)
        model_rounds: list[int] = []
        for model, model_df in df.groupby("model_idx"):
            diff = np.abs(model_df.defense_flops_fraction_pretrain - round)
            closest_round = max(
                1, model_df.loc[diff.idxmin()].adv_training_round  # type: ignore
            )
            if model_rounds:
                closest_round = min(closest_round, model_rounds[-1])
            round_data.append(
                model_df.loc[model_df.adv_training_round == closest_round]
            )
            model_rounds.append(closest_round)
        df = pd.concat(round_data)

    assert not df.duplicated(subset=["model_idx", "seed_idx", "iteration"]).any()
    df.sort_values("model_idx", inplace=True)
    return df


def round_to_str(round: int | float | None) -> str:
    if round == -1:
        return "final_round"
    elif isinstance(round, int):
        return f"round_{round}"
    elif isinstance(round, float):
        return f"pretrain_fraction_{round * 1e4:.0f}_bps"
    else:
        return "all_rounds"


def add_columns_for_attack_scaling(
    data: pd.DataFrame,
):
    data["iteration_x_params"] = data.iteration * data.num_params
    data["mean_flops_per_iteration"] = data.groupby(["model_idx", "iteration"])[
        "flops_per_iteration"
    ].transform("mean")
    data["iteration_flops"] = data.iteration * data.mean_flops_per_iteration
    data["pretrain_compute"] = data.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)
    data["attack_flops_fraction_pretrain"] = (
        data.iteration_flops / data.pretrain_compute
    )
    return data


def plot_attack_scaling_base(
    orig_df: pd.DataFrame,
    metadata: PlotMetadata | None,
    attack: str,
    dataset: str,
    round_info: str,
    smoothing: int = DEFAULT_SMOOTHING,
    x: str = "attack_flops_fraction_pretrain",
    y: str = "logit_asr",
    color_data_name: str = "num_params",
) -> None:
    """Common method used by finetuning and adversarial attack scaling plots"""
    df = orig_df.copy()
    if smoothing != 0:
        apply_laplace_smoothing(df, "asr", smoothing)
    if "flops" in x:
        df = df.loc[df.iteration.gt(0)]
    add_columns_for_attack_scaling(df)
    if y == "sigmoid_asr":
        df["sigmoid_asr"] = 1 / (1 + np.exp(-df.asr))
    if y == "logit_asr":
        df["logit_asr"] = TRANSFORMS["logit"](df.asr)

    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)
    palette = get_color_palette(df, color_data_name)
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=color_data_name,
        ax=ax,
        palette=palette,
        legend=False,  # We will add the legend later manually
    )
    if x != "iteration":
        # The only case where we don't want log scale is plotting iterations directly
        ax.set_xscale("log")
    # Manually set the ylim here
    if (
        dataset in ["imdb", "spam"]
        and round_info == "finetuned"
        and attack == "gcg"
        and y == "logit_asr"
    ):
        ax.set_ylim(-2.4, 2.4)
    elif (
        dataset in ["helpful", "harmless"]
        and round_info == "finetuned"
        and attack == "gcg"
        and y == "logit_asr"
    ):
        ax.set_ylim(-0.8, 2.5)
    elif y == "asr":
        ax.set_ylim(0, 1)
    if y == "logit_asr":
        set_yticks_for_logit(ax)
    ax.set_xlabel(AXIS_LABELS[x])
    ax.set_ylabel(AXIS_LABELS[y])

    round_pretty = round_info
    if "bps" in round_info:
        bps = int(round_info.split("_")[-2])
        round_pretty = round_pretty.replace(f"{bps}_bps", f"{bps / 100}%")

    if round_pretty == "finetuned":
        # HACK: avoid '(finetuned)' in title
        fig.suptitle(f"{name_to_dataset(dataset)}, {name_to_attack(attack)} Attack")
    else:
        fig.suptitle(
            f"{name_to_dataset(dataset)}, {name_to_attack(attack)} Attack"
            + " ("
            + round_pretty.replace("_", " ").title()
            + ")"
        )
    create_path_and_savefig(
        fig,
        "asr",
        attack,
        dataset,
        round_info,
        x,
        y,
        f"smoothing-{smoothing}",
        "no_legend",
        close=False,
    )
    # Now add the legend and export again
    legend_handles = get_legend_handles(df, color_data_name, palette)
    create_legend(color_data_name, ax, legend_handles, outside=False)
    create_path_and_savefig(
        fig,
        "asr",
        attack,
        dataset,
        round_info,
        x,
        y,
        f"smoothing-{smoothing}",
        "legend",
        data=df,
        metadata=metadata,
    )


def get_cached_asr_data(
    group_name: str, for_offense_defense: bool = False
) -> pd.DataFrame:
    root = compute_repo_path()
    path = os.path.join(root, "cache_csvs", f"asr_{group_name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    if for_offense_defense:
        assert set(df.columns.tolist()) > set(
            ["model_idx", "seed_idx", "asr", "iteration", "adv_training_round"]
        )
    else:
        assert set(df.columns.tolist()) > set(
            ["model_idx", "seed_idx", "asr", "iteration"]
        )
        n_models = 10
        n_seeds = 5
        n_iterations = 11 if "gcg" in group_name else 1281
        assert df.model_idx.between(0, n_models - 1).all()
        assert df.seed_idx.between(0, n_seeds - 1).all()
        assert df.iteration.between(0, n_iterations - 1).all()
        assert len(df) == n_models * n_seeds * n_iterations
        if n_iterations > 1000:
            df = df.loc[df.iteration.mod(100) == 0]
    df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])
    df["iteration_x_params"] = df.iteration * df.num_params
    df.sort_values("model_idx", inplace=True)
    return df


def get_unstacked_cached_asr_data(group_name: str, for_offense_defense: bool = False):
    df = get_cached_asr_data(group_name, for_offense_defense)
    # Assuming your DataFrame is named 'df'
    df_unstacked = df.set_index(
        ["model_size", "seed_idx", "adv_training_round", "model_idx", "iteration"]
    )["asr"].unstack()

    # Rename the columns to the desired format
    df_unstacked.columns = [f"asr_at_{col}" for col in df_unstacked.columns]

    # Reset the index to bring back the other columns as regular columns
    df_unstacked = df_unstacked.reset_index()

    return df_unstacked


def save_asr_data(
    group_names: iter_str | str,
    merge_runs: iter_str | str,
    attack: str,
    dataset: str,
    summary_keys: list[str],
    metrics: list[str],
    n_models: int,
    n_seeds: int,
    n_iterations: int,
    rounds: list[int | float],
    check_seeds: bool = True,
):
    if isinstance(group_names, str):
        group_names = (group_names,)
    if isinstance(merge_runs, str):
        merge_runs = (merge_runs,)
    adv_data = prepare_adv_training_data(
        group_names=group_names,
        summary_keys=summary_keys,
        metrics=metrics,
        merge_runs=merge_runs,
    )

    asr_data = prepare_asr_data(adv_data)
    for round in rounds:
        round_df = restrict_asr_to_round(
            asr_data,
            n_models=n_models,
            n_seeds=n_seeds,
            n_iterations=n_iterations,
            round=round,
            check_seeds=check_seeds,
        )
        export_csv_and_metadata(
            round_df,
            save_asr_data.__name__,
            {
                "group_names": group_names,
                "merge_runs": merge_runs,
                "attack": attack,
                "dataset": dataset,
                "summary_keys": summary_keys,
                "metrics": metrics,
                "n_models": n_models,
                "n_seeds": n_seeds,
                "n_iterations": n_iterations,
                "round": round,
            },
            "asr",
            attack,
            dataset,
            round_to_str(round),
        )


def load_and_plot_asr(
    attack: str,
    dataset: str,
    rounds: list[int | float],
    x: str = "iteration_flops",
    y: str = "logit_asr",
    smoothing: int = DEFAULT_SMOOTHING,
    datapoints: int | None = None,
):
    for round in rounds:
        asr_data, metadata = read_csv_and_metadata(
            "asr", attack, dataset, round_to_str(round)
        )
        if datapoints is not None and asr_data.iteration.nunique() > datapoints:
            # If datapoints=10, filter out all but empirical deciles
            asr_data = asr_data.loc[
                asr_data.iteration.isin(
                    asr_data.iteration.quantile(np.linspace(0, 1, datapoints))
                )
            ]
        plot_attack_scaling_base(
            asr_data,
            metadata,
            attack,
            dataset,
            round_to_str(round),
            x=x,
            y=y,
            smoothing=smoothing,
        )


def prepare_offense_defense_data(
    group_names: iter_str | str,
    attack: str,
    dataset: str,
    merge_runs: iter_str | str | None = None,
    summary_keys: list[str] | None = None,
    metrics: list[str] | None = None,
    target_asr: int = 5,
    use_group_cache: bool = True,
):
    if isinstance(group_names, str):
        group_names = (group_names,)
    if summary_keys is None:
        summary_keys = SUMMARY_KEYS

    if metrics is None:
        metrics = METRICS

    adv_data = prepare_adv_training_data(
        group_names=group_names,
        merge_runs=merge_runs,
        summary_keys=summary_keys,
        metrics=metrics,
        use_group_cache=use_group_cache,
        adjust_flops_for_n_val=True,
    )

    asr_data = pd.concat(
        [get_cached_asr_data(name, for_offense_defense=True) for name in group_names],
        ignore_index=True,
    )
    # HACK: reset model_idx based on model_size because if we merged runs, the model_idx
    # might be wrong if each run only had a subset of the models.
    add_model_idx_inplace(asr_data, reference_col="model_size", exists_ok=True)

    df = adv_data.copy()
    if "num_params" not in df:
        df["num_params"] = df.model_idx.apply(lambda x: MODEL_SIZES[x])

    # We want the same number of iterations for the same model to be grouped
    # together in the plot
    df["mean_train_total_flops"] = df.groupby(["model_idx", "adv_training_round"])[
        "train_total_flops"
    ].transform("mean")
    df["pretrain_compute"] = df.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)
    df["defense_flops_fraction_pretrain"] = (
        df.mean_train_total_flops / df.pretrain_compute
    )

    # Now that we have some kind of x axis data, we need to construct data for
    # the y axis.
    # We do this by getting all the ASRs for each model at each adv
    # training round, and then interpolating to the iteration needed for 10%
    # ASR.

    assert asr_data.asr.between(0, 1).all()
    # We need to sort by iteration so the asr values are in order.
    asr_data = asr_data.sort_values(
        by=["model_idx", "seed_idx", "adv_training_round", "iteration"]
    )
    list_asr_data = (
        asr_data.groupby(["model_size", "seed_idx", "model_idx", "adv_training_round"])[
            "asr"
        ]
        .apply(list)
        .reset_index()
    )

    list_asr_data = drop_duplicates(
        list_asr_data,
        ["model_idx", "seed_idx", "adv_training_round"],
        name="list_asr_data",
    )

    threshold = target_asr / 100
    interpolate_fn = partial(interpolated_iteration_for_asr, asr_threshold=threshold)
    interp_column = f"interpolated_iteration_for_{target_asr}_percent"
    list_asr_data[interp_column] = list_asr_data["asr"].apply(interpolate_fn)

    list_asr_data = drop_duplicates(
        list_asr_data,
        ["model_idx", "seed_idx", "adv_training_round"],
        name="list_asr_data",
    )

    adv_data = adv_data.merge(
        list_asr_data,
        on=["model_idx", "seed_idx", "adv_training_round"],
        how="left",
        validate="one_to_one",
    )
    adv_data["mean_flops_per_iteration"] = adv_data.groupby(["model_idx"])[
        "flops_per_iteration"
    ].transform("mean")
    interp_flops_column = f"{interp_column}_flops"
    adv_data[interp_flops_column] = (
        adv_data[interp_column] * adv_data["mean_flops_per_iteration"]
    )
    adv_data["pretrain_compute"] = adv_data.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)
    # The pretrain compute values are too large to be represented as even as int64
    adv_data["pretrain_compute"] = adv_data["pretrain_compute"].astype(np.float64)
    adv_data[f"{interp_flops_column}_fraction_pretrain"] = (
        adv_data[interp_flops_column] / adv_data.pretrain_compute
    )
    export_csv_and_metadata(
        adv_data,
        prepare_offense_defense_data.__name__,
        {
            "group_names": group_names,
            "attack": attack,
            "dataset": dataset,
            "merge_runs": merge_runs,
            "summary_keys": summary_keys,
            "metrics": metrics,
            "target_asr": target_asr,
            "use_group_cache": use_group_cache,
        },
        "offense_defense",
        attack,
        dataset,
    )


def load_and_plot_offense_defense_plots(
    attack: str,
    dataset: str,
    x_data_name: str = "train_total_flops",
    y_data_name: str = "iteration_x_flops",
    color_data_name: str = "num_params",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    check_seeds: int | None = None,
    # Don't need laplace smoothing for offense-defense
    smoothing: int = 0,
    xscale: str = "log",
    yscale: str = "log",
    add_parity_line: bool = False,
    diagonal_gridlines: bool = False,
):
    adv_data, metadata = read_csv_and_metadata("offense_defense", attack, dataset)
    draw_plot_adv_training(
        data=adv_data,
        metadata=metadata,
        x_data_name=x_data_name,
        color_data_name=color_data_name,
        attack=attack,
        dataset=dataset,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        check_seeds=check_seeds,
        y_data_name=y_data_name,
        y_transform="none",
        smoothing=smoothing,
        xscale=xscale,
        yscale=yscale,
        add_parity_line=add_parity_line,
        diagonal_gridlines=diagonal_gridlines,
    )


def postprocess_attack_compute(df: pd.DataFrame, attack: str, dataset: str) -> None:
    if "logit_asr" not in df.columns:
        df["logit_asr"] = TRANSFORMS["logit"](df["asr"])
    if "model_idx" not in df.columns:
        df = add_model_idx_inplace(df, reference_col="model_size")
    # Fudge the compute for the 12b model due to issue recording multi-GPU
    # flops.
    fudge_factor = get_fudge_factor(attack, dataset)
    df.loc[df.model_idx == 9, "flops_per_iteration"] *= fudge_factor
    df["pretrain_compute"] = df.model_idx.map(ESTIMATED_PRETRAIN_COMPUTE)

    assert df.flops_per_iteration.notnull().all()
    df.flops_per_iteration = df.groupby(
        ["model_idx", "iteration"]
    ).flops_per_iteration.transform("mean")
    df["iteration_x_flops"] = df.iteration * df.flops_per_iteration
    df["attack_flops_fraction_pretrain"] = (
        df.iteration_x_flops / df.pretrain_compute
    ).astype(float)
