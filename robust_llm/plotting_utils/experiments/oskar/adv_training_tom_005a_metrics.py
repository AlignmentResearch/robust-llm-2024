"""Plot robustness metrics using Tom's transfer evals."""

import numpy as np
import pandas as pd

from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import load_and_plot_asr_and_ifs

pd.set_option("display.max_columns", 100)
# Set the plot style
set_plot_style("paper")

# Include the name for easier debugging
summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.dataset.n_val",
    "model_size",
    "experiment_yaml.model.revision",
]
METRICS = (
    ["adversarial_eval/attack_success_rate"]
    + [f"metrics/asr@{i}" for i in list(range(0, 128, 12)) + [128]]
    + [f"metrics/ifs@{r:.1f}" for r in np.arange(0, 1.1, 0.1)]
)

load_and_plot_asr_and_ifs(
    run_names=("tom_005a_eval_niki_149_gcg ",),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
    attack="rt_gcg",
    dataset="imdb",
    n_models=10,
    n_seeds=3,
    check_seeds=False,
    n_iterations=128,
)
load_and_plot_asr_and_ifs(
    run_names=("tom_006_eval_niki_150_gcg ",),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
    attack="rt_gcg",
    dataset="spam",
    n_models=10,
    n_seeds=3,
    check_seeds=False,
    n_iterations=128,
)
load_and_plot_asr_and_ifs(
    run_names=("tom_007_eval_niki_152_gcg ", "tom_007_eval_niki_152a_gcg "),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
    attack="gcg_gcg",
    dataset="imdb",
    n_models=10,
    n_seeds=3,
    check_seeds=False,
    n_iterations=128,
)
load_and_plot_asr_and_ifs(
    run_names=(
        "tom_008_eval_niki_152_gcg_infix90 ",
        "tom_008_eval_niki_152a_gcg_infix90 ",
    ),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
    attack="gcg_infix90",
    dataset="imdb",
    n_models=10,
    n_seeds=3,
    check_seeds=False,
    n_iterations=128,
)
load_and_plot_asr_and_ifs(
    run_names=("tom_009_eval_niki_170_gcg ",),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
    attack="gcg_gcg",
    dataset="spam",
    n_models=10,
    n_seeds=3,
    check_seeds=False,
    n_iterations=128,
)
load_and_plot_asr_and_ifs(
    run_names=("tom_010_eval_niki_170_gcg_infix90 ",),
    summary_keys=summary_keys,
    metrics=METRICS,
    use_cache=True,
    attack="gcg_infix90",
    dataset="spam",
    n_models=10,
    n_seeds=3,
    check_seeds=False,
    n_iterations=128,
)
