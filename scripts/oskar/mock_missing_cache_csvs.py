from pathlib import Path

import pandas as pd
from wandb_api_tools import get_group_enriched_history

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.tools import (
    add_model_idx_inplace,
    iter_str,
    postprocess_data,
)
from robust_llm.plotting_utils.utils import drop_duplicates

SUMMARY_KEYS = [
    "experiment_yaml.dataset.n_val",
    "model_family",
    "model_size",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
    "experiment_yaml.evaluation.evaluation_attack.seed",
]

METRICS = [
    "flops_per_iteration",
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/n_correct_pre_attack",
    "adversarial_eval/n_incorrect_pre_attack",
    "adversarial_eval/n_examples",
    "adversarial_eval/n_correct_post_attack",
    "adversarial_eval/n_incorrect_post_attack",
    "adversarial_eval/post_attack_accuracy",
] + [f"metrics/asr@{i}" for i in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 128]]

root = Path(compute_repo_path())

GROUPS = [
    "tom_011_eval_niki_151_gcg",
    "tom_012_eval_niki_153_gcg",
    "tom_013_eval_niki_171_gcg",
    "tom_014_eval_niki_172_gcg",
    "tom_015_eval_niki_152_gcg_prefix",
    "tom_015_eval_niki_152a_gcg_prefix",
    "tom_016_eval_niki_170_gcg_prefix",
]


def old_prepare_adv_training_data(
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
    return run_info_df  # type: ignore


if __name__ == "__main__":
    for group in GROUPS:
        path = root / "cache_csvs" / "evaluation" / f"{group}.csv"
        # if path.exists():
        #     print(f"Skipping {path}")
        #     continue
        adv_data = old_prepare_adv_training_data(
            (group,), metrics=METRICS, summary_keys=SUMMARY_KEYS
        )
        adv_data = drop_duplicates(
            adv_data,
            [
                "model_name_or_path",
                "model_revision",
                "adv_training_round",
                "evaluation_evaluation_attack_seed",
            ],
            name="adv_data",
        )
        assert adv_data.flops_per_iteration.notnull().all()
        asr_cols = [c for c in adv_data.columns if "asr_at" in c]
        # Convert columns like "metrics_asr_at_12", "metrics_asr_at_24"...
        # to "asr" and have a row per iteration.
        asr_data = adv_data.melt(
            id_vars=[c for c in adv_data.columns if c not in asr_cols],
            value_vars=asr_cols,
            var_name="asr_at",
            value_name="asr",
        )
        asr_data.rename(columns={"asr_at": "iteration"}, inplace=True)
        asr_data.iteration = asr_data.iteration.str.extract(r"asr_at_(\d+)").astype(int)
        asr_data.to_csv(path, index=False)
