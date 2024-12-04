from pathlib import Path

from wandb_api_tools import get_group_enriched_history

from robust_llm.file_utils import compute_repo_path
from robust_llm.plotting_utils.tools import drop_duplicates, postprocess_data

SUMMARY_KEYS = [
    "experiment_yaml.dataset.n_val",
    "model_family",
    "model_size",
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.model.revision",
    "experiment_yaml.evaluation.evaluation_attack.seed",
    "experiment_yaml.evaluation.num_iterations",
]

METRICS = [
    "flops_per_iteration",
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/n_correct_pre_attack",
    "adversarial_eval/n_incorrect_pre_attack",
    "adversarial_eval/n_examples",
    "adversarial_eval/attack_success_rate",
    "adversarial_eval/n_correct_post_attack",
    "adversarial_eval/n_incorrect_post_attack",
    "adversarial_eval/post_attack_accuracy",
]
GROUP = "ian_142a_gen_strongreject_gcg_qwen25"


def old_make_finetuned_data(
    group: str,
    metrics: list[str],
    eval_summary_keys: list[str],
):
    run = get_group_enriched_history(
        group_name=group,
        metrics=metrics,
        summary_keys=eval_summary_keys,
    )
    postprocess_data(run)
    run = drop_duplicates(
        run, ["model_idx", "seed_idx", "adv_training_round"], "wandb_data"
    )
    return run


root = Path(compute_repo_path())

if __name__ == "__main__":
    df = old_make_finetuned_data(GROUP, metrics=METRICS, eval_summary_keys=SUMMARY_KEYS)
    df.rename(
        columns={
            "adversarial_eval/attack_success_rate": "asr",
            "evaluation_num_iterations": "iteration",
        },
        inplace=True,
    )
    path = root / "cache_csvs" / "evaluation" / f"{GROUP}.csv"
    df.to_csv(path, index=False)
