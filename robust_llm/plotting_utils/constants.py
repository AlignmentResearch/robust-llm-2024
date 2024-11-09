import re
from typing import Any

from robust_llm.wandb_utils.constants import MODEL_SIZES

MODEL_PLOTTING_NAMES = [
    "7.6M",
    "17.6M",
    "44.7M",
    "123.7M",
    "353.8M",
    "908.8M",
    "1.3B",
    "2.6B",
    "6.7B",
    "11.6B",
]

assert len(MODEL_SIZES) == len(MODEL_PLOTTING_NAMES)
PLOTTING_NAME_DICT = dict(zip(MODEL_SIZES, MODEL_PLOTTING_NAMES))


def get_fudge_factor(attack: str, dataset: str) -> float:
    """This is a per-task multiplier we use to fix multi-GPU flops for the 12b model.

    The values are all chosen empirically.
    """
    if dataset.lower() == "imdb":
        if attack.lower() == "gcg":
            return 1.6

    elif dataset.lower() == "spam":
        if attack.lower() == "gcg":
            return 1.78

    elif dataset.lower() == "pm":
        if attack.lower() == "gcg":
            return 1.75

    return 1.0


DEFAULT_SMOOTHING = 1

AXIS_LABELS = {
    "adversarial_eval_attack_success_rate": "Attack Success Rate",
    "adversarial_eval_pre_attack_accuracy": "Pre-Attack Accuracy",
    "pre_attack_accuracy": "Pre-Attack Accuracy",
    "logit_pre_attack_accuracy": "Pre-Attack Accuracy",
    "post_attack_accuracy": "Post-Attack Accuracy",
    "logit_post_attack_accuracy": "Post-Attack Accuracy",
    "logit_adversarial_eval_attack_success_rate": "Attack Success Rate",
    "logit_asr": "Attack Success Rate",
    "log_asr": "log(Attack Success Rate)",
    "asr": "Attack Success Rate",
    "model_size": "Model Size (# Parameters)",
    "num_params": "Model Size (# Parameters)",
    "pretrain_compute": "Pretraining FLOPs",
    "log_pretrain_compute": "log(Pretraining FLOPs)",
    "adv_training_round": "Adversarial Training Round",
    "iteration": "Attack Iteration",
    "iteration_x_params": "Attack Compute\n(Iterations $\times$ Parameters)",
    "iteration_x_flops": "Attack Compute (FLOPs)",
    "iteration_flops": "Attack Compute (FLOPs)",
    "n_parameter_updates": "Number of Parameter Updates",
    "train_total_flops": "Adversarial Training Compute (FLOPs)",
    "ifs": "Iterations to Reach Attack Success Rate",
    "log_ifs": "log(Iterations to Reach Attack Success Rate)",
    "attack_flops_fraction_pretrain": "Attack Compute\n(Proportion of Pretraining)",
    "defense_flops_fraction_pretrain": "Adversarial Training Compute\n(Proportion of Pretraining)",  # noqa: E501
    "logit_attack_success_rate": "Attack Success Rate",
    "metrics_asr_at_2": "Attack Success Rate",
    "metrics_asr_at_12": "Attack Success Rate",
    "metrics_asr_at_60": "Attack Success Rate",
    "metrics_asr_at_72": "Attack Success Rate",
    "metrics_asr_at_120": "Attack Success Rate",
    "metrics_asr_at_128": "Attack Success Rate ",
    "logit_asr_at_2": "Attack Success Rate",
    "logit_asr_at_12": "Attack Success Rate",
    "logit_asr_at_60": "Attack Success Rate",
    "logit_asr_at_72": "Attack Success Rate",
    "logit_asr_at_120": "Attack Success Rate",
    "logit_asr_at_128": "Attack Success Rate",
    "logit_metrics_asr_at_2": "Attack Success Rate",
    "logit_metrics_asr_at_12": "Attack Success Rate",
    "logit_metrics_asr_at_60": "Attack Success Rate",
    "logit_metrics_asr_at_72": "Attack Success Rate",
    "logit_metrics_asr_at_120": "Attack Success Rate",
    "logit_metrics_asr_at_128": "Attack Success Rate",
    "mean_log_prob": "Mean Log Probability",
    "log_mean_prob": "log(Mean Probability)",
    "mean_logit_prob": "Mean Log Probability (Logit)",
    "logit_mean_prob": "logit(Mean Attack Success Probability)",
    "mean_loss": "Mean Cross Entropy Loss",
    "asp": "Mean Attack Success Probability",
}
LOG_SCALE_VARIABLES = [
    "num_params",
    "pretrain_compute",
    "iteration_x_params",
    "iteration_x_flops",
    "iteration_flops",
    "n_parameter_updates",
    "train_total_flops",
    "attack_flops_fraction_pretrain",
    "defense_flops_fraction_pretrain",
]

RUN_NAMES = {
    "rt_gcg": {
        "imdb": {
            "group_names": "tom_005a_eval_niki_149_gcg",
            "merge_runs": "niki_149_adv_tr_rt_imdb_small",
        },
        "spam": {
            "group_names": "tom_006_eval_niki_150_gcg",
            "merge_runs": ("niki_150_adv_tr_rt_spam_small",),
        },
        "wl": {
            "group_names": "tom_012_eval_niki_153_gcg",
            "merge_runs": (
                "niki_153_adv_tr_rt_wl_small",
                "niki_159_adv_tr_rt_wl_large",
            ),
        },
        "pm": {
            "group_names": "tom_011_eval_niki_151_gcg",
            "merge_runs": (
                "niki_151_adv_tr_rt_pm_small",
                "niki_158_adv_tr_rt_pm_large",
            ),
        },
    },
    "gcg_gcg_infix90": {
        "imdb": {
            "group_names": (
                "tom_008_eval_niki_152_gcg_infix90",
                "tom_008_eval_niki_152a_gcg_infix90",
            ),
            "merge_runs": (
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
            ),
        },
        "spam": {
            "group_names": "tom_010_eval_niki_170_gcg_infix90",
            "merge_runs": "niki_170_adv_tr_gcg_spam_small",
        },
    },
    "gcg_gcg_prefix": {
        "imdb": {
            "group_names": (
                "tom_015_eval_niki_152_gcg_prefix",
                "tom_015_eval_niki_152a_gcg_prefix",
            ),
            "merge_runs": (
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
            ),
        },
        "spam": {
            "group_names": "tom_016_eval_niki_170_gcg_prefix",
            "merge_runs": "niki_170_adv_tr_gcg_spam_small",
        },
    },
    "gcg_gcg": {
        "imdb": {
            "group_names": (
                "tom_007_eval_niki_152_gcg",
                "tom_007_eval_niki_152a_gcg",
            ),
            "merge_runs": (
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
            ),
        },
        "spam": {
            "group_names": "tom_009_eval_niki_170_gcg",
            "merge_runs": "niki_170_adv_tr_gcg_spam_small",
        },
        "wl": {
            "group_names": "tom_014_eval_niki_172_gcg",
            "merge_runs": "niki_172_adv_tr_gcg_wl_small",
        },
        "pm": {
            "group_names": "tom_013_eval_niki_171_gcg",
            "merge_runs": "niki_171_adv_tr_gcg_pm_small",
        },
    },
    "gcg_gcg_match_seed": {
        "imdb": {
            "group_names": ("oskar_019b_suffix_eval_adv_tr_gcg_imdb_small",),
            "merge_runs": (
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
                # "ian_141_rerun_adv_tr_gcg_imdb_small", # Flops data unuseable
            ),
        },
    },
    "gcg_gcg_infix90_match_seed": {
        "imdb": {
            "group_names": ("oskar_019a_infix_eval_adv_tr_gcg_imdb_small",),
            "merge_runs": (
                "niki_152a_adv_tr_gcg_imdb_small",
                "niki_152_adv_tr_gcg_imdb_small",
                # "ian_141_rerun_adv_tr_gcg_imdb_small", # Flops data unuseable
            ),
        },
    },
    "gcg_no_ramp_gcg": {
        "imdb": {
            "group_names": "niki_iki_eval_niki_182_gcg",
            "merge_runs": "niki_182_adv_tr_no_ramp_small",
        },
    },
}


def get_run_names(attack: str, dataset: str) -> dict[str, Any]:
    attack_info = RUN_NAMES.get(attack)
    assert isinstance(attack_info, dict)
    dataset_info = attack_info.get(dataset)
    assert isinstance(dataset_info, dict)
    return dataset_info


def get_offense_defense_ylabel_title(y_data_name: str, title: str) -> tuple[str, str]:
    # Try to match the offense-defense plot naming convention.
    regex = r"^interpolated_iteration_for_(\d+\.?\d*)_percent(_flops|_flops_fraction_pretrain)?$"  # noqa: E501
    match = re.match(regex, y_data_name)
    if not match:
        raise ValueError(f"Could not match {y_data_name} for offense-defense")
    target_asr = match.group(1)
    title = f"{title}\nTarget Attack Success Rate {target_asr}%"
    if y_data_name.endswith("flops"):
        return "Attack Compute FLOPs", title
    elif y_data_name.endswith("pretrain"):
        return AXIS_LABELS["attack_flops_fraction_pretrain"], title
    else:
        return "Interpolated Iteration for Target Attack Success Rate", title
