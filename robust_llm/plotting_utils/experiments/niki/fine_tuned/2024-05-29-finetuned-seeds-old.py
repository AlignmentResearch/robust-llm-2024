from robust_llm.plotting_utils.style import set_plot_style
from robust_llm.plotting_utils.tools import make_finetuned_plot

metrics = [
    "adversarial_eval/attack_success_rate",
]

save_dir = "old"

set_plot_style("paper")

# PasswordMatch GCG
make_finetuned_plot(
    run_names=("mz_070_search_based_tt_eval_seeds_models",),
    title="PasswordMatch, GCG attack",
    save_as="pm_gcg",
    save_dir=save_dir,
    eval_summary_keys=(
        [
            "experiment_yaml.environment.model_name_or_path",
            "experiment_yaml.evaluation.evaluation_attack.search_based_attack_config.n_its",  # noqa: E501
            "experiment_yaml.evaluation.evaluation_attack.search_based_attack_config.search_type",  # noqa: E501
        ],
    ),
    custom_ys=[
        0.79,  # 7
        0.77,  # 8
        0.96,  # 9
        0.075,  # 10
    ],
    metrics=(metrics,),
)

# IMDB GCG
make_finetuned_plot(
    run_names=("mz_072_search_based_imdb_eval_seeds_models",),
    title="IMDB, GCG attack",
    save_as="imdb_gcg",
    save_dir=save_dir,
    eval_summary_keys=(
        [
            "experiment_yaml.environment.model_name_or_path",
            "experiment_yaml.evaluation.evaluation_attack.search_based_attack_config.n_its",  # noqa: E501
            "experiment_yaml.evaluation.evaluation_attack.search_based_attack_config.search_type",  # noqa: E501
        ],
    ),
    custom_ys=[
        0.28125,
        0.41025641025641024,
        0.33505154639175255,
        0.6071428571428571,
    ],
    metrics=(metrics,),
)

# Spam GCG
make_finetuned_plot(
    run_names=("mz_074_search_based_spam_eval_seeds_models",),
    title="Spam, GCG attack",
    save_as="spam_gcg",
    save_dir=save_dir,
    eval_summary_keys=(
        [
            "experiment_yaml.environment.model_name_or_path",
            "experiment_yaml.evaluation.evaluation_attack.search_based_attack_config.n_its",  # noqa: E501
            "experiment_yaml.evaluation.evaluation_attack.search_based_attack_config.search_type",  # noqa: E501
        ],
    ),
    custom_ys=[
        0.7424242424242424,
        0.601010101010101,
        0.898989898989899,
        0.5606060606060606,
    ],
    metrics=(metrics,),
)

# PasswordMatch RandomToken
make_finetuned_plot(
    run_names=("niki_096_rt_pm_eval_seeds_models",),
    title="PasswordMatch, RT attack",
    save_as="pm_rt",
    save_dir=save_dir,
    eval_summary_keys=(
        [
            "experiment_yaml.model.name_or_path",
        ],
    ),
    custom_xs_and_ys=[
        (1311629312, 0.03),  # 1.4b
        (2646435840, 0.03),  # 2.8b
        (6650740736, 0.01),  # 6.9b
        (11586560000, 0.0),  # 12b
    ],
    metrics=(metrics,),
)

# Spam RandomToken
make_finetuned_plot(
    run_names=("niki_097_rt_spam_eval_seeds_models",),
    title="Spam, RandomToken attack",
    save_as="spam_rt",
    save_dir=save_dir,
    eval_summary_keys=(
        [
            "experiment_yaml.model.name_or_path",
        ],
    ),
    custom_xs_and_ys=[
        (1311629312, 0.45454545454545453),  # 1.4b
        (2646435840, 0.3165829145728643),  # 2.8b
        (6650740736, 0.4472361809045226),  # 6.9b
        (11586560000, 0.3939393939393939),  # 12b
    ],
    metrics=(metrics,),
)

# IMDB RandomToken
make_finetuned_plot(
    run_names=("niki_098_rt_imdb_eval_seeds_models",),
    title="IMDB, RandomToken attack",
    save_as="imdb_rt",
    save_dir=save_dir,
    eval_summary_keys=(
        [
            "experiment_yaml.model.name_or_path",
        ],
    ),
    custom_xs_and_ys=[
        (1311629312, 0.11458333333333333),  # 1.4b
        (2646435840, 0.13020833333333334),  # 2.8b
        (6650740736, 0.035897435897435895),
        (11586560000, 0.04639175257731959),
    ],
    metrics=(metrics,),
)
