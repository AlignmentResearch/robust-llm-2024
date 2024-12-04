"""Generate data for basic finetuned plots (robustness vs. size)"""

from robust_llm.plotting_utils.tools import (
    get_attack_from_name,
    get_dataset_from_name,
    get_family_from_name,
    make_finetuned_data,
)

metrics = [
    "adversarial_eval/attack_success_rate",
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/n_correct_post_attack",
    "adversarial_eval/n_incorrect_post_attack",
    "adversarial_eval/n_correct_pre_attack",
    "adversarial_eval/n_examples",
    "model_size",
    "adv_training_round",
]
summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.dataset.n_val",
    "experiment_yaml.evaluation.evaluation_attack.seed",
]

FINETUNED_RUNS = [
    "ian_102a_gcg_pythia_harmless",
    "ian_103a_gcg_pythia_helpful",
    "ian_104a_rt_pythia_harmless",
    "ian_105a_rt_pythia_helpful",
    "ian_106_gcg_pythia_imdb",
    "ian_107_gcg_pythia_pm",
    "ian_108_gcg_pythia_wl",
    "ian_109_gcg_pythia_spam",
    "ian_110_rt_pythia_imdb",
    "ian_111_rt_pythia_pm",
    "ian_112_rt_pythia_wl",
    "ian_113_rt_pythia_spam",
    "oskar_025a_gcg_eval_qwen25_ft_harmless",
    "oskar_025b_gcg_eval_qwen25_ft_spam",
    "ian_142a_gen_strongreject_gcg_qwen25",
    "tom_019b_beast_eval_pythia_ft_harmless",
    "tom_021b_beast_eval_qwen25_ft_harmless",
    "tom_022b_beast_eval_qwen25_ft_spam",
]


def main():
    for run in FINETUNED_RUNS:
        family = get_family_from_name(run)
        attack = get_attack_from_name(run)
        dataset = get_dataset_from_name(run)

        make_finetuned_data(
            group_names=[
                run,
            ],
            save_as=("finetuned", family, attack, dataset),
            metrics=metrics,
            eval_summary_keys=summary_keys,
        )


if __name__ == "__main__":
    main()
