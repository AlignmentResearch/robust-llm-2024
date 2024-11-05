"""Generate data for basic finetuned plots (robustness vs. size)"""

from robust_llm.plotting_utils.tools import make_finetuned_data

metrics = [
    "adversarial_eval/attack_success_rate",
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/n_correct_post_attack",
    "adversarial_eval/n_examples",
    "model_size",
    "adv_training_round",
]
summary_keys = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.dataset.n_val",
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
]


def main():
    for run in FINETUNED_RUNS:
        attack, dataset = run.split("_")[2], run.split("_")[-1]

        make_finetuned_data(
            group_names=[
                run,
            ],
            save_as=("finetuned", attack, dataset),
            metrics=metrics,
            eval_summary_keys=summary_keys,
        )


if __name__ == "__main__":
    main()
