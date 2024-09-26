from pathlib import Path

from t005b_gcg_infix_vs_rt_imdb import launch

if __name__ == "__main__":
    path = Path(__file__)
    experiment_name_prefix = f"{path.parent.name}_{path.stem.split('_')[0][1:]}"

    launch(
        experiment_name_prefix=experiment_name_prefix,
        adv_train_exp="niki_170",
        adv_train_attack="gcg",
        eval_attack="gcg",
        dataset="spam",
        is_infix_eval_attack=True,
        # eval one seed at a time
        seed_subrange=[0],
        cluster="h100",
        skip_git_checks=False,
        is_first_job_high_priority=False,
    )
