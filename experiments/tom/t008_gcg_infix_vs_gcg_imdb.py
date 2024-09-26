"""Eval niki_152 with 90%-infix GCG"""

from pathlib import Path

from t005b_gcg_infix_vs_rt_imdb import launch

if __name__ == "__main__":
    path = Path(__file__)
    experiment_name_prefix = f"{path.parent.name}_{path.stem.split('_')[0][1:]}"

    # models split in two because they're split across niki_152 and niki_152a
    launch(
        experiment_name_prefix=experiment_name_prefix,
        adv_train_exp="niki_152",
        adv_train_attack="gcg",
        eval_attack="gcg",
        dataset="imdb",
        is_infix_eval_attack=True,
        correct_seeds=False,
        # eval one seed at a time
        seed_subrange=[0],
        model_subrange=range(4, 8),
        cluster="h100",
        skip_git_checks=False,
    )

    launch(
        experiment_name_prefix=experiment_name_prefix,
        # maybe should rename this to niki_152 to match above, since this is
        # only used for adv train name? I think I'll still keep it different
        # since niki may rerun niki_152 to use correct seeds
        adv_train_exp="niki_152a",
        adv_train_attack="gcg",
        eval_attack="gcg",
        dataset="imdb",
        is_infix_eval_attack=True,
        correct_seeds=True,
        # eval one seed at a time
        seed_subrange=[0],
        model_subrange=range(0, 4),
        cluster="a6k",
        skip_git_checks=True,
        is_first_job_high_priority=False,
    )
