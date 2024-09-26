"""Eval niki_150 with GCG"""

from pathlib import Path

from t005b_gcg_infix_vs_rt_imdb import launch

if __name__ == "__main__":
    path = Path(__file__)
    experiment_name_prefix = f"{path.parent.name}_{path.stem.split('_')[0][1:]}"

    launch(
        experiment_name_prefix=experiment_name_prefix,
        adv_train_exp="niki_151",
        adv_train_attack="rt",
        eval_attack="gcg",
        dataset="pm",
        is_infix_eval_attack=False,
        correct_seeds=False,
        # eval one seed at a time
        seed_subrange=[0],
        cluster="h100",
        skip_git_checks=False,
    )
