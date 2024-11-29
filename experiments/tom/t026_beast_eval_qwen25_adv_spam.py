from pathlib import Path

from t023_beast_eval_pythia_adv_harmless import launch_adv_eval

if __name__ == "__main__":
    experiment_prefix = f"tom_{Path(__file__).stem.split('_')[0][1:]}"
    launch_adv_eval(
        experiment_prefix=experiment_prefix,
        model_family="qwen25",
        adv_tr_attack="gcg",
        dataset="spam",
        seed_subrange=range(3),
        run_multiple_kwargs={
            "dry_run": False,
            "skip_git_checks": True,
        },
    )
