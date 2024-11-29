from pathlib import Path

from t019_beast_eval_pythia_ft_harmless import launch_ft_eval

if __name__ == "__main__":
    experiment_prefix = f"tom_{Path(__file__).stem.split('_')[0][1:]}"
    launch_ft_eval(
        experiment_prefix=experiment_prefix,
        model_family="pythia",
        dataset="spam",
        seed_subrange=range(1),
        beast_iters=25,
        beast_beam_size=7,
        is_infix_eval_attack=True,
        run_multiple_kwargs={
            "dry_run": False,
            "skip_git_checks": True,
            "priority": "low-batch",
        },
    )
