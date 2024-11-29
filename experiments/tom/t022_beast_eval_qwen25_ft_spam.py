from pathlib import Path

from t019_beast_eval_pythia_ft_harmless import launch_ft_eval

if __name__ == "__main__":
    experiment_prefix = f"tom_{Path(__file__).stem.split('_')[0][1:]}"
    launch_ft_eval(
        experiment_prefix=experiment_prefix,
        model_family="qwen25",
        dataset="spam",
        seed_subrange=range(3),
        run_multiple_kwargs={
            "dry_run": False,
            "skip_git_checks": True,
        },
    )
