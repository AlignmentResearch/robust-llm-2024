from pathlib import Path

from t023_beast_eval_pythia_adv_harmless import launch_adv_eval

if __name__ == "__main__":
    experiment_prefix = f"tom_{Path(__file__).stem.split('_')[0][1:]}"
    launch_adv_eval(
        experiment_prefix=experiment_prefix,
        model_family="qwen25",
        adv_tr_attack="gcg",
        dataset="harmless",
        seed_subrange=range(1),
        model_sizes_to_skip=["14B"],  # cut out 14B
        # Evaling only rounds [1, 3, 6, 8, 10] for now
        rounds_to_skip=[2, 4, 5, 7, 9],
        beast_iters=25,
        beast_beam_size=7,
        run_multiple_kwargs={
            "dry_run": False,
            "skip_git_checks": True,
        },
    )
