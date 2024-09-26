"""Eval niki_149 with GCG

From now onwards I will prefix these experiment files with a letter so that they
can be imported (imports cannot start with a number).
"""

from pathlib import Path
from typing import Iterable

import numpy as np

from robust_llm.batch_job_utils import run_multiple


def launch(
    experiment_name_prefix: str,
    adv_train_exp: str,
    adv_train_attack: str,
    eval_attack: str,
    dataset: str,
    correct_seeds=True,
    seed_range: Iterable[int] | None = None,
    seed_subrange: Iterable[int] | None = None,
    model_subrange: Iterable[int] | None = None,
    is_infix_eval_attack: bool = False,
    nfs: bool = False,
    dry_run: bool = False,
    cluster: str | None = None,
    skip_git_checks: bool = False,
    experiment_name_suffix: str = "",
):
    PROJECT = "AlignmentResearch/robust_llm"
    BASE_MODELS = [
        "pythia-14m",
        "pythia-31m",
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
    ]
    # We can pack more on h100 but it seems to be worse for throughput.
    # 3-parallel 14mil takes 6 hours, 4-parallel 14mil takes 7.5 hours,
    # 7-parallel 14mil takes 17.5 hours.
    # Not sure what the sweet spot for throughput is.
    n_parallels = (
        [4, 4, 4, 4, 4, 3, 2, 1] if cluster == "h100" else [3, 3, 3, 3, 3, 2, 1, 1]
    )

    hydra_config = f"ian/iclr2024_stronger_{eval_attack}_{dataset}"
    # list of (finetune seed, adv train seed)
    seeds = [(i, i) for i in range(5)] if correct_seeds else [(0, i) for i in range(5)]
    if seed_range is not None:
        seeds = [seeds[i] for i in seed_range]
    max_adv_tr_rounds = 250 if adv_train_attack == "rt" else 60
    n_adv_tr_rounds = [
        np.clip(x, 5, max_adv_tr_rounds) for x in [953, 413, 163, 59, 21, 8, 6, 3]
    ]

    # how many adv training rounds to sample from start, middle, and end
    NUM_START_ROUNDS = 10
    NUM_MIDDLE_ROUNDS = 8
    NUM_END_ROUNDS = 5
    tr_rounds_to_eval = [
        [
            x
            for x in set(
                np.concatenate(
                    [
                        np.arange(NUM_START_ROUNDS + 1),
                        # evenly spaced points in log space. stop is
                        # inclusive (not exclusive like arange). set() will
                        # de-dupe the endpoints
                        np.geomspace(
                            start=NUM_START_ROUNDS,
                            stop=n_rounds + 1 - NUM_END_ROUNDS,
                            num=NUM_MIDDLE_ROUNDS + 2,
                            dtype=int,
                        ),
                        np.arange(n_rounds + 1 - NUM_END_ROUNDS, n_rounds + 1),
                    ],
                )
            )
            if 0 <= x <= n_rounds
        ]
        for n_rounds in n_adv_tr_rounds
    ]

    base_overrides: dict[str, str | int | float] = {}
    if eval_attack == "gcg":
        base_overrides["evaluation.num_iterations"] = 128
    elif eval_attack == "rt":
        base_overrides["evaluation.num_iterations"] = 4096
    else:
        raise ValueError(f"Unknown eval_attack: {eval_attack}")
    if is_infix_eval_attack:
        base_overrides["evaluation.evaluation_attack.perturb_position_min"] = 0.9
        base_overrides["evaluation.evaluation_attack.perturb_position_max"] = 0.9

    # forgot to eval 1.4b initially so i'm sticking it at the end of
    # overrides_and_parallelism because I think that will make the wandb job
    # completion checking still work
    base_models_14b = [BASE_MODELS[-2]]
    n_parallels_14b = [n_parallels[-2]]
    tr_rounds_to_eval_14b = [tr_rounds_to_eval[-2]]
    del BASE_MODELS[-2]
    del n_parallels[-2]
    del tr_rounds_to_eval[-2]

    def make_overrides_and_parallelism(
        base_models: Iterable[str],
        n_parallels: Iterable[int],
        tr_rounds_to_eval: Iterable[Iterable[int]],
    ):
        return [
            (
                {
                    **base_overrides,
                    "+model": f"Default/clf/{dataset}/{base_model}-s{finetune_seed}",
                    "model.name_or_path": (
                        f"{PROJECT}_clf_{dataset}_{base_model}"
                        f"_s-{finetune_seed}_adv_tr_{adv_train_attack}_t-{adv_tr_seed}"
                    ),
                    "model.revision": f"adv-training-round-{tr_round}",
                },
                n_parallel,
            )
            for base_model, n_parallel, tr_rounds in zip(
                base_models, n_parallels, tr_rounds_to_eval
            )
            for tr_round in tr_rounds
            for finetune_seed, adv_tr_seed in seeds
        ]

    overrides_and_parallelism = make_overrides_and_parallelism(
        BASE_MODELS, n_parallels, tr_rounds_to_eval
    ) + make_overrides_and_parallelism(
        base_models_14b, n_parallels_14b, tr_rounds_to_eval_14b
    )
    overrides, n_parallel = tuple(list(t) for t in zip(*overrides_and_parallelism))

    max_memory = 114 if cluster == "h100" else 52
    memories = [f"{min(max_memory, par * 17)}G" for par in n_parallel]

    skip_runs_mask = [False] * len(overrides)
    if seed_subrange is not None:
        seed_subrange = set(seed_subrange)
        skip_runs_mask = [
            int(ov["model.name_or_path"].split("_t-")[-1]) not in seed_subrange
            for ov in overrides
        ]
    if model_subrange is not None:
        model_subrange = list(model_subrange)
        for i in model_subrange:
            assert 0 <= i < len(BASE_MODELS)
        model_set = set(BASE_MODELS[i] for i in model_subrange)
        skip_runs_mask = [
            bit or ov["+model"].split("/")[-1].split("-s")[0] not in model_set
            for bit, ov in zip(skip_runs_mask, overrides)
        ]

    experiment_name = f"{experiment_name_prefix}_eval_{adv_train_exp}_{eval_attack}"
    if is_infix_eval_attack:
        experiment_name += "_infix90"
    if experiment_name_suffix:
        experiment_name += f"_{experiment_name_suffix}"

    run_multiple(
        experiment_name,
        hydra_config,
        overrides,
        n_parallel,
        memory=memories,
        cpu=1,
        skip_runs_mask=skip_runs_mask,
        use_cluster_storage=nfs,
        cluster=cluster,
        dry_run=dry_run,
        skip_git_checks=skip_git_checks or dry_run,
    )


if __name__ == "__main__":
    path = Path(__file__)
    experiment_name_prefix = f"{path.parent.name}_{path.stem.split('_')[0][1:]}"

    launch(
        experiment_name_prefix=experiment_name_prefix,
        adv_train_exp="niki_149",
        adv_train_attack="rt",
        eval_attack="gcg",
        dataset="imdb",
        is_infix_eval_attack=False,
        correct_seeds=False,
        # let's eval just 3 seeds for now since these evals are so expensive.
        # (in later experiments (t006 and later) we set seed_range to be
        # everything and just use seed_subrange; for this run if we want to eval
        # the remaining two seeds then we'll need to also change the exp name to
        # avoid mismatching names with the existing runs)
        seed_range=range(3),
        # jk let's just eval 1 seed at a time. seed_range is used for numbering
        # the runs since we already launched some 3-seed evals that
        # partially completed
        seed_subrange=[0],
        cluster="a6k",
    )
