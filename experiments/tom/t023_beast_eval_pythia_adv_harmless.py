from pathlib import Path
from typing import Any, Iterable

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import QWEN_EVAL_ROUNDS

# (model size, gpus, memory, cluster, n_parallel)
MODEL_SETTINGS: dict[str, list[tuple[str, int, str, str, int]]] = {
    "pythia": [
        ("14m", 1, "40G", "a6k", 3),
        ("31m", 1, "40G", "a6k", 3),
        ("70m", 1, "40G", "a6k", 3),
        ("160m", 1, "40G", "a6k", 3),
        ("410m", 1, "40G", "a6k", 3),
        ("1b", 1, "40G", "a6k", 2),
        ("1.4b", 1, "40G", "a6k", 1),
        ("2.8b", 1, "40G", "h100", 1),
        ("6.9b", 1, "100G", "h100", 1),
        ("12b", 1, "110G", "h100", 1),
    ],
    "qwen25": [
        ("0.5B", 1, "40G", "a6k", 1),
        ("1.5B", 1, "40G", "a6k", 1),
        ("3B", 1, "50G", "a6k", 1),
        ("7B", 1, "100G", "h100", 1),
        ("14B", 1, "110G", "h100", 1),
    ],
}


# Result of experiment_utils.get_rounds_to_evaluate("pythia", "gcg") prior to a
# bug fix.
PYTHIA_ROUNDS_TO_EVAL = [
    [1, 2, 3, 4, 6, 9, 15, 24, 38, 60],
    [1, 2, 3, 4, 6, 9, 15, 24, 38, 60],
    [1, 2, 3, 4, 6, 9, 15, 24, 38, 60],
    [1, 2, 3, 4, 6, 9, 15, 23, 37, 59],
    [1, 2, 3, 4, 5, 6, 7, 10, 14, 21],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
]
# Buggy points in PYTHIA_ROUNDS_TO_EVAL.
PYTHIA_ROUNDS_TO_SKIP = {
    "14m": [],
    "31m": [],
    "70m": [],
    "160m": [],
    "410m": [],
    "1b": [9],
    "1.4b": [7],
    "2.8b": [6],
    "6.9b": [6],
    "12b": [6],
}


def launch_adv_eval(
    experiment_prefix: str,
    model_family: str,
    adv_tr_attack: str,
    dataset: str,
    # which seeds to eval in this launch
    seed_subrange: Iterable[int],
    model_sizes_to_skip: Iterable[str] = [],
    rounds_to_skip: Iterable[int] = [],
    beast_iters: int | None = None,
    beast_beam_size: int | None = None,
    is_infix_eval_attack: bool = False,
    run_multiple_kwargs: dict[str, Any] = {},
):
    eval_attack_name = "beast"
    if is_infix_eval_attack:
        eval_attack_name += "_infix90"
    experiment_name = (
        f"{experiment_prefix}_{eval_attack_name}_eval"
        f"_{model_family}_adv_{adv_tr_attack}_{dataset}"
    )
    hydra_config = f"tom/beast_{model_family}_{dataset}"

    if model_family == "pythia":
        model_family_config_str = "pythia"
        model_sizes = [x[0] for x in MODEL_SETTINGS[model_family]]
        rounds_by_model_size = dict(
            zip(
                model_sizes,
                PYTHIA_ROUNDS_TO_EVAL,
                strict=True,
            )
        )
    elif model_family == "qwen25":
        model_family_config_str = "Qwen2.5"
        rounds_by_model_size = QWEN_EVAL_ROUNDS
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    beast_override_args: dict[str, Any] = {}
    if beast_iters is not None:
        beast_override_args["evaluation.num_iterations"] = beast_iters
    if beast_beam_size is not None:
        beast_override_args["evaluation.evaluation_attack.beam_search_width"] = (
            beast_beam_size
        )
        beast_override_args["evaluation.evaluation_attack.beam_branch_factor"] = (
            beast_beam_size
        )
    if is_infix_eval_attack:
        beast_override_args["evaluation.evaluation_attack.perturb_position_min"] = 0.9
        beast_override_args["evaluation.evaluation_attack.perturb_position_max"] = 0.9

    override_tuples = [
        (
            {
                "+model": (
                    f"AdvTrained/clf/{adv_tr_attack}/{dataset}"
                    f"/{model_family_config_str}-{model_size}-s{seed}"
                ),
                "model.revision": f"adv-training-round-{evaluation_round}",
                "environment.minibatch_multiplier": 0.25,
                "evaluation.evaluation_attack.seed": seed,
                **beast_override_args,
            },
            n_gpus,
            memory,
            cluster,
            parallel,
        )
        for model_size, n_gpus, memory, cluster, parallel in MODEL_SETTINGS[
            model_family
        ]
        for seed in range(5)
        for evaluation_round in rounds_by_model_size[model_size]
    ]
    override_args_list = [x[0] for x in override_tuples]
    n_gpus = [x[1] for x in override_tuples]
    memory = [x[2] for x in override_tuples]
    cluster = [x[3] for x in override_tuples]
    parallel = [x[4] for x in override_tuples]

    seed_subrange = set(seed_subrange)
    skip_runs_mask = [
        int(ov["+model"].split("-s")[-1]) not in seed_subrange  # type: ignore [attr-defined]  # noqa: E501
        for ov in override_args_list
    ]

    model_sizes_to_skip = set(model_sizes_to_skip)
    skip_runs_mask = [
        bit or ov["+model"].split("-")[-2] in model_sizes_to_skip  # type: ignore [attr-defined]  # noqa: E501
        for bit, ov in zip(skip_runs_mask, override_args_list)
    ]

    rounds_to_skip = set(rounds_to_skip)
    skip_runs_mask = [
        bit
        or int(ov["model.revision"].split("-")[-1]) in rounds_to_skip  # type: ignore [attr-defined]  # noqa: E501
        or int(ov["model.revision"].split("-")[-1]) in PYTHIA_ROUNDS_TO_SKIP[ov["+model"].split("-")[-2]]  # type: ignore [attr-defined]  # noqa: E501
        for bit, ov in zip(skip_runs_mask, override_args_list)
    ]

    run_multiple(
        experiment_name,
        hydra_config,
        override_args_list,
        gpu=n_gpus,
        memory=memory,
        cluster=cluster,
        n_max_parallel=parallel,
        cpu=1,
        skip_runs_mask=skip_runs_mask,
        **run_multiple_kwargs,
    )


if __name__ == "__main__":
    experiment_prefix = f"tom_{Path(__file__).stem.split('_')[0][1:]}"
    launch_adv_eval(
        experiment_prefix=experiment_prefix,
        model_family="pythia",
        adv_tr_attack="gcg",
        dataset="harmless",
        seed_subrange=range(3),
        run_multiple_kwargs={
            "dry_run": False,
            "skip_git_checks": True,
        },
    )
