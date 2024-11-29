from pathlib import Path
from typing import Any, Iterable

from robust_llm.batch_job_utils import run_multiple

MODEL_GPU_MEMORY_CLUSTER: dict[str, list[tuple[str, int, str, str]]] = {
    "pythia": [
        (
            "Default/clf/{dataset}/pythia-14m-s{seed}",
            1,
            "10G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-31m-s{seed}",
            1,
            "10G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-70m-s{seed}",
            1,
            "10G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-160m-s{seed}",
            1,
            "10G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-410m-s{seed}",
            1,
            "15G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-1b-s{seed}",
            1,
            "15G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-1.4b-s{seed}",
            1,
            "20G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/pythia-2.8b-s{seed}",
            1,
            "30G",
            "h100",
        ),
        (
            "Default/clf/{dataset}/pythia-6.9b-s{seed}",
            1,
            "100G",
            "h100",
        ),
        (
            "Default/clf/{dataset}/pythia-12b-s{seed}",
            1,
            "100G",
            "h100",
        ),
    ],
    "qwen25": [
        (
            "Default/clf/{dataset}/Qwen2.5-0.5B-s{seed}",
            1,
            "30G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/Qwen2.5-1.5B-s{seed}",
            1,
            "40G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/Qwen2.5-3B-s{seed}",
            1,
            "50G",
            "a6k",
        ),
        (
            "Default/clf/{dataset}/Qwen2.5-7B-s{seed}",
            1,
            "100G",
            "h100",
        ),
        (
            "Default/clf/{dataset}/Qwen2.5-14B-s{seed}",
            1,
            "110G",
            "h100",
        ),
    ],
}


def launch_ft_eval(
    experiment_prefix: str,
    model_family: str,
    dataset: str,
    # which seeds to eval in this launch
    seed_subrange: Iterable[int],
    beast_iters: int | None = None,
    beast_beam_size: int | None = None,
    is_infix_eval_attack: bool = False,
    run_multiple_kwargs: dict[str, Any] = {},
):
    eval_attack_name = "beast"
    if is_infix_eval_attack:
        eval_attack_name += "_infix90"
    experiment_name = (
        f"{experiment_prefix}_{eval_attack_name}_eval" f"_{model_family}_ft_{dataset}"
    )
    hydra_config = f"tom/beast_{model_family}_{dataset}"

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
                "+model": model.format(dataset=dataset, seed=seed),
                **beast_override_args,
            },
            n_gpus,
            memory,
            cluster,
        )
        for seed in range(5)
        for (model, n_gpus, memory, cluster) in MODEL_GPU_MEMORY_CLUSTER[model_family]
    ]
    override_args_list = [x[0] for x in override_tuples]
    n_gpus = [x[1] for x in override_tuples]
    memory = [x[2] for x in override_tuples]
    cluster = [x[3] for x in override_tuples]

    seed_subrange = set(seed_subrange)
    skip_runs_mask = [
        int(ov["+model"].split("-s")[-1]) not in seed_subrange  # type: ignore [attr-defined]  # noqa: E501
        or cluster == "h100"
        for ov, cluster in zip(override_args_list, memory, strict=True)
    ]

    run_multiple(
        experiment_name,
        hydra_config,
        override_args_list,
        gpu=n_gpus,
        memory=memory,
        cluster=cluster,
        cpu=1,
        skip_runs_mask=skip_runs_mask,
        **run_multiple_kwargs,
    )


if __name__ == "__main__":
    experiment_prefix = f"tom_{Path(__file__).stem.split('_')[0][1:]}"
    launch_ft_eval(
        experiment_prefix=experiment_prefix,
        model_family="pythia",
        dataset="harmless",
        seed_subrange=range(1),
        run_multiple_kwargs={
            "dry_run": False,
            "skip_git_checks": True,
        },
    )
