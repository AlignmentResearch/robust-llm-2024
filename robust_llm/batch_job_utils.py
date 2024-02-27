# Utils for running k8s batch jobs.
# Based on https://github.com/AlignmentResearch/learned-planners/blob/a3db9a8b4933d85031d26d32f28e2ad29ee630ed/experiments/sokoban.py#L35-L134  # noqa: E501

import dataclasses
import functools
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from git.repo import Repo
from names_generator import generate_name

from robust_llm.utils import ask_for_confirmation

JOB_TEMPLATE_PATH = Path(__file__).parent.parent / "k8s" / "batch_job.yaml"
with JOB_TEMPLATE_PATH.open() as f:
    JOB_TEMPLATE = f.read()


@functools.cache
def git_latest_commit() -> str:
    repo = Repo(".")
    commit_hash = str(repo.head.object.hexsha)
    return commit_hash


@dataclass
class FlamingoRun:
    script_path: str
    hydra_config: str
    override_args: dict
    n_max_parallel: int = 1
    CONTAINER_TAG: str = "latest"
    COMMIT_HASH: str = dataclasses.field(default_factory=git_latest_commit)
    CPU: Union[int, str] = 4
    MEMORY: str = "20G"
    GPU: int = 1
    PRIORITY: str = "normal-batch"

    def format_args(self) -> dict[str, Union[str, int]]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name
            not in ["script_path", "hydra_config", "override_args", "n_max_parallel"]
        }


def organize_by_containers(runs: Sequence[FlamingoRun]) -> list[list[FlamingoRun]]:
    """Splits runs into groups that will be run together in a single k8s container."""
    # Sort from "less demanding" to "more demanding" jobs.
    runs = list(sorted(runs, key=lambda x: -x.n_max_parallel))
    current_container: list[FlamingoRun] = []
    runs_by_containers = [current_container]
    for run in runs:
        # Run can fit into the current container.
        if len(current_container) + 1 <= run.n_max_parallel:
            current_container.append(run)
        else:
            current_container = [run]
            runs_by_containers.append(current_container)
    return runs_by_containers


def create_job_for_multiple_runs(
    runs: Sequence[FlamingoRun],
    name: str,
    index: int,
    launch_id: str,
    project: str,
    entity: str,
    wandb_mode: str,
) -> str:
    # K8s job/pod names should be short for readability (hence cutting the name).
    k8s_job_name = (
        f"rllm-{name[:16]}-{index}"
        if len(runs) == 1
        else f"rllm-{name[:16]}-{index}-{index+len(runs)-1}"
    )

    single_commands = []
    for i, run in enumerate(runs):
        aux_args = [f"{k}={v}" for k, v in run.override_args.items()]
        split_command = [
            "PYTHONPATH=.",
            "python",
            run.script_path,
            f"+experiment={run.hydra_config}",
            f"experiment.run_name={name}-{index+i}",
            *aux_args,
        ]
        single_commands.append(shlex.join(split_command))
    single_commands.append("wait")
    command = "(" + " & ".join(single_commands) + ")"

    # Currently, we keep too much info in the FlamingoRun, including info that should be
    # shared across all runs. Hence, we check below that it is indeed the same.
    # TODO(michal): refactor to make it reasonable.
    for run in runs:
        assert runs[0].format_args() == run.format_args()

    job = JOB_TEMPLATE.format(
        NAME=k8s_job_name,
        LAUNCH_ID=launch_id,
        WANDB_ENTITY=entity,
        WANDB_PROJECT=project,
        WANDB_MODE=wandb_mode,
        COMMAND=command,
        **runs[0].format_args(),
    )

    return job


def create_jobs(
    runs: Sequence[FlamingoRun],
    project: str = "robust-llm",
    entity: str = "farai",
    wandb_mode: str = "online",
    experiment_name: Optional[str] = None,
) -> tuple[Sequence[str], str]:
    launch_id = generate_name(style="hyphen")

    jobs = []
    name = (experiment_name or generate_name(style="hyphen")).replace("_", "-")

    runs_by_containers = organize_by_containers(runs)

    index = 0
    for runs in runs_by_containers:
        jobs.append(
            create_job_for_multiple_runs(
                runs, name, index, launch_id, project, entity, wandb_mode
            )
        )
        index += len(runs)

    return jobs, launch_id


def launch_jobs(
    runs: Sequence[FlamingoRun],
    project: str = "robust-llm",
    entity: str = "farai",
    experiment_name: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[str, str]:
    repo = Repo(".")
    # Push to git as we want to run the code with the current commit.
    repo.remote("origin").push(repo.active_branch.name).raise_if_error()
    # Check if repo is dirty.
    if repo.is_dirty(untracked_files=True):
        should_continue = ask_for_confirmation(
            "Git repo is dirty. Are you sure you want to continue?"
        )
        if not should_continue:
            print("Aborting")
            sys.exit(1)

    jobs, launch_id = create_jobs(
        runs,
        project=project,
        entity=entity,
        experiment_name=experiment_name,
    )
    yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

    print(yamls_for_all_jobs)

    if not dry_run:
        print(f"Launching jobs with launch_id={launch_id}...")
        subprocess.run(
            ["kubectl", "create", "-f", "-"],
            check=True,
            input=yamls_for_all_jobs.encode(),
        )
        print(
            "Jobs launched. To delete them run:\n"
            f"kubectl delete jobs -l launch-id={launch_id}"
        )

    return yamls_for_all_jobs, launch_id


def run_multiple(
    experiment_name: str,
    hydra_config: str,
    override_args_list: Sequence[dict],
    n_max_parallel: Optional[Sequence[int]] = None,
    script_path: str = "robust_llm",
    container_tag: str = "latest",
    cpu: int = 4,
    memory: str = "20G",
    gpu: int = 1,
    priority: str = "normal-batch",
) -> None:
    """Run an experiment containing multiple runs and multiple k8s jobs.

    Potentially, several runs can be fit into a single k8s job and share a GPU.

    Args:
        experiment_name: descriptive name of the experiment, used to set wandb group.
        hydra_config: hydra config name.
        override_args_list: list of dictionaries with override arguments for each run.
        n_max_parallel: If provided, each element `n_max_parallel[i]` denotes the
            maximum number of runs that can be fit together in the container that
            includes a run corresponding to `override_args_list[i]`. If None, every run
            will be allocated a separate container.
        script_path: path of the Python script to run.
        container_tag: Docker container tag to use.
        cpu: number of cpu cores per container.
        memory: memory per container.
        gpu: GPUs per container.
        priority: K8s priority.
    """
    if n_max_parallel is not None:
        assert len(n_max_parallel) == len(override_args_list)

    runs = [
        (
            FlamingoRun(
                script_path=script_path,
                hydra_config=hydra_config,
                override_args={
                    "experiment.experiment_name": experiment_name,
                    **override_args,
                },
                n_max_parallel=n_max_parallel[i] if n_max_parallel else 1,
                CONTAINER_TAG=container_tag,
                CPU=cpu,
                MEMORY=memory,
                GPU=gpu,
                PRIORITY=priority,
            )
        )
        for (i, override_args) in enumerate(override_args_list)
    ]

    launch_jobs(runs, experiment_name=experiment_name)
