# Utils for running k8s batch jobs.
# Based on https://github.com/AlignmentResearch/learned-planners/blob/a3db9a8b4933d85031d26d32f28e2ad29ee630ed/experiments/sokoban.py#L35-L134  # noqa: E501

import dataclasses
import functools
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import hydra
from git.repo import Repo
from hydra.core.config_store import ConfigStore
from hydra.errors import HydraException
from names_generator import generate_name

from robust_llm import logger
from robust_llm.config.configs import ExperimentConfig
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
    base_command: str
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

    def __post_init__(self):
        """Validate the override_args by using the Hydra Compose API.

        We validate the override_args by using hydra.compose to build a full
        config, and if it errors then we know there was a mistake in the args.
        However, we can't use this Hydra config to run the experiment since
        there's no straightforward way to pass it to k8s, so we throw it
        away and pass the override args like normal anyway."""
        # since the entry point is not __main__.py, we need to initialize Hydra
        with hydra.initialize(version_base=None, config_path="hydra_conf"):
            cs = ConfigStore.instance()
            cs.store(name="base_config", node=ExperimentConfig)
            formatted_overrides = []
            formatted_overrides.append(f"+experiment={self.hydra_config}")
            formatted_overrides.extend(_prepare_override_args(self.override_args))
            try:
                hydra.compose("base_config", overrides=formatted_overrides)
            except HydraException as e:
                raise ValueError("override_args are invalid, aborting.") from e

    def format_args(self) -> dict[str, Union[str, int]]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name
            not in [
                "base_command",
                "script_path",
                "hydra_config",
                "override_args",
                "n_max_parallel",
            ]
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
        aux_args = _prepare_override_args(run.override_args)
        split_command = [
            "PYTHONPATH=.",
            *run.base_command.split(" "),
            run.script_path,
            f"+experiment={run.hydra_config}",
            f"run_name={name}-{index+i}",
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


def _prepare_override_args(override_args: dict[str, Any]) -> list[str]:
    args = []
    for k, v in override_args.items():
        if v is None:
            args.append(f"{k}=null")
        else:
            args.append(f"{k}={v}")
    return args


def create_jobs(
    runs: Sequence[FlamingoRun],
    project: str = "robust-llm",
    entity: str = "farai",
    wandb_mode: str = "online",
    experiment_name: Optional[str] = None,
    only_jobs_with_starting_indices: Optional[Sequence[int]] = None,
) -> tuple[Sequence[str], str]:
    launch_id = generate_name(style="hyphen")

    jobs = []
    name = (experiment_name or generate_name(style="hyphen")).replace("_", "-")

    runs_by_containers = organize_by_containers(runs)

    index = 0
    for runs in runs_by_containers:
        # If only_jobs_with_starting_indices is specified, we only launch jobs with
        # a starting index in this list.
        if (
            only_jobs_with_starting_indices is None
            or index in only_jobs_with_starting_indices
        ):
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
    only_jobs_with_starting_indices: Optional[Sequence[int]] = None,
    dry_run: bool = False,
    skip_git_checks: bool = False,
) -> tuple[str, str]:
    """Launch k8s jobs for the given runs.

    Args:
        runs: a list of FlamingoRun objects.
        project: wandb project to use.
        entity: wandb entity to use.
        experiment_name: descriptive name of the experiment, used to set wandb group.
        only_jobs_with_starting_indices: if not None, only jobs with starting indices
            contained in this list will be launched. Useful for rerunning a small subset
            of jobs from an experiment.
        dry_run: if True, only print the k8s job yaml files without launching them.
        skip_git_checks: if True, skip the remote push and the check for dirty git repo.
            This is useful when running unit tests.

    Returns:
        pair of strings -- yaml file with k8s jobs definitions, and the launch_id.
    """
    repo = Repo(".")
    if not skip_git_checks:
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
        only_jobs_with_starting_indices=only_jobs_with_starting_indices,
    )
    yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

    logger.info(yamls_for_all_jobs)

    if not dry_run:
        logger.info("Launching jobs with launch_id=%s...", launch_id)
        subprocess.run(
            ["kubectl", "create", "-f", "-"],
            check=True,
            input=yamls_for_all_jobs.encode(),
        )
        logger.info(
            "Jobs launched. To delete them run:\n"
            "kubectl delete jobs -l launch-id=%s",
            launch_id,
        )

    return yamls_for_all_jobs, launch_id


def run_multiple(
    experiment_name: str,
    hydra_config: str,
    override_args_list: Sequence[dict],
    n_max_parallel: Optional[Sequence[int]] = None,
    use_accelerate: bool = False,
    script_path: str = "robust_llm",
    container_tag: str = "latest",
    cpu: int = 4,
    memory: str = "20G",
    gpu: int = 1,
    priority: str = "normal-batch",
    only_jobs_with_starting_indices: Optional[Sequence[int]] = None,
    dry_run: bool = False,
    skip_git_checks: bool = False,
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
        use_accelerate: whether to use accelerate for distributed training.
        script_path: path of the Python script to run.
        container_tag: Docker container tag to use.
        cpu: number of cpu cores per container.
        memory: memory per container.
        gpu: GPUs per container.
        priority: K8s priority.
        only_jobs_with_starting_indices: if not None, only jobs with starting indices
            contained in this list will be launched. Useful for rerunning a small subset
            of jobs from an experiment (for example, if a few jobs failed).
        dry_run: if True, only print the k8s job yaml files without launching them.
        skip_git_checks: if True, skip the remote push and the check for dirty git repo.
    """
    if n_max_parallel is not None:
        assert len(n_max_parallel) == len(override_args_list)

    assert use_accelerate == (gpu > 1)

    base_command = (
        f"accelerate launch --config_file=accelerate_config.yaml --num_processes={gpu}"
        if use_accelerate
        else "python"
    )

    runs = [
        (
            FlamingoRun(
                base_command=base_command,
                script_path=script_path,
                hydra_config=hydra_config,
                override_args={
                    "experiment_name": experiment_name,
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

    launch_jobs(
        runs,
        experiment_name=experiment_name,
        only_jobs_with_starting_indices=only_jobs_with_starting_indices,
        dry_run=dry_run,
        skip_git_checks=skip_git_checks,
    )
