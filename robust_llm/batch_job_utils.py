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
            if f.name not in ["script_path", "hydra_config", "override_args"]
        }


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
    for i, run in enumerate(runs):
        # Set random job_name. experiment_name and job_type are either specified
        # by the user or taken from hydra config.
        wandb_job_name = f"{name}-{i}"
        # K8s job/pod names should be short for readability.
        job_name = f"rllm-{name[:16]}-{i}"

        aux_args = [f"{k}={v}" for k, v in run.override_args.items()]
        split_command = [
            "PYTHONPATH=.",
            "python",
            run.script_path,
            f"+experiment={run.hydra_config}",
            f"experiment.run_name={wandb_job_name}",
            *aux_args,
        ]
        job = JOB_TEMPLATE.format(
            NAME=job_name,
            LAUNCH_ID=launch_id,
            WANDB_ENTITY=entity,
            WANDB_PROJECT=project,
            WANDB_MODE=wandb_mode,
            COMMAND=shlex.join(split_command),
            **run.format_args(),
        )
        jobs.append(job)

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
        runs, project=project, entity=entity, experiment_name=experiment_name
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
    script_path: str = "robust_llm",
    container_tag: str = "latest",
    cpu: int = 4,
    memory: str = "20G",
    gpu: int = 1,
    priority: str = "normal-batch",
) -> None:
    runs = [
        FlamingoRun(
            script_path=script_path,
            hydra_config=hydra_config,
            override_args={
                "experiment.experiment_name": experiment_name,
                **override_args,
            },
            CONTAINER_TAG=container_tag,
            CPU=cpu,
            MEMORY=memory,
            GPU=gpu,
            PRIORITY=priority,
        )
        for override_args in override_args_list
    ]

    launch_jobs(runs, experiment_name=experiment_name)
