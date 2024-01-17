# Runs a batch job on k8s using the code from the current commit.
# Based on https://github.com/AlignmentResearch/learned-planners/blob/main/experiments/sokoban.py  # noqa: E501

import argparse
import dataclasses
import functools
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

from git.repo import Repo
from names_generator import generate_name

from robust_llm.utils import ask_for_confirmation

JOB_TEMPLATE_PATH = Path(__file__).parent / "k8s" / "batch_job.yaml"
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
) -> tuple[Sequence[str], str]:
    launch_id = generate_name(style="hyphen")

    jobs = []
    for i, run in enumerate(runs):
        # Set random job_name. experiment_name and job_type are either specified
        # by the user or taken from hydra config.
        random_name = generate_name(style="hyphen")
        wandb_job_name = f"{random_name}-{i}"
        job_name = f"rllm-{wandb_job_name}"

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

    jobs, launch_id = create_jobs(runs)
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


def main(args):
    run = FlamingoRun(
        script_path=args.script_path,
        hydra_config=args.hydra_config,
        override_args={
            "experiment.experiment_name": args.experiment_name,
            "experiment.job_type": args.job_type,
        },
        CONTAINER_TAG=args.container_tag,
        CPU=args.cpu,
        MEMORY=args.memory,
        GPU=args.gpu,
        PRIORITY=args.priority,
    )

    launch_jobs([run], dry_run=args.dryrun)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script_path",
        type=str,
        help="Path to the script to run",
        default="robust_llm",
    )
    parser.add_argument(
        "--hydra_config",
        type=str,
        help="Name of the hydra config to use",
        required=True,
    )
    parser.add_argument("--experiment_name", type=str, help="experiment_name for wandb")
    parser.add_argument("--job_type", type=str, help="job_type for wandb")
    parser.add_argument(
        "--container_tag", type=str, help="container tag to use", default="latest"
    )
    parser.add_argument("--cpu", type=int, help="number of cpus to use", default=4)
    parser.add_argument("--memory", type=str, help="memory to use", default="20G")
    parser.add_argument("--gpu", type=int, help="number of gpus to use", default=1)
    parser.add_argument(
        "--priority", type=str, help="priority of job", default="normal-batch"
    )
    parser.add_argument("--dryrun", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
