# Utils for running k8s batch jobs.
# Based on https://github.com/AlignmentResearch/learned-planners/blob/a3db9a8b4933d85031d26d32f28e2ad29ee630ed/experiments/sokoban.py#L35-L134  # noqa: E501

import dataclasses
import functools
import random
import shlex
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import hydra
import wandb
from git.repo import Repo
from hydra.core.config_store import ConfigStore
from hydra.errors import HydraException
from names_generator import generate_name

from robust_llm.config.configs import ExperimentConfig
from robust_llm.utils import ask_for_confirmation

T = TypeVar("T")

JOB_TEMPLATE_PATH = Path(__file__).parent.parent / "k8s" / "batch_job.yaml"
with JOB_TEMPLATE_PATH.open() as f:
    JOB_TEMPLATE = f.read()

# Container tag to use by default for jobs. Update this when uploading a
# new version of the canonical Docker image. (Avoid reusing tags as
# older versions of that image may be cached on K8s nodes.)
DEFAULT_CONTAINER_TAG = "2024-08-07-backoff"


def memory_str_to_bytes(memory: str) -> int:
    """Converts Kubernetes memory string to bytes."""
    if len(memory) < 2:
        raise ValueError(f"Unexpected memory string: {memory}")
    if memory[-1] == "G":
        return int(memory[:-1]) * (1000**3)
    if memory[-2:] == "Gi":
        return int(memory[:-2]) * (1024**3)
    # Job memory limits are nearly always in G or Gi, so we don't bother with
    # other cases.
    raise ValueError(f"Unexpected memory string: {memory}")


def run_multiple(
    experiment_name: str,
    hydra_config: str,
    override_args_list: Sequence[dict],
    n_max_parallel: int | list[int] = 1,
    script_path: str = "robust_llm",
    container_tag: str = DEFAULT_CONTAINER_TAG,
    cpu: int | list[int] = 4,
    memory: str | list[str] = "20G",
    gpu: int | list[int] = 1,
    priority: str | list[str] = "normal-batch",
    cluster: str | Sequence[str | None] | None = None,
    only_jobs_with_starting_indices: Optional[Sequence[int]] = None,
    skip_runs_mask: Sequence[bool] | None = None,
    use_cluster_storage: bool = True,
    wandb_mode: str = "online",
    dry_run: bool = False,
    skip_git_checks: bool = False,
    unique_identifier: str | None = None,
) -> None:
    """Run an experiment containing multiple runs and multiple k8s jobs.

    Potentially, several runs can be fit into a single k8s job and share a GPU.

    Args:
        experiment_name: descriptive name of the experiment, used to set wandb group.
        hydra_config: hydra config name.
        override_args_list: list of dictionaries with override arguments for each run.
        n_max_parallel: Entry i of the list (or the global value if an int is passed)
            is the maximum number of runs that can be fit together in the
            container that includes a run corresponding to `override_args_list[i]`.
            By default, every run will be allocated a separate container.
        script_path: path of the Python script to run.
        container_tag: Docker container tag to use.
        cpu: number of cpu cores per container (can set globally or one per run).
        memory: memory per container (can set globally or one per run).
        gpu: GPUs per container (can set globally or one per run).
        priority: K8s priority (can set globally or one per run).
        cluster: K8s cluster to use (can set globally or one per run). As of
            2024/08/24, options are "a6k" and "h100".
        only_jobs_with_starting_indices: if not None, only jobs with starting indices
            contained in this list will be launched. Useful for rerunning a small subset
            of jobs from an experiment (for example, if a few jobs failed).
        skip_runs_mask: Mask of runs to skip. Useful for running a subset of
            jobs.
        use_cluster_storage: Mounts cluster storage to the job if true.
        wandb_mode: Value to give the WANDB_MODE environment variable.
        dry_run: if True, only print the k8s job yaml files without launching them.
        skip_git_checks: if True, skip the remote push and the check for dirty git repo.
        unique_identifier: A unique identifier to append to the k8s job names
            to avoid name conflicts. If None, a random identifier will be generated.
    """

    n_max_parallel = ensure_list(n_max_parallel, len(override_args_list))
    cpu = ensure_list(cpu, len(override_args_list))
    memory = ensure_list(memory, len(override_args_list))
    gpu = ensure_list(gpu, len(override_args_list))
    priority = ensure_list(priority, len(override_args_list))
    if skip_runs_mask is None:
        skip_runs_mask = [False] * len(override_args_list)
    skip_runs_mask = ensure_list(list(skip_runs_mask), len(override_args_list))

    cluster_list = fill_cluster_list(
        ensure_list(
            cluster,  # type: ignore # TODO(ian): Fix typing here
            len(override_args_list),
        ),
    )

    unique_identifier = unique_identifier or generate_chars(length=4)

    hyphened_name = experiment_name.replace("_", "-")
    runs = [
        (
            FlamingoRun(
                base_command=(
                    "accelerate launch --config_file=accelerate_config.yaml"
                    f" --num_processes={gpu[i]}"
                    if gpu[i] > 1
                    else "python"
                ),
                script_path=script_path,
                hydra_config=hydra_config,
                experiment_name=experiment_name,
                unique_identifier=unique_identifier,
                run_name=f"{hyphened_name}-{zero_pad(i)}",
                override_args=override_args,
                n_max_parallel=n_max_parallel[i],
                CONTAINER_TAG=container_tag,
                CPU=cpu[i],
                MEMORY=memory[i],
                GPU=gpu[i],
                PRIORITY=priority[i],
                CLUSTER=cluster_list[i],
            )
        )
        for (i, override_args) in enumerate(override_args_list)
        if not skip_runs_mask[i]
    ]

    launch_jobs(
        runs,
        experiment_name=experiment_name,
        only_jobs_with_starting_indices=only_jobs_with_starting_indices,
        use_cluster_storage=use_cluster_storage,
        wandb_mode=wandb_mode,
        dry_run=dry_run,
        skip_git_checks=skip_git_checks,
    )


@functools.cache
def git_repo() -> Repo:
    return Repo(__file__, search_parent_directories=True)


@functools.cache
def git_latest_commit() -> str:
    commit_hash = str(git_repo().head.object.hexsha)
    return commit_hash


@dataclass
class FlamingoRun:
    base_command: str
    script_path: str
    hydra_config: str
    experiment_name: str
    unique_identifier: str
    run_name: str
    override_args: dict
    CLUSTER: str
    n_max_parallel: int = 1
    CONTAINER_TAG: str = DEFAULT_CONTAINER_TAG
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
        away and pass the override args like normal anyway.
        """
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
                "run_name",
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


def get_job_args_from_runs(runs: Sequence[FlamingoRun]) -> dict[str, Union[str, int]]:
    """Combines parameters from parallel runs to get their job's args."""

    # Give the job the max priority and memory among all the runs.
    assert len(runs) > 0
    PRIORITIES = ["low-batch", "normal-batch", "high-batch", "interactive"]
    max_priority = max((run.PRIORITY for run in runs), key=PRIORITIES.index)
    # It makes more sense to sum the memories instead (GH #929) but that's not
    # how our experiments currently specify the memory limits for parallel runs.
    max_memory = max((run.MEMORY for run in runs), key=memory_str_to_bytes)

    combined_args = dataclasses.replace(
        runs[0], PRIORITY=max_priority, MEMORY=max_memory
    ).format_args()

    # Besides args we've explicitly combined, args need to be uniform across all
    # runs in the job.
    EXCLUDED_KEYS = ["PRIORITY", "MEMORY"]
    for run in runs:
        assert {k: v for k, v in combined_args.items() if k not in EXCLUDED_KEYS} == {
            k: v for k, v in run.format_args().items() if k not in EXCLUDED_KEYS
        }
    return combined_args


def create_job_for_multiple_runs(
    runs: Sequence[FlamingoRun],
    name: str,
    index: int,
    launch_id: str,
    project: str,
    entity: str,
    wandb_mode: str,
    use_cluster_storage: bool = True,
) -> str:
    # K8s job/pod names should be short for readability (hence cutting the name).
    unique_identifier = runs[0].unique_identifier
    assert all(run.unique_identifier == unique_identifier for run in runs)
    exp_name_prefix = get_exp_name_prefix(name)
    k8s_name_prefix = f"rllm-{exp_name_prefix}-{unique_identifier}"
    k8s_job_name = (
        f"{k8s_name_prefix}-{zero_pad(index)}"
        if len(runs) == 1
        else f"{k8s_name_prefix}-{zero_pad(index)}-{zero_pad(index+len(runs)-1)}"
    )
    k8s_job_name = k8s_job_name.lower()  # K8s requires lowercase names

    single_commands = []
    for run in runs:
        aux_args = _prepare_override_args(run.override_args)
        split_command = [
            *run.base_command.split(" "),
            run.script_path,
            f"+experiment={run.hydra_config}",
            f"experiment_name={run.experiment_name}",
            f"run_name={run.run_name}",
            *aux_args,
        ]
        single_commands.append(shlex.join(split_command))
    # Use \0 separator instead of \n (with --null) as adding newlines will
    # break when integrated into the K8s YAML template.
    concat_commands = "\\0".join(single_commands)
    num_jobs = len(single_commands)
    # Uses GNU parallel to run all jobs simultaneously
    command = (
        f'echo -ne "{concat_commands}"'
        f" | parallel --line-buffer --null --jobs {num_jobs}"
    )

    combined_run_args = get_job_args_from_runs(runs)
    job = JOB_TEMPLATE.format(
        NAME=k8s_job_name,
        LAUNCH_ID=launch_id,
        WANDB_ENTITY=entity,
        WANDB_PROJECT=project,
        WANDB_MODE=wandb_mode,
        COMMAND=command,
        **combined_run_args,
    )
    if not use_cluster_storage:
        storage_strings_to_remove = [
            (
                "        - name: robust-llm-storage\n"
                "          persistentVolumeClaim:\n"
                "            claimName: st-blobfuse-robust-llm\n"
            ),
            (
                "            - name: robust-llm-storage\n"
                "              mountPath: /robust_llm_data\n"
            ),
        ]
        for storage_string in storage_strings_to_remove:
            assert (
                storage_string in job
            ), f"Expected to find storage string in job: {storage_string}."
            job = job.replace(storage_string, "")

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
    use_cluster_storage: bool = True,
) -> tuple[dict[str, list[str]], str]:
    launch_id = generate_name(style="hyphen")

    jobs_by_cluster: dict[str, list[str]] = defaultdict(list)
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
            # Check that there are actually runs to run
            if len(runs) == 0:
                raise ValueError(
                    "No runs passed the filters. Have you checked "
                    "wandb to see if the run already exists?"
                )

            cluster = runs[0].CLUSTER
            jobs_by_cluster[cluster].append(
                create_job_for_multiple_runs(
                    runs,
                    name,
                    index,
                    launch_id,
                    project,
                    entity,
                    wandb_mode,
                    use_cluster_storage,
                )
            )
        index += len(runs)

    return jobs_by_cluster, launch_id


def launch_jobs(
    runs: Sequence[FlamingoRun],
    project: str = "robust-llm",
    entity: str = "farai",
    wandb_mode: str = "online",
    experiment_name: Optional[str] = None,
    only_jobs_with_starting_indices: Optional[Sequence[int]] = None,
    use_cluster_storage: bool = True,
    dry_run: bool = False,
    skip_git_checks: bool = False,
) -> tuple[dict[str, str], str]:
    """Launch k8s jobs for the given runs.

    Args:
        runs: a list of FlamingoRun objects.
        project: wandb project to use.
        entity: wandb entity to use.
        wandb_mode: Value to give WANDB_MODE environment variable.
        experiment_name: descriptive name of the experiment, used to set wandb group.
        only_jobs_with_starting_indices: if not None, only jobs with starting indices
            contained in this list will be launched. Useful for rerunning a small subset
            of jobs from an experiment.
        use_cluster_storage: Mounts storage to the job if true.
        dry_run: if True, only print the k8s job yaml files without launching them.
        skip_git_checks: if True, skip the remote push and the check for dirty git repo.
            This is useful when running unit tests.

    Returns:
        Tuple containing a dict mapping cluster names to yaml files with k8s
        jobs definitions, and the launch_id.
    """
    repo = git_repo()
    if not skip_git_checks:
        # Check if repo is dirty.
        if repo.is_dirty(untracked_files=True):
            should_continue = ask_for_confirmation(
                "Git repo is dirty. Are you sure you want to continue?"
            )
            if not should_continue:
                print("Aborting")
                sys.exit(1)

        # Push to git as we want to run the code with the current commit.
        repo.remote("origin").push(repo.active_branch.name).raise_if_error()

    filtered_runs = get_unfinished_runs(runs, experiment_name)
    jobs_by_cluster, launch_id = create_jobs(
        filtered_runs,
        project=project,
        entity=entity,
        wandb_mode=wandb_mode,
        experiment_name=experiment_name,
        only_jobs_with_starting_indices=only_jobs_with_starting_indices,
        use_cluster_storage=use_cluster_storage,
    )
    yamls_by_cluster = {
        cluster: "\n\n---\n\n".join(jobs) for cluster, jobs in jobs_by_cluster.items()
    }

    for cluster, yamls in yamls_by_cluster.items():
        print(f"========\nJobs for cluster {cluster}:\n========\n{yamls}")

        if not dry_run:
            print(f"Launching jobs with launch_id={launch_id}...")
            subprocess.run(
                ["kubectl", "--context", cluster, "create", "-f", "-"],
                check=True,
                input=yamls.encode(),
            )
            print(
                "Jobs launched. To delete them run:\n"
                f"kubectl --context {cluster} delete jobs -l launch-id={launch_id}"
            )

    joined_runs = "\n".join(run.run_name for run in filtered_runs)
    print(f"========\nWandb run IDs:\n{joined_runs}\n========\n")
    return yamls_by_cluster, launch_id


def ensure_list(list_or_element: T | list[T], length: int) -> list[T]:
    """Take a list or a single element and return a list of the length."""
    if not isinstance(list_or_element, list):
        list_or_element = [list_or_element] * length

    assert len(list_or_element) == length
    return list_or_element


def fill_cluster_list(cluster: list[str | None]) -> list[str]:
    """Put the current k8s context in place of None in the cluster list."""
    if all(isinstance(c, str) for c in cluster):
        return cluster  # type: ignore  # (we just checked that all are strings)
    current_context = get_current_k8s_context()
    return [current_context if c is None else c for c in cluster]


def zero_pad(number: int | str, length: int = 4) -> str:
    """Pad a number with zeros to the given length."""
    if isinstance(number, str):
        try:
            number = int(number)
        except ValueError:
            raise ValueError(f"Expected an integer or integer string, got '{number}'.")
    return str(number).zfill(length)


def get_wandb_running_finished_runs(experiment_name: str) -> list[str]:
    """Get a list of run names that are 'running' or 'finished' on wandb."""
    runs = wandb_api().runs(
        path="farai/robust-llm",
        filters={"group": experiment_name},
    )
    return [run.name for run in runs if run.state in ["finished", "running"]]


@functools.cache
def wandb_api() -> wandb.Api:
    return wandb.Api()


def generate_chars(length: int = 4, seed: int | None = None) -> str:
    """Generate a random character string."""
    if seed is not None:
        random.seed(seed)
    # just use lower-case letters
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=length))


def get_exp_name_prefix(name: str, n: int = 10) -> str:
    """Try using just the first two parts of the experiment name as the prefix.

    If that doesn't work, use the first n chars. For example:
    - "ian-043-gen-pm-ihateyou-gcg-qwen-base" -> "ian-043"
    - "iansexperimentnumber1" -> "iansexperi"
    """

    try:
        return "-".join(name.split("-")[:2])
    except IndexError:
        return name[:n]


def get_current_k8s_context() -> str:
    """Get the current k8s context."""
    context = subprocess.run(
        ["kubectl", "config", "current-context"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    return context


def get_unfinished_runs(
    runs: Sequence[FlamingoRun], experiment_name: str | None
) -> Sequence[FlamingoRun]:
    # If experiment_name is None, there's no need to filter out completed runs because
    # we haven't provided a wandb group to filter on.
    if experiment_name is None:
        return runs

    # Skip runs that are already finished or still in progress.
    wandb_runs = get_wandb_running_finished_runs(experiment_name)
    runs = [run for run in runs if run.run_name not in wandb_runs]
    return runs
