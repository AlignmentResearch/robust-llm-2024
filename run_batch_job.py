# Runs a batch job on k8s using the code from the current commit.

import argparse

from robust_llm.batch_job_utils import FlamingoRun, launch_jobs


def main(args):
    override_args = {}
    assert args.experiment_name is not None
    if args.job_type is not None:
        override_args["job_type"] = args.job_type

    run = FlamingoRun(
        base_command="python",
        experiment_name=args.experiment_name,
        unique_identifier=args.unique_identifier,
        script_path=args.script_path,
        hydra_config=args.hydra_config,
        override_args=override_args,
        run_name=args.run_name,
        CONTAINER_TAG=args.container_tag,
        CLUSTER=args.cluster,
        CPU=args.cpu,
        MEMORY=args.memory,
        GPU=args.gpu,
        PRIORITY=args.priority,
    )

    launch_jobs([run], dry_run=args.dryrun, experiment_name=args.experiment_name)


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
    parser.add_argument("--run_name", type=str, help="run_name for wandb")
    parser.add_argument(
        "--cluster",
        type=str,
        help="cluster to run on, default a6k",
        default="a6k",
    )
    parser.add_argument(
        "--unique_identifier",
        type=str,
        help="string to be included in k8s job names",
        default="aaaa",
    )
    parser.add_argument("--job_type", type=str, help="job_type for wandb")
    parser.add_argument(
        "--container_tag",
        type=str,
        help="container tag to use",
        default="2024-08-07-backoff",
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
