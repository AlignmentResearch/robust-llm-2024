# Runs a batch job on k8s using the code from the current commit.

import argparse

from robust_llm.batch_job_utils import FlamingoRun, launch_jobs


def main(args):
    run = FlamingoRun(
        base_command="python",
        script_path=args.script_path,
        hydra_config=args.hydra_config,
        override_args={
            "experiment_name": args.experiment_name,
            "job_type": args.job_type,
        },
        CONTAINER_TAG=args.container_tag,
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
