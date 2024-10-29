#!/bin/env python
"""Run and profile a Python job using py-spy.

If the Python job needs to use accelerate, run `accelerate launch` on this
script. A few remarks on this:
- It would be more natural to call `accelerate launch` directly on the job, then
  use `py-spy record --pid <pid>` to profile the launched processes. But on
  Kubernetes --pid requires enabling SYS_PTRACE
  (https://github.com/benfred/py-spy?tab=readme-ov-file#how-do-i-run-py-spy-in-kubernetes),
  and Flamingo's pod security settings
  ("baseline" in
  https://kubernetes.io/docs/concepts/security/pod-security-standards/) do not
  allow this.
- Running py-spy (even with the --subprocesses flag) on `accelerate launch` does
  not work, it just profiles the Python script implementing the `accelerate`
  CLI.
- Annoyingly, we need to avoid using the accelerate library inside this script
  because it tends to cause a mysterious accelerate error either in the launched
  job or in this script (whichever uses accelerate second):
      torch.distributed.DistBackendError: NCCL error in:
        ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1691, unhandled system
        error (run with NCCL_DEBUG=INFO for details), NCCL version 2.19.3
      ncclSystemError: System call (e.g. socket, malloc) or external library call
        failed or device error.
      Last error: socketStartConnect: Connect to 10.233.70.50<48425> failed : Software
        caused connection abort

Profiles are saved to disk. The other obvious alternative is to upload them as
artifacts to W&B, but this has issues:
- A straightforward attempt to add an artifact to a completed run using the W&B
  API gives an error. The suggested way to do this is to resume the run:
  https://docs.wandb.ai/guides/artifacts/artifacts-faqs/#how-do-i-log-an-artifact-to-an-existing-run
  However this overwrites some info of the existing run. For example,
  https://wandb.ai/farai/robust-llm/runs/66gtgd5y/overview
  is a run from June 2024 that was resumed in October 2024 to add an artifact. Now
  the "Start time" is October, and the "Hostname" is "MacBookPro".
- Another way is to create a new run, add the artifact to that run, and then we
  can link the artifact the finished main job run. But it's convoluted to have a
  100% overhead of dummy runs that are only there to hold artifacts, plus W&B
  storage is likely more expensive than disk storage.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_with_profiling(
    save_directory: Path, sampling_rate: int, command: list[str]
) -> None:
    # We use speedscope (out of the possible output formats [flamegraph,
    # speedscope, raw]) because it seems to take the least disk space.
    PROFILE_FORMAT = "speedscope"
    save_directory.mkdir(parents=True, exist_ok=True)
    # We'll rename this file, so we don't auto-delete it (the auto-delete would
    # throw an exception).
    profile_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    # We want to use the W&B run name and ID in the profile's final filename.
    # The job will write them into wandb_info_file.
    # Note this file is different for each accelerate process. Really we should
    # just have the main process generate this file, and then we'd gather either
    # the filename or the contents of the file. However, as noted in the
    # docstring, we need to avoid the accelerate library in this file, which
    # precludes gathering.
    wandb_info_file = tempfile.NamedTemporaryFile(suffix=".json")
    rank = os.environ.get("RANK", "0")  # Populated by accelerate

    full_command = (
        [
            "py-spy",
            "record",
            "--rate",
            str(sampling_rate),
            "--output",
            profile_file.name,
            "--format",
            PROFILE_FORMAT,
            "--idle",
            "--",
        ]
        + command
        + [f"environment.wandb_info_filename={wandb_info_file.name}"]
    )
    if platform.system() == "Darwin":
        print("py-spy needs sudo on macOS. You may be prompted for your password.")
        full_command = ["sudo", "--preserve-env"] + full_command

    try:
        subprocess.run(full_command, check=True)

        # Rename the profiling data to contain W&B name. Path.rename() doesn't
        # work across filesystems, so we use shutil.move instead.
        wandb_info = json.loads(wandb_info_file.read())
        new_filename = save_directory / (
            f"{wandb_info['wandb_run_name']}-{wandb_info['wandb_run_id']}"
            f"-process{rank}.{PROFILE_FORMAT}.json"
        )
        shutil.move(profile_file.name, new_filename)

        print(f"Profiling data saved to: {new_filename}")
    finally:
        # Maybe we got interrupted or errored, in which case we get rid of the
        # profiling data temporary file.
        temp_file = Path(profile_file.name)
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run and profile a Python job using py-spy.

If the Python job needs to use accelerate, run `accelerate launch` on this
script.""",
        epilog="Specify the command to profile by passing it after '--'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-directory",
        type=Path,
        default=Path("/robust_llm_data/profiles/"),
        help="Directory to save the profiling data",
    )
    parser.add_argument(
        "--rate",
        type=int,
        # Default py-spy value is 100, but we reduce it drastically to reduce
        # the disk space of the profile.
        default=1,
        help="The sampling rate for py-spy in samples per second.",
    )
    default_usage = parser.format_usage().strip().removeprefix("usage: ")
    parser.usage = f"{default_usage} -- [command_to_profile]"

    known_args, unknown_args = parser.parse_known_args()
    if len(unknown_args) < 2 or unknown_args[0] != "--":
        parser.error(
            "No command provided to run and profile."
            " Use -- to separate script arguments from the command to profile."
            f" Detected args: {unknown_args}",
        )
    command = unknown_args[1:]
    run_with_profiling(known_args.save_directory, known_args.rate, command)
