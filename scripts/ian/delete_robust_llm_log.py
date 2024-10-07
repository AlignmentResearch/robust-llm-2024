import concurrent.futures
import threading
from datetime import datetime
from functools import partial
from pathlib import Path

import wandb

LOG_DIR = Path("/robust_llm_data/ian/deleting_robust_llm_logs")
FILE_LOCKS = {
    "deleted.csv": threading.Lock(),
    "processed.csv": threading.Lock(),
    "error.csv": threading.Lock(),
}
PROCESSED_FILE = LOG_DIR / "processed.csv"
DELETED_FILE = LOG_DIR / "deleted.csv"
ERROR_FILE = LOG_DIR / "error.csv"
WANDB_PROJECT_PATH = "farai/robust-llm"


def append_to_file(file_path: Path, content: str):
    with FILE_LOCKS[file_path.name]:
        with open(file_path, "a") as f:
            f.write(content)


def get_already_processed_run_ids():
    with FILE_LOCKS[PROCESSED_FILE.name]:
        if not PROCESSED_FILE.exists():
            return set()
        with open(PROCESSED_FILE, "r") as f:
            return {line.split(",")[1] for line in f}


def process_run(run, processed_run_ids: set[str]):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if run.id in processed_run_ids:
        return

    for file in run.files():
        if file.name == "robust_llm.log":
            try:
                file_size_gb = file.size / 1e9
                file.delete()
                append_to_file(
                    file_path=DELETED_FILE,
                    content=f"{time},{run.id},{run.name},{file_size_gb}\n",
                )
                break
            except Exception as e:
                print(f"Error deleting file for run {run.id}: {e}")
                append_to_file(
                    file_path=ERROR_FILE,
                    content=f"{time},{run.id},{run.name},{e}\n",
                )

    append_to_file(
        file_path=PROCESSED_FILE,
        content=f"{time},{run.id},{run.name}\n",
    )


def delete_robust_llm_log_for_runs(runs, max_workers: int):
    processed_run_ids = get_already_processed_run_ids()

    process_run_with_processed_run_ids = partial(
        process_run, processed_run_ids=processed_run_ids
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_run_with_processed_run_ids, runs)


def main(max_workers: int = 10):
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT_PATH)
    delete_robust_llm_log_for_runs(runs, max_workers=max_workers)


if __name__ == "__main__":
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    main(max_workers=10)
