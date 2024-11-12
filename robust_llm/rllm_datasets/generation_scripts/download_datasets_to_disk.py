"""Downloads datasets from HF to /robust_llm_data/datasets."""

import logging
import shutil
from pathlib import Path

import datasets
from huggingface_hub import HfApi
from tqdm import tqdm

from robust_llm.config.constants import SHARED_DATA_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

datasets.utils.logging.disable_progress_bar()


def delete_failed_download(dataset_dir: Path) -> None:
    """Deletes a download directory that did not complete.

    Raises:
        Exception: If the directory cannot be deleted.
    """
    try:
        shutil.rmtree(dataset_dir)
    except Exception as e:
        logger.error(
            f"Error deleting failed download {dataset_dir}: {str(e)}"
            "\nIs the filesystem down?"
            " You will need to manually delete this directory."
        )
        raise


def download_one_dataset_revision(
    dataset_name: str, revision: str, base_save_path: Path
) -> bool:
    """Downloads a specific revision of a dataset, returns False if download fails.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        revision: Git revision/tag
        base_save_path: Base directory to save datasets

    Raises:
        KeyboardInterrupt: If download was interrupted by keyboard interrupt.
    """
    dataset_dir = base_save_path / dataset_name / revision
    if dataset_dir.exists():
        logger.info(f"Revision {dataset_name}:{revision} already exists, skipping")
        return True
    dataset_dir.mkdir(parents=True)

    try:
        splits = datasets.get_dataset_split_names(dataset_name, revision=revision)
        for split in splits:
            split_destination = dataset_dir / split
            logger.info(
                f"Downloading {dataset_name}:{revision}:{split} to {split_destination}"
            )
            dataset = datasets.load_dataset(
                dataset_name, split=split, revision=revision
            )
            assert isinstance(dataset, datasets.Dataset)
            dataset.save_to_disk(split_destination)
        return True

    except Exception as e:
        logger.error(f"Failed to download {dataset_name}:{revision}: {str(e)}")
        delete_failed_download(dataset_dir)
        return False

    except KeyboardInterrupt:
        delete_failed_download(dataset_dir)
        raise


def download_dataset_revisions(dataset_name: str, base_save_path: Path) -> bool:
    """Downloads all revisions of a dataset, returns False if any download fails.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        base_save_path: Base directory to save datasets

    Raises:
        ValueError: If no revisions are found for the dataset.
    """
    try:
        api = HfApi()
        refs = api.list_repo_refs(dataset_name, repo_type="dataset")
    except Exception as e:
        logger.error(f"Error getting revisions for {dataset_name}: {str(e)}")
        return False

    revisions = [ref.name for ref in refs.tags]
    if not revisions:
        raise ValueError(f"No revisions found for dataset: {dataset_name}")

    logger.info(f"Found {len(revisions)} revisions for {dataset_name}: {revisions}")
    return all(
        download_one_dataset_revision(dataset_name, revision, base_save_path)
        for revision in revisions
    )


def main() -> None:
    SHARED_DATASET_DIR = Path(SHARED_DATA_DIR) / "datasets"
    DATASETS = [
        "AlignmentResearch/WordLength",
        "AlignmentResearch/IMDB",
        "AlignmentResearch/PasswordMatch",
        "AlignmentResearch/EnronSpam",
        "AlignmentResearch/Helpful",
        "AlignmentResearch/Harmless",
    ]

    success = all(
        download_dataset_revisions(dataset_name, SHARED_DATASET_DIR)
        for dataset_name in tqdm(DATASETS, desc="Downloading datasets")
    )
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
