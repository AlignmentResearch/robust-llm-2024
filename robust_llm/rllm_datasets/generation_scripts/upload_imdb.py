"""Script to generate the IMDB dataset"""

from robust_llm.rllm_datasets.dataset_utils import prep_huggingface_dataset
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/IMDB"


def main(minor_version: int, patch_version: int):
    """Create and save the IMDB dataset.

    We use the generic prep_huggingface_dataset function to prepare the dataset,
    which assumes it's binary classification and that the columns `text` and
    `label` exist.
    """
    ds_dicts = prep_huggingface_dataset("imdb")
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 4
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
