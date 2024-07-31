"""Script to reformat and save the StrongREJECT dataset.

Dataset from:
https://github.com/alexandrasouly/strongreject/tree/3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset
"""

import datasets
import pandas as pd
from datasets import Dataset

from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)
from robust_llm.utils import ask_for_confirmation

DATASET_REPO_NAME = "AlignmentResearch/StrongREJECT"


def main(minor_version: int, patch_version: int):
    """Create and save the StrongREJECT dataset."""
    df = get_strongreject_dataset()
    ds = process_strongreject_dataset(df)
    # There is no train set, so take an empty slice with the correct columns.
    train_ds = ds.select(range(0))
    val_ds = ds

    ds_dict = datasets.DatasetDict({"train": train_ds, "validation": val_ds})
    ds_dicts = {"default": ds_dict}
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    assert ask_for_confirmation("Are you sure you want to upload the dataset?")
    dataset_uploader.push_to_hub_and_create_tag()


def get_strongreject_dataset() -> pd.DataFrame:
    """Load the StrongREJECT dataset from github."""

    STRONGREJECT_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_dataset.csv"  # noqa: E501
    df = pd.read_csv(STRONGREJECT_URL)
    return df


def process_strongreject_dataset(df: pd.DataFrame) -> Dataset:
    """Add new columns to a single StrongREJECT datasets.dataset."""

    # We only need to grab the "forbidden_prompt" column, not the 'source' or
    # 'category'.
    text = df["forbidden_prompt"].tolist()

    INSTRUCTIONS = ""  # The examples are self-explanatory.
    ANSWER_PROMPT = "\n\nAnswer:"

    # We don't have clf_labels because this is a generation-only task,
    # so we make the ClassLabel feature empty.
    label_feature = datasets.ClassLabel(num_classes=0)
    clf_label = [None] * len(text)
    proxy_clf_label = [None] * len(text)
    text_feature = datasets.Value("string")
    features = datasets.Features(
        {
            "text": text_feature,
            "clf_label": label_feature,
            "proxy_clf_label": label_feature,
        }
    )

    ds = Dataset.from_dict(
        {"text": text, "clf_label": clf_label, "proxy_clf_label": proxy_clf_label},
        features=features,
    )

    ds = ds.map(
        lambda x: {
            "instructions": INSTRUCTIONS,
            "content": [x["text"]],
            "answer_prompt": ANSWER_PROMPT,
            # We use a generic target that GCG can optimize for.
            # Note the leading space, which can be stripped from DatasetConfig.
            "gen_target": " I'm sorry, as an AI language model I can't help you with that.",  # noqa: E501
            "proxy_gen_target": " Sure, here",
        },
        remove_columns=["text"],
    )
    return ds


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 1
    PATCH_VERSION = 0
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
