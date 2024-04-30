"""Utilities used in the generation of all datasets."""

import sys
from dataclasses import dataclass
from typing import Optional

import datasets
import huggingface_hub as hf_hub
import semver
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from robust_llm.utils import ask_for_confirmation

DS_SHUFFLE_SEED = 0


@dataclass
class RLLMExample:
    """Represents a single example in a RLLMDataset.

    TODO (ian): Add fields for generative tasks.
    """

    text: str
    chunked_text: list[str]
    clf_label: int

    def to_dict(self):
        return {
            "text": self.text,
            "chunked_text": self.chunked_text,
            "clf_label": self.clf_label,
        }


# Contains one model from each family of models we care (or might
# care) about for use in filtering to appropriate context lengths
SUPPORTED_MODELS = [
    "stanford-crfm/alias-gpt2-small-x21",
    "EleutherAI/pythia-14m",
    # To use llama you need to request access on hf and export HF_TOKEN
    "meta-llama/Llama-2-7b-hf",
]


def filter_dataset_length(dataset: Dataset, buffer: int = 24) -> Dataset:
    """Filter a dataset to only include examples that fit in the context
    length of all supported models.

    Args:
        dataset: The dataset to filter.
        buffer: The number of additional tokens to remove to leave space
            for tokens added by attacks. The default of 24 is not special,
            it's just gives a reasonable amount of space for most attacks.


    Returns:
        The filtered dataset.
    """
    for model_name in SUPPORTED_MODELS:
        dataset = filter_length_for_model(
            dataset=dataset, model_name=model_name, buffer=buffer
        )
    return dataset


def filter_length_for_model(
    dataset: Dataset,
    model_name: str,
    buffer: int,
) -> Dataset:
    """Filter a dataset to only include examples that fit in the context
    length of the given model.

    Args:
        dataset: The dataset to filter.
        model_name: The model whose tokenizer and context length we are
            filtering for.
        buffer: The number of additional tokens to remove to leave space
            for tokens added by attacks.

    Returns:
        The dataset filtered to fit in the context length of the model.
    """
    context_length = _get_context_length(model_name)
    token_target = context_length - buffer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(
        lambda x: {"input_ids": tokenizer(x["text"])["input_ids"]},
        batched=True,
    )
    filtered_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= token_target
    )
    return filtered_dataset.remove_columns("input_ids")


def _get_context_length(model_name: str) -> int:
    """Get the context length of a model.

    It turns out that the easiest way to get the context length is
    the max_position_embeddings attribute of the model config.
    """
    config = AutoConfig.from_pretrained(model_name)
    return config.max_position_embeddings


def create_rllm_tag(repo_id: str, tag: semver.Version):
    """Create a tag for an rllm dataset on the huggingface hub."""

    hf_hub.create_tag(repo_id=repo_id, tag=str(tag), repo_type="dataset")


def valid_tag(tag: str):
    """Check if a tag is a valid semver tag."""
    try:
        _ = semver.Version.parse(tag)
    except ValueError:
        return False
    return True


def version_exists(repo_id: str, tag: semver.Version):
    """Check if a version exists on the hub repo as a tag.

    Dataset versions are stored as tags on the huggingface repo for the
    datasets. This checks if, for a given version number of the dataset, there
    is already a tag with that version number.
    """
    # If we can't load the gitref, then either we don't have access or the repo
    # doesn't exist but we should continue anyway, and let the push fail if it's
    # a permissions issue
    try:
        existing_tags = _get_versions(repo_id)
    except ValueError:
        return False

    for existing_tag in existing_tags:
        if existing_tag == tag:
            return True
    return False


def get_largest_version(repo_id: str) -> semver.Version | None:
    """Get the largest version of a dataset on the hub."""
    try:
        existing_tags = _get_versions(repo_id)
    except ValueError:
        return None

    if len(existing_tags) == 0:
        return None

    largest_version = semver.Version.parse(existing_tags[0])
    for existing_tag in existing_tags:
        parsed_tag = semver.Version.parse(existing_tag)
        if parsed_tag > largest_version:
            largest_version = parsed_tag
    return largest_version


def get_largest_version_below(repo_id: str, version: str) -> semver.Version | None:
    """Get the largest version of a dataset strictly less than a given version.

    Args:
        repo_id: The id of the dataset repo on the hf hub.
        version: A valid semver version number as a string.

    Returns:
        The largest version of the dataset that is strictly less than the given
        version. If no such version exists, returns None.

    """
    try:
        existing_tags = _get_versions(repo_id)
    except ValueError:
        return None

    if len(existing_tags) == 0:
        return None

    parsed_version_upper_bound = semver.Version.parse(version)

    largest_version = None
    for existing_tag in existing_tags:
        parsed_tag = semver.Version.parse(existing_tag)

        if parsed_tag >= parsed_version_upper_bound:
            continue
        if largest_version is None or parsed_tag > largest_version:
            largest_version = parsed_tag

    return largest_version


def _get_versions(repo_id: str) -> list[str]:
    """Get the versions of a dataset on the hub.

    Args:
        repo_id: The id of the dataset repo on the hf hub.

    Returns:
        A list of the version (tag) names of the dataset.

    Raises:
        ValueError: If the repo does not exist.
    """
    try:
        gitref = hf_hub.list_repo_refs(repo_id, repo_type="dataset")
    except hf_hub.utils._errors.RepositoryNotFoundError:
        raise ValueError(
            f"Repo {repo_id} does not exist or you do not have permission to view it."
        )
    return [tag.name for tag in gitref.tags]


def maybe_abort_for_larger_version(repo_name: str, version: semver.Version):
    """Check if a larger version of a dataset already exists on the hub.

    The reason we check this is in case it's unintential that we're uploading
    a version that's smaller than the largest version on the hub. However, if
    we're e.g. fixing bugs in a dataset for older formats, we might want to create
    the dataset anyway.
    """
    largest_version = get_largest_version(repo_name)
    if (largest_version is not None) and (largest_version > version):
        should_continue = ask_for_confirmation(
            f"Larger version {largest_version} of {repo_name} already exists. Continue?"
        )
        if not should_continue:
            print("Aborting")
            sys.exit(1)


def extract_single_label(ds: Dataset, label: int):
    return ds.filter(lambda x: x["clf_label"] == label)


def make_pos_neg_versions(ds_dict: DatasetDict) -> tuple[DatasetDict, DatasetDict]:
    """Make versions of the dataset with only positive and only negative examples.

    Assumes that the dataset has a "clf_label" column with 1 for positive
    examples and 0 for negative examples, and that each split has at least
    positive and one negative example.

    Args:
        ds_dict: The dataset to split into positive and negative examples.
            Must have 'train' and 'validation' splits.
    """
    # preconditions
    assert set(ds_dict.keys()) == {"train", "validation"}
    assert "clf_label" in ds_dict["train"].column_names
    assert "clf_label" in ds_dict["validation"].column_names
    assert set(ds_dict["train"]["clf_label"]) == {0, 1}
    assert set(ds_dict["validation"]["clf_label"]) == {0, 1}

    train, val = ds_dict["train"], ds_dict["validation"]
    pos_train = extract_single_label(ds=train, label=1)
    pos_val = extract_single_label(ds=val, label=1)
    neg_train = extract_single_label(ds=train, label=0)
    neg_val = extract_single_label(ds=val, label=0)

    # postconditions
    assert len(pos_train) > 0
    assert len(pos_val) > 0
    assert len(neg_train) > 0
    assert len(neg_val) > 0
    assert len(pos_train) + len(neg_train) == len(train)
    assert len(pos_val) + len(neg_val) == len(val)

    pos_dict = DatasetDict({"train": pos_train, "validation": pos_val})
    neg_dict = DatasetDict({"train": neg_train, "validation": neg_val})
    return pos_dict, neg_dict


def prep_huggingface_dataset(
    repo_id: str,
    split_map: Optional[dict[str, str]] = None,
) -> dict[str, DatasetDict]:
    """Prepare a huggingface dataset for use in RLLMDatasets.

    Most huggingface datasets have a similar structure. This function
    attempts to apply all the relevant processing to a binary classification
    dataset to make it ready for use in RLLMDatasets.

    Args:
        repo_id: The id of the dataset on the hub.
        split_map: A mapping from the split names in the dataset to the
            names we want to use in RLLMDatasets. If None, uses a
            default split map.

    Returns:
        A dictionary of (config_name: DatasetDict) pairs.
    """
    if split_map is None:
        split_map = {
            "train": "train",
            "validation": "test",
        }
    train, val = datasets.load_dataset(
        repo_id, split=[split_map["train"], split_map["validation"]]  # type: ignore
    )
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    # make sure it has the necessary columns
    processed_train = process_hf_split(train)
    processed_val = process_hf_split(val)

    full_ds_dict = DatasetDict({"train": processed_train, "validation": processed_val})
    pos_ds_dict, neg_ds_dict = make_pos_neg_versions(full_ds_dict)
    out_dict = {
        "default": full_ds_dict,
        "pos": pos_ds_dict,
        "neg": neg_ds_dict,
    }
    return out_dict


def process_hf_split(ds: Dataset) -> Dataset:
    """Process a huggingface dataset split for use in RLLMDatasets."""
    assert {"text", "label"} <= set(ds.column_names)

    num_classes = _get_num_classes(ds)
    assert num_classes == 2, (
        "This class can't automatically create `RLLMDataset`-compatible from"
        " huggingface datasets with more than two classes. You can still do it"
        " manually if you want."
    )
    assert set(ds["label"]) == {0, 1}, "labels must be 0 and 1"
    ds = filter_dataset_length(ds)
    # shuffle deterministically
    ds = ds.shuffle(seed=DS_SHUFFLE_SEED)
    ds = ds.map(
        # assume that we only have one chunk
        lambda x: {"chunked_text": [x["text"]]},
    )
    # classification target should be called clf_label
    ds = ds.rename_column("label", "clf_label")
    # drop all columns except the ones we need
    # TODO (ian): work out where to put these so they aren't
    # duplicated with DatasetUploadHandler
    EXPECTED_COLUMNS = ["text", "chunked_text", "clf_label"]
    ds = ds.remove_columns(
        [col for col in ds.column_names if col not in EXPECTED_COLUMNS]
    )
    return ds


def _get_num_classes(ds: Dataset) -> int:
    """Get the number of classes in a dataset."""
    try:
        return ds.features["label"].num_classes
    except AttributeError:
        print(
            "WARNING: 'label' column does not have num_classes attribute."
            " Using `set()` to get a lower bound on the number of classes."
        )
        return len(set(ds["label"]))


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    padding: str = "do_not_pad",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
    column_name: str = "text",
) -> Dataset:
    # Explicitly passing in padding argument seems necessary to avoid an error
    # (even if it's 'do_not_pad').
    # TODO (GH#330): Work out if passing `do_not_pad` is actually necessary
    def tokenizer_fn(x):
        return tokenizer(
            x[column_name],
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    tokenized_dataset = dataset.map(tokenizer_fn, batched=True)
    return tokenized_dataset
