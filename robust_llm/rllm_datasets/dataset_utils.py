"""Utilities used in the generation of all datasets."""

import sys
from dataclasses import dataclass, fields
from typing import Callable, Optional

import datasets
import huggingface_hub as hf_hub
import semver
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from robust_llm import logger
from robust_llm.utils import ask_for_confirmation

DS_SHUFFLE_SEED = 0


@dataclass
class RLLMExample:
    """Represents a single example in a RLLMDataset."""

    instructions: str
    content: list[str]
    answer_prompt: str
    # clf_label and gen_target are the correct responses we want to evaluate
    # against.
    # proxy_clf_label and proxy_gen_target are the bad responses we want to
    # optimize towards.
    clf_label: int
    proxy_clf_label: int
    gen_target: str
    proxy_gen_target: str


# Get the fields of a dataclass: https://stackoverflow.com/a/66499324
EXPECTED_COLUMNS = {f.name for f in fields(RLLMExample)}

# Contains one model from each family of models we care (or might
# care) about for use in filtering to appropriate context lengths
SUPPORTED_MODELS = [
    "stanford-crfm/alias-gpt2-small-x21",
    "EleutherAI/pythia-14m",
    # To use llama you need to request access on hf and export HF_TOKEN
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen1.5-0.5B",
]


def filter_dataset_length(ds: Dataset) -> Dataset:
    """Filter dataset for rows with length zero or greater than the context length.

    Args:
        ds: The dataset to filter.

    Returns:
        The filtered dataset.
    """
    return filter_dataset_for_context_length(filter_empty_rows(ds))


def filter_empty_rows(ds: Dataset) -> Dataset:
    """Filter out rows with no data in the 'content' column."""
    prev_len = len(ds)
    new_ds = ds.filter(lambda x: len("".join(x["content"])) > 0)
    if len(new_ds) < prev_len:
        logger.warning(f"Filtered out {prev_len - len(new_ds)} rows with no content")
    else:
        logger.debug("No empty rows found")
    return new_ds


def filter_dataset_for_context_length(dataset: Dataset, buffer: int = 24) -> Dataset:
    """Filter out rows that are too long for all supported models.

    Args:
        dataset: The dataset to filter.
        buffer: The number of additional tokens to remove to leave space
            for tokens added by attacks. The default of 24 is not special,
            it's just gives a reasonable amount of space for most attacks.


    Returns:
        The filtered dataset, i.e. the provided dataset *minus* all examples
        that do not fit in the context length of all supported models.
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
    """Filter for examples that fit in the context length of the given model.

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

    def tokenize(x):
        return tokenizer(x["text"])["input_ids"]

    # Add a default text column to the dataset to use for filtering.
    dataset = dataset.map(
        lambda x: {"text": example_dict_to_text(x)},
    )
    tokenized_dataset = dataset.map(
        lambda x: {"input_ids": tokenize(x)},
        batched=True,
    )
    filtered_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= token_target
    )
    # Remove the columns that were added for filtering.
    return filtered_dataset.remove_columns(["input_ids", "text"])


def example_dict_to_text(example: dict) -> str:
    """Default way to convert a dataset example to a single string.

    Note that we might want more complicated ways to do this for chat models
    that have different roles for different messages.
    """
    return "".join(
        (example["instructions"], *example["content"], example["answer_prompt"])
    )


def example_dict_to_chunked_text(example: dict) -> list[str]:
    """Default way to convert a dataset example to a 'chunked_text' list.

    Note that we might want more complicated ways to do this for chat models
    that have different roles for different messages.
    """
    return [example["instructions"], *example["content"], example["answer_prompt"]]


def construct_text_and_chunked_text(ds: Dataset) -> Dataset:
    """Construct a dataset with both text and chunked_text columns."""
    ds = ds.map(
        lambda x: {
            "text": example_dict_to_text(x),
            "chunked_text": example_dict_to_chunked_text(x),
        },
    )
    return ds


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
    one positive and one negative example.

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


def prepare_huggingface_dataset(
    repo_id: str,
    ds_specific_callback: Callable[[Dataset], Dataset],
    split_map: Optional[dict[str, str]] = None,
    **kwargs,
) -> dict[str, DatasetDict]:
    """Prepare a huggingface dataset for use in RLLMDatasets.

    Most huggingface datasets have a similar structure. This function
    attempts to apply all the relevant processing to a binary classification
    dataset to make it ready for use in RLLMDatasets.

    Args:
        repo_id: The id of the dataset on the hub.
        ds_specific_callback: A function specific to the dataset that processes
            it to have the necessary columns and structure for RLLMDatasets.
            Currently, this means adding 'instructions', 'content', 'answer_prompt',
            and 'gen_target' columns.
        split_map: A mapping from the split names in the dataset to the
            names we want to use in RLLMDatasets. If None, uses a
            default split map.
        kwargs: Additional keyword arguments to pass to the huggingface
            datasets.load_dataset function.

    Returns:
        A dictionary of (config_name: DatasetDict) pairs.
    """
    if split_map is None:
        split_map = {
            "train": "train",
            "validation": "test",
        }
    train, val = datasets.load_dataset(
        repo_id,
        split=[split_map["train"], split_map["validation"]],  # type: ignore
        **kwargs,
    )
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    # make sure it has the necessary columns
    prepped_train = prep_hf_split(train)
    prepped_val = prep_hf_split(val)

    processed_train = ds_specific_callback(prepped_train)
    processed_val = ds_specific_callback(prepped_val)

    filtered_train = filter_dataset_length(processed_train)
    filtered_val = filter_dataset_length(processed_val)

    full_ds_dict = DatasetDict({"train": filtered_train, "validation": filtered_val})
    pos_ds_dict, neg_ds_dict = make_pos_neg_versions(full_ds_dict)
    out_dict = {
        "default": full_ds_dict,
        "pos": pos_ds_dict,
        "neg": neg_ds_dict,
    }
    return out_dict


def prep_hf_split(ds: Dataset) -> Dataset:
    """Process a huggingface dataset split for use in RLLMDatasets."""
    if "text" in ds.column_names and "label" in ds.column_names:
        num_classes = _get_num_classes(ds)
        assert num_classes == 2, (
            "This class can't automatically create an `RLLMDataset`-compatible dataset "
            " from huggingface datasets with more than two classes. You can still do it"
            " manually if you want."
        )
        assert set(ds["label"]) == {0, 1}, "labels must be 0 and 1"
        # Classification target should be called clf_label.
        ds = ds.rename_column("label", "clf_label")
        # Drop all columns except the ones we need.
        HF_EXPECTED_COLUMNS = ["text", "clf_label"]
        ds = ds.remove_columns(
            [col for col in ds.column_names if col not in HF_EXPECTED_COLUMNS]
        )
    else:
        # The only other dataset format is used by Anthropic's Helpfulness and
        # Harmlessness dataset, which has 'chosen' and 'rejected' columns.
        assert ds.column_names == ["chosen", "rejected"]
    # Shuffle deterministically.
    ds = ds.shuffle(seed=DS_SHUFFLE_SEED)
    return ds


def _get_num_classes(ds: Dataset) -> int:
    """Get the number of classes in a dataset."""
    try:
        return ds.features["label"].num_classes
    except AttributeError:
        logger.warning(
            "'label' column does not have num_classes attribute."
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


def cast_column_to_feature(
    ds: Dataset,
    column_name: str,
    feature: datasets.ClassLabel | datasets.Value,
) -> Dataset:
    """Cast a column in a dataset to a given feature."""
    new_features = ds.features.copy()
    new_features[column_name] = feature
    return ds.cast(new_features)


def cast_and_concatenate(
    ds: datasets.Dataset, other_ds: datasets.Dataset
) -> datasets.Dataset:
    """Cast the second dataset to the features of the first dataset and concatenate.

    Args:
        ds: The first dataset, whose features will be used.
        other_ds: The second dataset, whose features will be updated.
    """
    if ds.features != other_ds.features:
        other_ds = cast_features_like(ds, other_ds)
    new_ds = datasets.concatenate_datasets([ds, other_ds])
    return new_ds


def cast_features_like(
    ds: datasets.Dataset, other_ds: datasets.Dataset
) -> datasets.Dataset:
    """Cast the second dataset to the features of the first dataset.

    Args:
        ds: The dataset with features we want to copy.
        other_ds: The dataset to cast to those features.
    """
    new_ds = other_ds.cast(ds.features)
    return new_ds


def strip_leading_whitespace(ds: Dataset) -> Dataset:
    """Strip leading whitespace from the 'gen_target' column of a dataset.

    Also updates the Feature of the 'clf_label' column to remove leading whitespace.
    """
    ds = ds.map(
        lambda x: {
            "gen_target": x["gen_target"].lstrip(),
        }
    )
    # Also update the feature of clf_label
    stripped_feature = datasets.ClassLabel(
        names=[name.lstrip() for name in ds.features["clf_label"].names]
    )
    ds = cast_column_to_feature(
        ds=ds,
        column_name="clf_label",
        feature=stripped_feature,
    )
    return ds
