import pytest
import semver
from datasets import Dataset

from robust_llm.config import DatasetConfig
from robust_llm.rllm_datasets.dataset_utils import (
    filter_empty_rows,
    get_largest_version_below,
)
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset


def test_get_largest_version_below():
    repo_id = "AlignmentResearch/PasswordMatch"
    # Test that we can get the largest version below a given version.
    # We can only test versions that already exist, because otherwise
    # a new larger version could be created and the test would fail.
    largest_below_0_0_4 = get_largest_version_below(repo_id, "0.0.4")
    assert isinstance(largest_below_0_0_4, semver.Version)
    assert largest_below_0_0_4 == "0.0.2"
    assert get_largest_version_below(repo_id, "0.0.3") == "0.0.2"
    assert get_largest_version_below(repo_id, "0.0.0") is None
    assert get_largest_version_below(repo_id, "0.1.0") == "0.0.4"
    assert get_largest_version_below(repo_id, "1.0.0") == "0.1.0"


def test_loading_largest_version_below():
    repo_id = "AlignmentResearch/PasswordMatch"
    cfg = DatasetConfig(
        dataset_type=repo_id,
        n_train=5,
        n_val=5,
        revision="<2.1.1",
    )
    dataset = load_rllm_dataset(cfg, split="validation")
    assert dataset.version == "2.1.0"


def test_failing_largest_version_below():
    repo_id = "AlignmentResearch/PasswordMatch"
    cfg = DatasetConfig(
        dataset_type=repo_id,
        n_train=5,
        n_val=5,
        revision="<0.0.0",
    )
    with pytest.raises(ValueError) as value_error:
        _ = load_rllm_dataset(cfg, split="validation")
    assert "No versions found" in str(value_error.value)


def test_filter_empty_rows():
    dataset = Dataset.from_dict(
        {
            "content": [["a"], [""], ["b"], ["c"], ["", ""]],
        }
    )
    filtered_dataset = filter_empty_rows(dataset)
    assert filtered_dataset["content"] == [["a"], ["b"], ["c"]]
