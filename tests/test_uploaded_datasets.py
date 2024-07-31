import pytest

from robust_llm.config.configs import DatasetConfig
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset

UPLOADED_DATASETS = [
    "AlignmentResearch/PasswordMatch",
    "AlignmentResearch/EnronSpam",
    "AlignmentResearch/WordLength",
    "AlignmentResearch/IMDB",
    "AlignmentResearch/StrongREJECT",
    "AlignmentResearch/Helpful",
    "AlignmentResearch/Harmless",
]


@pytest.mark.parametrize("repo_id", UPLOADED_DATASETS)
def test_uploaded_datasets(repo_id: str):
    """Test all datasets on the hub.

    We make the conscious decision to only test the default version
    because we can't go back and fix old versions.
    """
    config = DatasetConfig(
        dataset_type=repo_id,
        revision="main",
        n_train=5,
        n_val=5,
    )
    dataset = load_rllm_dataset(config, split="validation")
    assert len(dataset.ds) == 5
    for example in dataset.ds:
        # Make sure all expected columns are present
        assert isinstance(example, dict)
        assert "text" in example
        assert "chunked_text" in example
        assert "clf_label" in example
        text = example["text"]
        chunked_text = example["chunked_text"]
        clf_label = example["clf_label"]
        assert isinstance(text, str)
        assert isinstance(chunked_text, list)
        if repo_id == "AlignmentResearch/StrongREJECT":
            # Empty clf_label is OK for generation-only tasks like StrongREJECT.
            assert clf_label is None
        else:
            assert isinstance(clf_label, int)

        # Make sure the text is not empty and lines up
        # with the chunked_text.
        assert len(text) > 0
        assert len(chunked_text) > 0
        assert text == "".join(chunked_text)
        # Make sure the clf_label is valid.
        print("Before:", example)
        example = dataset.update_example_based_on_text(example)
        print("After:", example)
        assert example["clf_label"] == clf_label
