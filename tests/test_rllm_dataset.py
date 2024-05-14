import pytest

from robust_llm.config.configs import DatasetConfig
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@pytest.fixture()
def dataset() -> RLLMDataset:
    """Fixture for the PasswordMatch dataset.

    We use the 'pos' version of the dataset with all positive examples so we know
    in advance that the `clf_label`s should be 1 and we can easily flip them by
    inserting any other word.
    """

    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        n_train=5,
        n_val=5,
        config_name="pos",
    )
    dataset = load_rllm_dataset(cfg, split="validation")
    return dataset


@pytest.fixture()
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def test_initialized_rllm_dataset(dataset: RLLMDataset):
    assert dataset.num_classes == 2
    # Length should match underlying dataset
    assert len(dataset) == len(dataset.ds)
    assert len(dataset) == 5
    # We didn't tokenize yet
    assert not dataset.is_tokenized


def test_tokenization_and_subset(dataset: RLLMDataset, tokenizer):
    tokenized_dataset = dataset.tokenize(tokenizer)
    # We should have tokenized the dataset
    assert tokenized_dataset.is_tokenized

    # Check that the tokenized dataset has the right keys
    assert {"input_ids", "attention_mask"} <= set(tokenized_dataset.ds[0].keys())
    smaller_dataset = tokenized_dataset.get_random_subset(2)
    assert len(smaller_dataset) == 2
    assert smaller_dataset.is_tokenized


def test_ground_truth_label_fn(dataset: RLLMDataset):
    """Test the ground_truth_label_fn works as expected.

    We use the config_name='pos' PasswordMatch dataset as a test case, since
    all of its labels are 1 so it's easy to flip the label.
    """
    # Test the ground truth label function
    example = dataset.ds[0]
    assert example["clf_label"] == 1
    text = example["text"]
    chunks = example["chunked_text"][:]  # Copy the list to avoid mutation
    chunks[1] = "some_other_word"
    attacked_text = "".join(chunks)

    # The ground truth label function should return 1 for the original text and 0
    # for the attacked text.
    assert dataset.ground_truth_label_fn(text, example["clf_label"]) == 1
    assert dataset.ground_truth_label_fn(attacked_text, example["clf_label"]) == 0


def test_with_attacked_text(dataset: RLLMDataset, tokenizer):
    attacked_texts = []
    for example in dataset.ds:
        assert isinstance(example, dict)
        assert example["clf_label"] == 1
        chunks = example["chunked_text"][:]
        chunks[1] = "some_other_word"
        attacked_text = "".join(chunks)
        attacked_texts.append(attacked_text)

    attacked_dataset = dataset.with_attacked_text(attacked_texts)
    assert len(attacked_dataset) == len(dataset)
    # Check that the attacked dataset has the original text in it
    assert attacked_dataset.ds["text"] == dataset.ds["text"]
    # Check that the labels have been flipped properly
    assert all(
        [ex["attacked_clf_label"] == 0 for ex in attacked_dataset.ds]  # type: ignore
    )
    assert all([ex["clf_label"] == 1 for ex in attacked_dataset.ds])  # type: ignore

    # Test 'as_adversarial_examples'
    with pytest.raises(ValueError):
        # Dataset has to be tokenized first
        adv_dataset = attacked_dataset.as_adversarial_examples()

    adv_dataset = attacked_dataset.tokenize(tokenizer).as_adversarial_examples()
    assert len(adv_dataset) == len(dataset)
    assert adv_dataset.ds["text"] == attacked_dataset.ds["attacked_text"]
    assert all([ex["clf_label"] == 0 for ex in adv_dataset.ds])  # type: ignore
    # Check that the 'attacked_text' and 'attacked_clf_label' columns are gone
    assert "attacked_text" not in adv_dataset.ds
    assert "attacked_clf_label" not in adv_dataset.ds


def test_for_hf_trainer(dataset: RLLMDataset, tokenizer):
    with pytest.raises(AssertionError):
        dataset.for_hf_trainer()

    tokenized_dataset = dataset.tokenize(tokenizer)
    hf_ds = tokenized_dataset.for_hf_trainer()
    assert len(hf_ds) == len(tokenized_dataset)
    # Check that the features are correct
    assert "input_ids" in hf_ds.features
    assert "attention_mask" in hf_ds.features
    assert "label" in hf_ds.features
    assert "clf_label" not in hf_ds.features
    assert "chunked_text" not in hf_ds.features
