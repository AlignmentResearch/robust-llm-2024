from unittest.mock import patch

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from robust_llm.utils import (
    BalancedSampler,
    is_correctly_padded,
    nested_list_to_tuple,
    remove_directory,
)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "test_file.txt").write_text("Test content")
    yield test_dir


def test_remove_directory_success(temp_directory):
    """Test successful removal of a directory."""
    remove_directory(str(temp_directory))
    assert not temp_directory.exists()


def test_remove_directory_not_exists():
    """Test removal of a non-existent directory."""
    with pytest.raises(FileNotFoundError):
        remove_directory("/path/to/nonexistent/directory", retries=1)


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_retry(mock_sleep, mock_rmtree, temp_directory):
    """Test retrying removal on OSError."""
    mock_rmtree.side_effect = [OSError("Test error"), None]

    remove_directory(str(temp_directory))

    assert mock_rmtree.call_count == 2
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(1)


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_max_retries(mock_sleep, mock_rmtree, temp_directory):
    """Test maximum retries reached."""
    mock_rmtree.side_effect = OSError("Test error")

    with pytest.raises(OSError, match="Test error"):
        remove_directory(str(temp_directory), retries=4)

    assert mock_rmtree.call_count == 5
    assert mock_sleep.call_count == 4


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_exponential_backoff(mock_sleep, mock_rmtree, temp_directory):
    """Test exponential backoff in sleep times."""
    mock_rmtree.side_effect = [OSError("Test error")] * 4 + [None]

    remove_directory(str(temp_directory))

    assert mock_rmtree.call_count == 5
    assert mock_sleep.call_count == 4


def test_remove_directory_permission_error(temp_directory):
    """Test handling of PermissionError."""
    with patch("shutil.rmtree", side_effect=PermissionError("Permission denied")):
        with pytest.raises(PermissionError, match="Permission denied"):
            remove_directory(str(temp_directory), retries=0)


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.parametrize("regular_data_size", [1, 2, 3])
@pytest.mark.parametrize("adversarial_data_size", [1, 2, 3])
def test_balanced_sampler(regular_data_size: int, adversarial_data_size: int):
    regular_data = [0] * regular_data_size
    adversarial_data = [1] * adversarial_data_size
    balanced_sampler = BalancedSampler(regular_data, adversarial_data)

    assert len(balanced_sampler) == 2 * regular_data_size

    indices_for_regular_data = []

    for i, idx in enumerate(balanced_sampler):
        if i % 2 == 0:
            assert idx < regular_data_size
            indices_for_regular_data.append(idx)
        else:
            assert regular_data_size <= idx < regular_data_size + adversarial_data_size

    # Ensure all regular data points were sampled exactly once
    assert list(sorted(indices_for_regular_data)) == list(range(regular_data_size))


@given(text1=st.text(), text2=st.text())
def test_is_correctly_padded_true(tokenizer, text1: str, text2: str):
    """Test the `is_correctly_padded` function.

    We do this by tokenizing some input texts and
    checking that the returned masks pass the test.
    """
    # If both texts are empty then the mask is empty.
    assume(text1 != "" and text2 != "")
    texts = [text1, text2]
    padding_side = "right"
    tokenizer.padding_side = padding_side
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    masks = tokenized["attention_mask"]
    for mask in masks:
        assert is_correctly_padded(mask, padding_side)

    padding_side = "left"
    tokenizer.padding_side = padding_side
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    masks = tokenized["attention_mask"]
    for mask in masks:
        assert is_correctly_padded(mask, padding_side)


def test_is_correctly_padded_false():
    left_mask = torch.tensor([0, 1, 1, 1, 1, 1])
    right_mask = torch.tensor([1, 1, 1, 1, 1, 0])
    bad_mask = torch.tensor([1, 1, 0, 0, 0, 1, 1])

    assert not is_correctly_padded(left_mask, "right")
    assert not is_correctly_padded(right_mask, "left")
    assert not is_correctly_padded(bad_mask, "right")
    assert not is_correctly_padded(bad_mask, "left")


def test_nested_list_to_tuple():
    nested = [[1, 2], [3, 4]]
    assert nested_list_to_tuple(nested) == ((1, 2), (3, 4))
