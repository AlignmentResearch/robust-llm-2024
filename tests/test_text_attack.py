from unittest.mock import MagicMock

import pytest

from robust_llm.attacks.text_attack.text_attack import _preprocess_example
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


@pytest.fixture
def unprocessed_example():
    chunk_1 = "Does this sentence contain any letters?\nSentence: "
    chunk_2 = "The quick brown fox jumps over the lazy dog."
    chunk_3 = "\nAnswer:"
    chunked_text = [chunk_1, chunk_2, chunk_3]
    text = "".join(chunked_text)
    clf_label = 1
    return {"text": text, "chunked_text": chunked_text, "clf_label": clf_label}


@pytest.fixture
def mock_dataset():
    def mock_update_example_based_on_text(example, column_prefix=""):
        example = example.copy()
        example[f"{column_prefix}clf_label"] = 1 - example[f"{column_prefix}clf_label"]
        return example

    dataset = MagicMock()
    dataset.update_example_based_on_text.side_effect = mock_update_example_based_on_text
    return dataset


def test_preprocess_example_overwritable(
    unprocessed_example: dict[str, str], mock_dataset
):
    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.OVERWRITABLE, ChunkType.IMMUTABLE
    )
    mock_dataset.modifiable_chunk_spec = modifiable_chunk_spec
    num_modifiable_words_per_chunk = 1

    processed_example = _preprocess_example(
        example=unprocessed_example,
        dataset=mock_dataset,
        num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
    )
    expected_text = "Does this sentence contain any letters?\nSentence:  special_modifiable_word \nAnswer:"  # noqa: E501
    assert processed_example["text"] == expected_text


def test_preprocess_example_perturbable(
    unprocessed_example: dict[str, str], mock_dataset
):
    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.PERTURBABLE, ChunkType.IMMUTABLE
    )
    mock_dataset.modifiable_chunk_spec = modifiable_chunk_spec

    num_modifiable_words_per_chunk: int | None = 1

    # We have to get expected_original_text before we preprocess the example
    # because it's modified in place.
    expected_original_text = unprocessed_example["text"]

    with pytest.raises(ValueError) as value_error:
        _ = _preprocess_example(
            example=unprocessed_example,
            dataset=mock_dataset,
            num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
        )
    # If we have a PERTURBABLE chunk with num_modifiable_word_per_chunk > 0,
    # we should raise an error.
    assert "should be no PERTURBABLE chunks" in str(value_error)

    num_modifiable_words_per_chunk = None
    processed_example = _preprocess_example(
        example=unprocessed_example,
        dataset=mock_dataset,
        num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
    )
    assert processed_example["text"] == expected_original_text
