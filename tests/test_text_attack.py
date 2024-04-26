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


def test_preprocess_example_overwritable(unprocessed_example: dict[str, str]):
    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.OVERWRITABLE, ChunkType.IMMUTABLE
    )
    num_modifiable_words_per_chunk = 1

    def ground_truth_label_fn(text, label):
        return 1 - label

    # We have to get this before we preprocess the example
    # because it's modified in place.
    expected_original_text = unprocessed_example["text"]
    processed_example = _preprocess_example(
        example=unprocessed_example,
        modifiable_chunk_spec=modifiable_chunk_spec,
        num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
        ground_truth_label_fn=ground_truth_label_fn,
    )
    # text should be saved as original_text
    assert processed_example["original_text"] == expected_original_text
    # label should be flipped
    assert processed_example["original_label"] != unprocessed_example["clf_label"]
    expected_text = "Does this sentence contain any letters?\nSentence:  special_modifiable_word \nAnswer:"  # noqa: E501
    assert processed_example["text"] == expected_text


def test_preprocess_example_perturbable(unprocessed_example: dict[str, str]):
    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.PERTURBABLE, ChunkType.IMMUTABLE
    )
    num_modifiable_words_per_chunk: int | None = 1

    def ground_truth_label_fn(text, label):
        return 1 - label

    # We have to get this before we preprocess the example
    # because it's modified in place.
    expected_original_text = unprocessed_example["text"]

    with pytest.raises(ValueError) as value_error:
        _ = _preprocess_example(
            example=unprocessed_example,
            modifiable_chunk_spec=modifiable_chunk_spec,
            num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
            ground_truth_label_fn=ground_truth_label_fn,
        )
    # If we have a PERTURBABLE chunk with num_modifiable_word_per_chunk > 0,
    # we should raise an error.
    assert "should be no PERTURBABLE chunks" in str(value_error)

    num_modifiable_words_per_chunk = None
    processed_example = _preprocess_example(
        example=unprocessed_example,
        modifiable_chunk_spec=modifiable_chunk_spec,
        num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
        ground_truth_label_fn=ground_truth_label_fn,
    )
    # text should be saved as original_text
    assert processed_example["original_text"] == expected_original_text
    # label should not be flipped because we don't change the text
    assert processed_example["original_label"] == unprocessed_example["clf_label"]
    assert processed_example["text"] == expected_original_text
